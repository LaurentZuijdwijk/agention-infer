//! CPU-orchestrated forward pass: `run()` and every mixer's CPU implementation.
//! Dispatches individual matmuls to the GPU backend when available
//! (`matmul_into` and friends), but always round-trips through CPU between
//! them — see `gpu_resident.rs` for the alternative that keeps the whole
//! layer stack on GPU.

use super::state::trace_cpu;
use super::types::{GatedDeltaNet, Mixer, ShortConv};
use super::{InferenceState, LlamaModel};
use crate::error::{GgufError, Result};
use crate::model::KvCache;
use crate::ops::AnyBackend;
use rayon::prelude::*;

impl<'a> LlamaModel<'a> {
    /// RMS normalization into a caller-provided buffer using a cached weight vector.
    /// `out[i] = (x[i] / rms) * weight[i]`.
    fn rms_norm_into(&self, x: &[f32], weight: &[f32], out: &mut [f32]) {
        let eps = self.cfg.layer_norm_rms_epsilon;
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + eps).sqrt();
        for ((&xi, &wi), o) in x.iter().zip(weight.iter()).zip(out.iter_mut()) {
            *o = wi * xi / rms;
        }
    }

    /// Per-head RMSNorm over `head_dim`, applied in place to each of the
    /// `n_heads` contiguous head slices of `x`. Used for Qwen3 QK-norm.
    fn qk_norm(&self, x: &mut [f32], weight_name: &str, n_heads: usize) -> Result<()> {
        let weight = self.weights.dequant_1d(weight_name)?;
        let head_dim = self.cfg.head_dim as usize;
        let eps = self.cfg.layer_norm_rms_epsilon;
        debug_assert_eq!(weight.len(), head_dim);

        for h in 0..n_heads {
            let head = &mut x[h * head_dim..h * head_dim + head_dim];
            let sum_sq: f32 = head.iter().map(|v| v * v).sum();
            let rms = (sum_sq / head_dim as f32 + eps).sqrt();
            for (d, xi) in head.iter_mut().enumerate() {
                *xi = weight[d] * *xi / rms;
            }
        }
        Ok(())
    }

    /// Rotary position embedding applied per-head. `freq`'s (sin, cos) pair
    /// depends only on `(pos, d)`, not the head — computed once per `d` here
    /// and reused across every Q/K head, instead of recomputing `theta.powf`
    /// and `sin_cos` redundantly `n_heads`/`n_kv_heads` times per `d`.
    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize) {
        let head_dim = self.cfg.head_dim as usize;
        let n_heads = self.cfg.head_count as usize;
        let n_kv_heads = self.cfg.head_count_kv as usize;
        let theta = self.cfg.rope_freq_base;
        let half = head_dim / 2;

        let table: Vec<(f32, f32)> = (0..half)
            .map(|d| {
                let freq = pos as f32 / theta.powf(2.0 * d as f32 / head_dim as f32);
                freq.sin_cos()
            })
            .collect();

        for h in 0..n_heads {
            let start = h * head_dim;
            for (d, &(sin_val, cos_val)) in table.iter().enumerate() {
                let x0 = q[start + d];
                let x1 = q[start + d + half];
                q[start + d] = x0 * cos_val - x1 * sin_val;
                q[start + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }

        for h in 0..n_kv_heads {
            let start = h * head_dim;
            for (d, &(sin_val, cos_val)) in table.iter().enumerate() {
                let x0 = k[start + d];
                let x1 = k[start + d + half];
                k[start + d] = x0 * cos_val - x1 * sin_val;
                k[start + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    /// Naive attention into a caller-provided buffer: computes full attention
    /// scores for all cached positions and writes the attended output for one
    /// token into `out` (length `n_heads * head_dim`). `scores` is a reused
    /// scratchpad that must hold at least `k_all.len()` values.
    fn attention_into(
        &self,
        q: &[f32],
        k_all: &[Vec<f32>],
        v_all: &[Vec<f32>],
        out: &mut [f32],
        scores: &mut [f32],
    ) {
        let head_dim = self.cfg.head_dim as usize;
        let n_heads = self.cfg.head_count as usize;
        let group_size = self.cfg.gqa_group_size() as usize;
        let seq_len = k_all.len(); // number of cached positions

        // out is reused across tokens, so clear the region we accumulate into.
        for o in out.iter_mut() {
            *o = 0.0;
        }

        for h in 0..n_heads {
            let kv_head = h / group_size; // which KV head does this Q head attend to
            let q_start = h * head_dim;
            let q_head = &q[q_start..q_start + head_dim];

            // Attention scores for this head against all cached positions.
            let scores = &mut scores[..seq_len];
            for t in 0..seq_len {
                let k_start = kv_head * head_dim;
                let k_head = &k_all[t][k_start..k_start + head_dim];
                let dot: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                scores[t] = dot / (head_dim as f32).sqrt();
            }

            // Softmax
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
            }
            let sum: f32 = scores.iter().sum();
            for s in scores.iter_mut() {
                *s /= sum;
            }

            // Weighted sum of V
            for t in 0..seq_len {
                let v_start = kv_head * head_dim;
                let v_head = &v_all[t][v_start..v_start + head_dim];
                for d in 0..head_dim {
                    out[h * head_dim + d] += scores[t] * v_head[d];
                }
            }
        }
    }

    /// CPU fallback FFN gate+up matmul: computes both `gate = W_gate * x` and
    /// `up = W_up * x` in one pass over the weight data, then applies silu.
    /// Avoids reading `x` twice and spawning two rayon thread pools. Used by
    /// `matmul_ffn_into` when the GPU-chained path isn't available.
    fn ffn_gate_up(
        &self,
        gate_name: &str,
        up_name: &str,
        x: &[f32],
        ffn_act: &mut [f32],
    ) -> Result<()> {
        let gate_tensor = self
            .weights
            .get(gate_name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {gate_name}")))?;
        let up_tensor = self
            .weights
            .get(up_name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {up_name}")))?;

        let out_dim = gate_tensor.dims[1] as usize;
        debug_assert_eq!(ffn_act.len(), out_dim);
        debug_assert_eq!(gate_tensor.ggml_type, up_tensor.ggml_type);

        let data = self.raw_data;

        ffn_act[..out_dim]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row_idx, o)| -> Result<()> {
                let g = crate::quant::dot_row(gate_tensor, data, row_idx, x)?;
                let u = crate::quant::dot_row(up_tensor, data, row_idx, x)?;
                let silu = g / (1.0 + (-g).exp());
                *o = silu * u;
                Ok(())
            })
    }

    /// Matrix-vector product, dispatching to GPU backend when available.
    /// Falls back to CPU (rayon-parallel fused dequant+dot via WeightMap) otherwise.
    fn matmul_into(&self, name: &str, x: &[f32], out: &mut [f32]) -> Result<()> {
        let crosscheck = std::env::var("GGUF_CROSSCHECK_GPU").is_ok();

        // Try pre-uploaded GPU tensor first (already resident, no re-upload)
        #[cfg(feature = "wgpu")]
        if let Some(ref backend) = self.backend {
            if backend.name() != "cpu" {
                if let Some(handle) = self.gpu_tensors.get(name) {
                    use crate::ops::AnyBackend;
                    match backend {
                        AnyBackend::Wgpu(b) => {
                            b.matmul_dequant_preloaded(handle, x, out)?;
                            if crosscheck {
                                let mut cpu_out = vec![0f32; out.len()];
                                self.weights.matmul_into(name, x, &mut cpu_out)?;
                                let max_err = out.iter().zip(&cpu_out).map(|(g, c)| (g - c).abs()).fold(0f32, f32::max);
                                if max_err > 0.1 {
                                    eprintln!("[CROSSCHECK FAIL] {} out_dim={} max_err={:.6e}", name, out.len(), max_err);
                                    eprintln!("  gpu[0..4]={:?} cpu[0..4]={:?}", &out[..4.min(out.len())], &cpu_out[..4.min(cpu_out.len())]);
                                } else {
                                    eprintln!("[CROSSCHECK OK]   {} out_dim={} max_err={:.6e}", name, out.len(), max_err);
                                }
                            }
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }
        }

        // Fallback: ad-hoc GPU dispatch (packs + uploads on every call)
        if let Some(ref backend) = self.backend {
            if backend.name() != "cpu" {
                let tensor = self
                    .weights
                    .get(name)
                    .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {name}")))?;

                let out_dim: usize = if tensor.n_dims as usize == 1 {
                    1
                } else {
                    tensor.dims[1..].iter().product::<u64>() as usize
                };

                let row_data = &self.raw_data
                    [tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.byte_size()];

                if crate::ops::GPU_DEQUANT_DTYPES.contains(&tensor.ggml_type) {
                    backend.matmul_dequant(tensor.ggml_type, row_data, x, &mut out[..out_dim])?;
                    return Ok(());
                }
            }
        }

        // CPU fallback (rayon-parallel, fast)
        self.weights.matmul_into(name, x, out)
    }

    /// Q/K/V projection, batched into a single GPU round-trip when all three
    /// weights are GPU-resident (they all read the same input `x`). Falls
    /// back to three independent `matmul_into` calls otherwise.
    fn matmul_qkv_into(
        &self,
        wq: &str,
        wk: &str,
        wv: &str,
        x: &[f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<()> {
        #[cfg(feature = "wgpu")]
        if let Some(AnyBackend::Wgpu(b)) = &self.backend {
            if let (Some(hq), Some(hk), Some(hv)) = (
                self.gpu_tensors.get(wq),
                self.gpu_tensors.get(wk),
                self.gpu_tensors.get(wv),
            ) {
                return b.matmul_dequant_qkv(hq, hk, hv, x, q, k, v);
            }
        }

        self.matmul_into(wq, x, q)?;
        self.matmul_into(wk, x, k)?;
        self.matmul_into(wv, x, v)
    }

    /// Gated DeltaNet's initial projections (`wqkv`, `wgate`, `ssm_beta`,
    /// `ssm_alpha`) all read the same input `x`, batched into a single GPU
    /// round-trip when all four weights are GPU-resident (see
    /// `matmul_qkv_into` above for the same pattern with three tensors).
    /// Falls back to four independent `matmul_into` calls otherwise.
    #[allow(clippy::too_many_arguments)]
    fn matmul_gdn_proj_into(
        &self,
        gdn: &GatedDeltaNet,
        x: &[f32],
        qkv: &mut [f32],
        z: &mut [f32],
        beta: &mut [f32],
        alpha: &mut [f32],
    ) -> Result<()> {
        #[cfg(feature = "wgpu")]
        if let Some(AnyBackend::Wgpu(b)) = &self.backend {
            if let (Some(hqkv), Some(hz), Some(hbeta), Some(halpha)) = (
                self.gpu_tensors.get(gdn.wqkv.as_str()),
                self.gpu_tensors.get(gdn.wgate.as_str()),
                self.gpu_tensors.get(gdn.ssm_beta.as_str()),
                self.gpu_tensors.get(gdn.ssm_alpha.as_str()),
            ) {
                return b.matmul_dequant_multi(&[hqkv, hz, hbeta, halpha], x, &mut [qkv, z, beta, alpha]);
            }
        }

        self.matmul_into(&gdn.wqkv, x, qkv)?;
        self.matmul_into(&gdn.wgate, x, z)?;
        self.matmul_into(&gdn.ssm_beta, x, beta)?;
        self.matmul_into(&gdn.ssm_alpha, x, alpha)
    }

    /// FFN block (`gate`/`up`/SiLU-combine/`down`), chained entirely on GPU
    /// with a single readback when all three weights are GPU-resident (no
    /// CPU-side dependency sits between them, unlike attention). Falls back
    /// to `ffn_gate_up` + `matmul_into` otherwise. `ffn_act` is scratch space
    /// used only by the CPU fallback path.
    fn matmul_ffn_into(
        &self,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
        x: &[f32],
        ffn_act: &mut [f32],
        out: &mut [f32],
    ) -> Result<()> {
        #[cfg(feature = "wgpu")]
        if let Some(AnyBackend::Wgpu(b)) = &self.backend {
            if let (Some(hg), Some(hu), Some(hd)) = (
                self.gpu_tensors.get(gate_name),
                self.gpu_tensors.get(up_name),
                self.gpu_tensors.get(down_name),
            ) {
                return b.matmul_dequant_ffn(hg, hu, hd, x, out);
            }
        }

        self.ffn_gate_up(gate_name, up_name, x, ffn_act)?;
        self.matmul_into(down_name, ffn_act, out)
    }

    /// LFM2 short-convolution mixer for layer `layer_idx`. Reads the operator-
    /// normed activations from `state.xn` and writes the mixer output into
    /// `state.proj` (ready for the residual add). Updates the layer's recurrent
    /// conv state in place.
    ///
    /// Pipeline (matching LiquidAI's `Lfm2ShortConv`):
    ///   `(B, C, x) = split(in_proj(xn))`
    ///   `Bx = B * x`
    ///   `conv_out = depthwise_causal_conv1d(Bx)`   (kernel width `l_cache`)
    ///   `y = C * conv_out`
    ///   `out = out_proj(y)`
    fn short_conv(
        &self,
        layer_idx: usize,
        sc: &ShortConv,
        state: &mut InferenceState,
    ) -> Result<()> {
        // in_proj: xn[d] → bcx[3d], split into B, C, x gates.
        self.matmul_into(&sc.in_proj, &state.xn, &mut state.bcx)?;
        self.short_conv_gate_and_conv(layer_idx, sc, state);
        // out_proj: conv_out[d] → proj[d].
        self.matmul_into(&sc.out_proj, &state.conv_out, &mut state.proj)
    }

    /// The elementwise-gate + depthwise-causal-conv math between `in_proj`
    /// and `out_proj`: reads `state.bcx` (already populated by `in_proj`),
    /// writes `state.conv_out`, and advances the layer's recurrent conv
    /// history. Small (`d` channels, `l_cache` taps) and inherently
    /// sequential (per-channel recurrence), so this stays on CPU — shared by
    /// both `short_conv` (CPU/ad-hoc-GPU matmul dispatch) and
    /// `run_gpu_resident` (GPU-resident matmul dispatch).
    fn short_conv_gate_and_conv(&self, layer_idx: usize, sc: &ShortConv, state: &mut InferenceState) {
        let d = self.cfg.embedding_length as usize;
        let l = self.l_cache;

        let (b, rest) = state.bcx.split_at(d);
        let (c, xg) = rest.split_at(d);

        // Bx = B * x (elementwise gate).
        for j in 0..d {
            state.bx[j] = b[j] * xg[j];
        }

        // Depthwise causal conv1d. For channel `ch`, tap `k` aligns with the
        // input at time `t - (l-1) + k`: taps 0..l-2 come from the recurrent
        // history (oldest first), tap l-1 is the current `bx`.
        let history = &state.conv_state[layer_idx];
        let w = &sc.conv_weight; // [ch][l], contiguous taps per channel
        for ch in 0..d {
            let tap = &w[ch * l..ch * l + l];
            let mut acc = 0.0f32;
            for k in 0..l - 1 {
                acc += tap[k] * history[k * d + ch];
            }
            acc += tap[l - 1] * state.bx[ch];
            // y = C * conv_out, fused into conv_out.
            state.conv_out[ch] = c[ch] * acc;
        }

        // Advance the recurrent state: drop the oldest slot, append current bx.
        let history = &mut state.conv_state[layer_idx];
        for k in 0..l.saturating_sub(2) {
            for ch in 0..d {
                history[k * d + ch] = history[(k + 1) * d + ch];
            }
        }
        if l >= 2 {
            let last = (l - 2) * d;
            history[last..last + d].copy_from_slice(&state.bx);
        }
    }

    /// Partial rotary embedding: like `rope`, but only the first `self.n_rot`
    /// dims of each `head_dim`-sized head are rotated (Qwen3.5's `GatedAttention`
    /// layers leave the remaining "NoPE" tail dims untouched).
    fn rope_partial(&self, q: &mut [f32], k: &mut [f32], pos: usize) {
        let head_dim = self.cfg.head_dim as usize;
        let n_heads = self.cfg.head_count as usize;
        let n_kv_heads = self.cfg.head_count_kv as usize;
        let theta = self.cfg.rope_freq_base;
        let n_rot = self.n_rot;
        let half = n_rot / 2;

        let table: Vec<(f32, f32)> = (0..half)
            .map(|d| {
                let freq = pos as f32 / theta.powf(2.0 * d as f32 / n_rot as f32);
                freq.sin_cos()
            })
            .collect();

        for h in 0..n_heads {
            let start = h * head_dim;
            for (d, &(sin_val, cos_val)) in table.iter().enumerate() {
                let x0 = q[start + d];
                let x1 = q[start + d + half];
                q[start + d] = x0 * cos_val - x1 * sin_val;
                q[start + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }

        for h in 0..n_kv_heads {
            let start = h * head_dim;
            for (d, &(sin_val, cos_val)) in table.iter().enumerate() {
                let x0 = k[start + d];
                let x1 = k[start + d + half];
                k[start + d] = x0 * cos_val - x1 * sin_val;
                k[start + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    /// Qwen3.5's fused-QG full-attention layer: `wqg` projects to `[Q(head_dim)
    /// | gate(head_dim)]` per head; the gate is applied (sigmoid) to the
    /// attention output before `wo`. RoPE is partial (see `rope_partial`).
    /// Writes the mixer output into `state.proj`.
    #[allow(clippy::too_many_arguments)]
    fn gated_attention(
        &self,
        layer_idx: usize,
        wqg: &str,
        wk: &str,
        wv: &str,
        wo: &str,
        q_norm: &str,
        k_norm: &str,
        pos: usize,
        kv_cache: &mut KvCache,
        state: &mut InferenceState,
    ) -> Result<()> {
        let head_dim = self.cfg.head_dim as usize;
        let n_heads = self.cfg.head_count as usize;
        let n_kv_heads = self.cfg.head_count_kv as usize;

        // Fused Q+gate, K, and V projections all read `state.xn` — batched
        // into a single GPU round-trip (see `matmul_qkv_into`), then split
        // the interleaved-per-head layout `[Q(head_dim) | gate(head_dim)]`
        // into contiguous Q and gate buffers.
        self.matmul_qkv_into(
            wqg,
            wk,
            wv,
            &state.xn,
            &mut state.ga_qg_raw,
            &mut state.k,
            &mut state.v,
        )?;
        for h in 0..n_heads {
            let src = h * 2 * head_dim;
            state.q[h * head_dim..h * head_dim + head_dim]
                .copy_from_slice(&state.ga_qg_raw[src..src + head_dim]);
            state.ga_gate[h * head_dim..h * head_dim + head_dim]
                .copy_from_slice(&state.ga_qg_raw[src + head_dim..src + 2 * head_dim]);
        }

        self.qk_norm(&mut state.q, q_norm, n_heads)?;
        self.qk_norm(&mut state.k, k_norm, n_kv_heads)?;

        self.rope_partial(&mut state.q, &mut state.k, pos);

        kv_cache.write(layer_idx, pos, &state.k, &state.v);
        let (k_all, v_all) = kv_cache.read_up_to(layer_idx, pos);
        self.attention_into(
            &state.q,
            k_all,
            v_all,
            &mut state.attn_out,
            &mut state.scores,
        );

        // Output gate: attn_out *= sigmoid(gate), applied before wo.
        for (o, &g) in state.attn_out.iter_mut().zip(state.ga_gate.iter()) {
            *o *= 1.0 / (1.0 + (-g).exp());
        }

        self.matmul_into(wo, &state.attn_out, &mut state.proj)
    }

    /// Qwen3.5's linear-attention layer ("Gated DeltaNet"): fused QKV → causal
    /// depthwise conv (SiLU) → per-head L2-norm on Q/K → tile-repeat Q/K from
    /// `n_k_heads` to `n_v_heads` → per-token delta-rule recurrence with a
    /// learned scalar decay and write gate → gated RMSNorm → `ssm_out`.
    /// Writes the mixer output into `state.proj`.
    ///
    /// Per-head recurrence (state `S` is `head_k_dim x head_v_dim`, persistent
    /// across positions, matching llama.cpp's `build_delta_net_autoregressive`):
    ///   `v_pred[b] = sum_a S[a,b] * k[a]`
    ///   `delta[b]  = beta * (v[b] - v_pred[b])`
    ///   `S[a,b]    = decay * S[a,b] + k[a] * delta[b]`
    ///   `out[b]    = sum_a (q[a] * scale) * S[a,b]`,  `scale = 1/sqrt(head_k_dim)`
    fn gated_delta_net(
        &self,
        layer_idx: usize,
        gdn: &GatedDeltaNet,
        dims: &super::types::GdnDims,
        state: &mut InferenceState,
    ) -> Result<()> {
        self.matmul_gdn_proj_into(
            gdn,
            &state.xn,
            &mut state.gdn_qkv,
            &mut state.gdn_z,
            &mut state.gdn_beta_raw,
            &mut state.gdn_alpha_raw,
        )?;

        let n_v_heads = dims.n_v_heads;
        let n_k_heads = dims.n_k_heads;
        let head_k_dim = dims.head_k_dim;
        let head_v_dim = dims.head_v_dim;
        let key_dim = dims.key_dim;
        let value_dim = dims.value_dim;
        let eps = self.cfg.layer_norm_rms_epsilon;

        // beta = sigmoid(beta_raw); decay = exp(ssm_a * softplus(alpha_raw + dt_bias)).
        // Overwritten in place — the raw logits aren't needed again this token.
        for h in 0..n_v_heads {
            state.gdn_beta_raw[h] = 1.0 / (1.0 + (-state.gdn_beta_raw[h]).exp());
            let x = state.gdn_alpha_raw[h] + gdn.ssm_dt_bias[h];
            let softplus = if x > 20.0 { x } else { (1.0f32 + x.exp()).ln() };
            state.gdn_alpha_raw[h] = (gdn.ssm_a[h] * softplus).exp();
        }

        causal_depthwise_conv(
            &mut state.gdn_conv_state[layer_idx],
            &gdn.conv_weight,
            dims.conv_dim,
            dims.d_conv,
            &state.gdn_qkv,
            &mut state.gdn_conv_out,
            |acc| acc / (1.0 + (-acc).exp()), // SiLU
        );

        // L2-normalize each Q/K head (dim head_k_dim) in place.
        for h in 0..n_k_heads {
            for base in [h * head_k_dim, key_dim + h * head_k_dim] {
                let seg = &mut state.gdn_conv_out[base..base + head_k_dim];
                let sum_sq: f32 = seg.iter().map(|v| v * v).sum();
                let norm = (sum_sq + eps).sqrt();
                for v in seg.iter_mut() {
                    *v /= norm;
                }
            }
        }

        // Per-(value-)head delta-rule recurrence. Q/K heads are tile-repeated
        // (cyclic) up to the value-head count, matching `ggml_repeat_4d`.
        let scale = 1.0 / (head_k_dim as f32).sqrt();
        for h in 0..n_v_heads {
            let kh = h % n_k_heads;
            let q_h_start = kh * head_k_dim;
            let k_h_start = key_dim + kh * head_k_dim;
            let v_h_start = 2 * key_dim + h * head_v_dim;

            let s = &mut state.gdn_recurrent_state[layer_idx]
                [h * head_k_dim * head_v_dim..(h + 1) * head_k_dim * head_v_dim];

            for b in 0..head_v_dim {
                let mut acc = 0.0f32;
                for a in 0..head_k_dim {
                    acc += s[a * head_v_dim + b] * state.gdn_conv_out[k_h_start + a];
                }
                state.gdn_vpred[b] = acc;
            }
            for b in 0..head_v_dim {
                state.gdn_delta[b] = state.gdn_beta_raw[h]
                    * (state.gdn_conv_out[v_h_start + b] - state.gdn_vpred[b]);
            }
            let decay = state.gdn_alpha_raw[h];
            for a in 0..head_k_dim {
                let k_val = state.gdn_conv_out[k_h_start + a];
                for b in 0..head_v_dim {
                    let idx = a * head_v_dim + b;
                    s[idx] = decay * s[idx] + k_val * state.gdn_delta[b];
                }
            }
            for b in 0..head_v_dim {
                let mut acc = 0.0f32;
                for a in 0..head_k_dim {
                    acc += scale * state.gdn_conv_out[q_h_start + a] * s[a * head_v_dim + b];
                }
                state.gdn_out[h * head_v_dim + b] = acc;
            }
        }

        // Gated RMSNorm per head: norm(out) * silu(z), in place.
        for h in 0..n_v_heads {
            let base = h * head_v_dim;
            let sum_sq: f32 = state.gdn_out[base..base + head_v_dim]
                .iter()
                .map(|v| v * v)
                .sum();
            let rms = (sum_sq / head_v_dim as f32 + eps).sqrt();
            for d in 0..head_v_dim {
                let normed = gdn.ssm_norm[d] * state.gdn_out[base + d] / rms;
                let g = state.gdn_z[base + d];
                let silu = g / (1.0 + (-g).exp());
                state.gdn_out[base + d] = normed * silu;
            }
        }

        self.matmul_into(&gdn.ssm_out, &state.gdn_out[..value_dim], &mut state.proj)
    }

    /// Forward pass against the pre-allocated `state` buffers.
    ///
    /// Takes `&self` (weights/config, read-only) and `&mut state` (activation
    /// scratch) as disjoint borrows, so every buffer is reused in place with no
    /// per-token heap allocation apart from the returned logits `Vec`.
    pub(crate) fn run(
        &self,
        token: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>> {
        let trace = trace_cpu::enabled();

        // 1. Embedding lookup → residual stream.
        let t0 = trace_cpu::now(trace);
        let embd = self.weights.dequant_row(&self.token_embd, token as usize)?;
        state.x.copy_from_slice(&embd);
        trace_cpu::add(&trace_cpu::T_EMBED, t0);

        // 2. Transformer layers.
        for (i, layer) in self.layers.iter().enumerate() {
            // --- Token mixer (attention or short-conv) ---
            // Operator prenorm → xn, then the mixer writes its output to proj.
            self.rms_norm_into(&state.x, &layer.attn_norm, &mut state.xn);

            let t0 = trace_cpu::now(trace);
            match &layer.mixer {
                Mixer::Attention {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    self.matmul_qkv_into(wq, wk, wv, &state.xn, &mut state.q, &mut state.k, &mut state.v)?;

                    // Qwen3/LFM2 QK-norm: per-head RMSNorm on Q and K before RoPE.
                    if let Some(q_norm) = q_norm {
                        self.qk_norm(&mut state.q, q_norm, self.cfg.head_count as usize)?;
                    }
                    if let Some(k_norm) = k_norm {
                        self.qk_norm(&mut state.k, k_norm, self.cfg.head_count_kv as usize)?;
                    }

                    self.rope(&mut state.q, &mut state.k, pos);

                    // Write K/V to cache, then read all cached K/V.
                    kv_cache.write(i, pos, &state.k, &state.v);
                    let (k_all, v_all) = kv_cache.read_up_to(i, pos);

                    // Attention into attn_out (scores as reused scratch).
                    self.attention_into(
                        &state.q,
                        k_all,
                        v_all,
                        &mut state.attn_out,
                        &mut state.scores,
                    );

                    // Output projection.
                    self.matmul_into(wo, &state.attn_out, &mut state.proj)?;
                }
                Mixer::ShortConv(sc) => {
                    self.short_conv(i, sc, state)?;
                }
                Mixer::GatedAttention {
                    wqg,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    self.gated_attention(i, wqg, wk, wv, wo, q_norm, k_norm, pos, kv_cache, state)?;
                }
                Mixer::GatedDeltaNet(gdn) => {
                    let dims = self
                        .gdn_dims
                        .expect("gdn_dims must be set when a GatedDeltaNet layer exists");
                    self.gated_delta_net(i, gdn, &dims, state)?;
                }
            }
            trace_cpu::add(
                match &layer.mixer {
                    Mixer::Attention { .. } => &trace_cpu::T_ATTN,
                    Mixer::ShortConv(_) => &trace_cpu::T_SHORTCONV,
                    Mixer::GatedAttention { .. } => &trace_cpu::T_GATED_ATTN,
                    Mixer::GatedDeltaNet(_) => &trace_cpu::T_GDN,
                },
                t0,
            );

            // Mixer residual.
            for j in 0..state.x.len() {
                state.x[j] += state.proj[j];
            }

            // --- FFN (SwiGLU) — fused gate+up+down ---
            self.rms_norm_into(&state.x, &layer.ffn_norm, &mut state.xn);

            let t0 = trace_cpu::now(trace);
            self.matmul_ffn_into(
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
                &state.xn,
                &mut state.ffn_act,
                &mut state.proj,
            )?;
            trace_cpu::add(&trace_cpu::T_FFN, t0);

            for j in 0..state.x.len() {
                state.x[j] += state.proj[j];
            }
        }

        // 3. Final norm + project to vocab.
        self.rms_norm_into(&state.x, &self.output_norm, &mut state.xn);
        let t0 = trace_cpu::now(trace);
        self.matmul_into(&self.lm_head, &state.xn, &mut state.logits)?;
        trace_cpu::add(&trace_cpu::T_LMHEAD, t0);
        if trace {
            trace_cpu::maybe_report();
        }

        Ok(state.logits.clone())
    }
}

/// Generic causal depthwise conv1d over `channels` channels with kernel width
/// `kernel`. `history` holds `(kernel-1) * channels` values (oldest first,
/// layout `[tap][channel]`) and is advanced in place — the recurrent state
/// that gives the conv its history across forward passes. `activation` is
/// applied to each channel's raw conv output before it's written to `output`.
///
/// Shared by Qwen3.5's Gated DeltaNet mixer. (LFM2's `short_conv_gate_and_conv`
/// keeps its own copy since it also fuses in an elementwise `B*x` gate before
/// the conv, which this generic version has no notion of.)
fn causal_depthwise_conv(
    history: &mut [f32],
    conv_weight: &[f32],
    channels: usize,
    kernel: usize,
    input: &[f32],
    output: &mut [f32],
    activation: impl Fn(f32) -> f32,
) {
    for ch in 0..channels {
        let tap = &conv_weight[ch * kernel..ch * kernel + kernel];
        let mut acc = 0.0f32;
        for k in 0..kernel - 1 {
            acc += tap[k] * history[k * channels + ch];
        }
        acc += tap[kernel - 1] * input[ch];
        output[ch] = activation(acc);
    }

    // Advance the recurrent history: drop the oldest slot, append current input.
    for k in 0..kernel.saturating_sub(2) {
        for ch in 0..channels {
            history[k * channels + ch] = history[(k + 1) * channels + ch];
        }
    }
    if kernel >= 2 {
        let last = (kernel - 2) * channels;
        history[last..last + channels].copy_from_slice(&input[..channels]);
    }
}
