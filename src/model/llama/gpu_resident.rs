//! Fully GPU-resident forward pass: pre-uploading weights/state to GPU, and
//! `run_gpu_resident` itself, which keeps the residual stream (and every
//! mixer's own recurrent state) on GPU as handles for the whole layer stack —
//! no CPU round-trip until the final logits read. See `cpu_path.rs` for the
//! per-matmul CPU-orchestrated alternative.

use crate::error::Result;
use crate::ops::AnyBackend;
use crate::model::llama::Mixer;
use super::LlamaModel;

impl<'a> LlamaModel<'a> {
    /// Pre-upload all GPU-dequantizable weight tensors to GPU.
    /// Called once after model creation, before first forward pass.
    #[cfg(feature = "wgpu")]
    pub fn pre_upload_gpu(&mut self) {
        use std::time::Instant;

        let wgpu = match &self.backend {
            Some(AnyBackend::Wgpu(b)) => b,
            _ => return,
        };

        // Collect all tensor names on the matmul hot path
        let mut to_upload: Vec<String> = Vec::new();
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Attention { wq, wk, wv, wo, .. } => {
                    to_upload.push(wq.clone());
                    to_upload.push(wk.clone());
                    to_upload.push(wv.clone());
                    to_upload.push(wo.clone());
                }
                Mixer::ShortConv(sc) => {
                    to_upload.push(sc.in_proj.clone());
                    to_upload.push(sc.out_proj.clone());
                }
                Mixer::GatedAttention { wqg, wk, wv, wo, .. } => {
                    to_upload.push(wqg.clone());
                    to_upload.push(wk.clone());
                    to_upload.push(wv.clone());
                    to_upload.push(wo.clone());
                }
                Mixer::GatedDeltaNet(gdn) => {
                    to_upload.push(gdn.wqkv.clone());
                    to_upload.push(gdn.wgate.clone());
                    to_upload.push(gdn.ssm_beta.clone());
                    to_upload.push(gdn.ssm_alpha.clone());
                    to_upload.push(gdn.ssm_out.clone());
                }
            }
            to_upload.push(layer.ffn_gate.clone());
            to_upload.push(layer.ffn_up.clone());
            to_upload.push(layer.ffn_down.clone());
        }
        to_upload.push(self.lm_head.clone());
        to_upload.push(self.token_embd.clone());

        eprintln!("  uploading {} tensors to GPU...", to_upload.len());
        let mut uploaded = 0usize;
        for name in &to_upload {
            if let Some(tensor) = self.weights.get(name) {
                if crate::ops::GPU_DEQUANT_DTYPES.contains(&tensor.ggml_type) {
                    let t0 = Instant::now();
                    let tensor_data = &self.raw_data[tensor.byte_offset as usize
                        ..tensor.byte_offset as usize + tensor.byte_size()];
                    let in_dim = tensor.dims[0] as usize;
                    let handle = wgpu.upload_weight(tensor.ggml_type, tensor_data, in_dim);
                    self.gpu_tensors.insert(name.clone(), handle);
                    uploaded += 1;
                    let mb = tensor.byte_size() as f64 / 1048576.0;
                    eprintln!(
                        "    [{uploaded}] {name} ({mb:.1} MB) in {:.2}s",
                        t0.elapsed().as_secs_f32()
                    );
                }
            }
        }
        eprintln!("  pre-uploaded {uploaded} GPU tensors");

        // Norm vectors are always f32 and already dequantized in memory
        // (see LayerWeights/output_norm) — upload as-is, no packing needed.
        // These feed `run_gpu_resident`'s fully GPU-resident layer chain.
        for (i, layer) in self.layers.iter().enumerate() {
            self.gpu_attn_norms
                .insert(i, wgpu.upload_activation(&layer.attn_norm));
            self.gpu_ffn_norms
                .insert(i, wgpu.upload_activation(&layer.ffn_norm));
        }
        self.gpu_output_norm = Some(wgpu.upload_activation(&self.output_norm));

        // Short-conv layers: static conv-tap weights + zero-initialized
        // persistent history, for the GPU-resident `launch_short_conv` path.
        let hist_len = self.l_cache.saturating_sub(1) * self.cfg.embedding_length as usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Mixer::ShortConv(sc) = &layer.mixer {
                self.gpu_conv_weights
                    .insert(i, wgpu.upload_activation(&sc.conv_weight));
                self.gpu_conv_history
                    .insert(i, wgpu.upload_activation(&vec![0.0f32; hist_len]));
            }
        }

        // Attention/GatedAttention layers' QK-norm weights (shared across
        // heads), for the GPU-resident attention path.
        let qk_norm_names: Vec<String> = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.mixer {
                Mixer::Attention { q_norm, k_norm, .. } => Some([q_norm.clone(), k_norm.clone()]),
                Mixer::GatedAttention { q_norm, k_norm, .. } => {
                    Some([Some(q_norm.clone()), Some(k_norm.clone())])
                }
                _ => None,
            })
            .flatten()
            .flatten()
            .collect();
        for name in qk_norm_names {
            if let Ok(w) = self.weights.dequant_1d(&name) {
                self.gpu_qk_norm.insert(name, wgpu.upload_activation(&w));
            }
        }

        // GatedDeltaNet layers: conv kernel, ssm_norm, ssm_a, ssm_dt_bias,
        // plus persistent conv history and recurrent state.
        if let Some(dims) = &self.gdn_dims {
            let state_size = dims.n_v_heads * dims.head_k_dim * dims.head_v_dim;
            let hist_len = dims.conv_dim * (dims.d_conv.saturating_sub(1));

            for (i, layer) in self.layers.iter().enumerate() {
                if let Mixer::GatedDeltaNet(gdn) = &layer.mixer {
                    // Conv kernel weights (already dequantized to f32)
                    self.gpu_gdn_conv_weights
                        .insert(i, wgpu.upload_activation(&gdn.conv_weight));
                    // ssm_norm (per-channel, shape [head_v_dim])
                    self.gpu_gdn_ssm_norm
                        .insert(i, wgpu.upload_activation(&gdn.ssm_norm));
                    // ssm_a (per head, shape [n_v_heads])
                    self.gpu_gdn_ssm_a
                        .insert(i, wgpu.upload_activation(&gdn.ssm_a));
                    // ssm_dt_bias (per head, shape [n_v_heads])
                    self.gpu_gdn_ssm_dt_bias
                        .insert(i, wgpu.upload_activation(&gdn.ssm_dt_bias));

                    // Persistent conv history: (d_conv-1) * conv_dim
                    self.gpu_gdn_conv_history
                        .insert(i, wgpu.upload_activation(&vec![0.0f32; hist_len]));
                    // Persistent recurrent state: n_v_heads * head_k_dim * head_v_dim
                    self.gpu_gdn_recurrent_state
                        .insert(i, wgpu.upload_activation(&vec![0.0f32; state_size]));
                }
            }
        }
    }

    /// Lazily allocate the GPU-resident KV cache (and shared attention-score
    /// scratch buffer), sized to `max_seq_len`, for every `Attention` layer
    /// that doesn't have one yet. Cheap no-op on repeat calls once allocated
    /// — called once per `forward()` before the resident path can run, since
    /// `max_seq_len` isn't known until the caller builds its `KvCache`.
    #[cfg(feature = "wgpu")]
    pub(crate) fn ensure_gpu_kv_cache(&mut self, max_seq_len: usize) {
        let wgpu = match &self.backend {
            Some(AnyBackend::Wgpu(b)) => b,
            _ => return,
        };
        if self.gpu_kv_max_seq == max_seq_len
            && self
                .layers
                .iter()
                .enumerate()
                .filter(|(_, l)| {
                    matches!(l.mixer, Mixer::Attention { .. } | Mixer::GatedAttention { .. })
                })
                .all(|(i, _)| self.gpu_kv_cache_k.contains_key(&i))
        {
            return;
        }

        let n_heads = self.cfg.head_count as usize;
        let kv_dim = self.cfg.head_count_kv as usize * self.cfg.head_dim as usize;
        let zeros = vec![0.0f32; max_seq_len * kv_dim];

        let mut new_kv = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if matches!(layer.mixer, Mixer::Attention { .. } | Mixer::GatedAttention { .. }) {
                // KV cache stores the `Act` activation type (f16 by default).
                let hk = wgpu.upload_act(&zeros);
                let hv = wgpu.upload_act(&zeros);
                new_kv.push((i, hk, hv));
            }
        }
        let scores_handle = wgpu.upload_activation(&vec![0.0f32; n_heads * max_seq_len]);
        let weights_handle = wgpu.upload_activation(&vec![0.0f32; n_heads * max_seq_len]);

        for (i, hk, hv) in new_kv {
            self.gpu_kv_cache_k.insert(i, hk);
            self.gpu_kv_cache_v.insert(i, hv);
        }
        self.gpu_attn_scores = Some(scores_handle);
        self.gpu_attn_weights = Some(weights_handle);
        self.gpu_kv_max_seq = max_seq_len;
    }

    /// Non-cfg version for non-wgpu builds.
    #[cfg(not(feature = "wgpu"))]
    pub fn pre_upload_gpu(&mut self) {}

    /// Compile the GPU matmul kernel up front (one dummy launch per dtype
    /// actually present in this model's weights), so the ~7s shader compile
    /// happens with a visible message at load time instead of silently
    /// stalling on the model's first forward pass.
    ///
    /// Must run after `pre_upload_gpu`, which populates `gpu_tensors` with
    /// every GPU-dequantizable weight actually present in this model.
    #[cfg(feature = "wgpu")]
    pub fn warmup_gpu_kernels(&self) {
        let backend = match &self.backend {
            Some(b) => b,
            None => return,
        };

        let mut shapes: Vec<(crate::types::GgmlType, usize)> = Vec::new();
        for handle in self.gpu_tensors.values() {
            let (dtype, in_dim) = handle.shape();
            if !shapes.iter().any(|(dt, _)| *dt == dtype) {
                shapes.push((dtype, in_dim));
            }
        }

        crate::ops::warmup(backend, &shapes);
    }

    /// Non-cfg version for non-wgpu builds.
    #[cfg(not(feature = "wgpu"))]
    pub fn warmup_gpu_kernels(&self) {}

    /// Returns the GPU backend if every tensor this model needs (all layer
    /// weights — attention or short-conv — all norm vectors, lm_head) is
    /// GPU-resident — the preconditions for `run_gpu_resident`, which keeps
    /// the residual stream on GPU for the whole forward pass instead of
    /// crossing the CPU/GPU boundary once per matmul (see [[gpu-sync-bottleneck]]
    /// in project memory).
    #[cfg(feature = "wgpu")]
    pub(crate) fn gpu_resident_ready(&self) -> Option<&crate::ops::wgpu_backend::WgpuBackend> {
        let Some(AnyBackend::Wgpu(b)) = &self.backend else {
            return None;
        };
        if !self.gpu_tensors.contains_key(&self.lm_head) || self.gpu_output_norm.is_none() {
            return None;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if !self.gpu_attn_norms.contains_key(&i) || !self.gpu_ffn_norms.contains_key(&i) {
                return None;
            }
            let mixer_ready = match &layer.mixer {
                Mixer::Attention {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    [wq, wk, wv, wo]
                        .into_iter()
                        .all(|name| self.gpu_tensors.contains_key(name))
                        && q_norm.as_ref().is_none_or(|n| self.gpu_qk_norm.contains_key(n))
                        && k_norm.as_ref().is_none_or(|n| self.gpu_qk_norm.contains_key(n))
                }
                Mixer::ShortConv(sc) => {
                    [&sc.in_proj, &sc.out_proj]
                        .into_iter()
                        .all(|name| self.gpu_tensors.contains_key(name))
                        && self.gpu_conv_weights.contains_key(&i)
                        && self.gpu_conv_history.contains_key(&i)
                }
                Mixer::GatedAttention {
                    wqg,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    [wqg, wk, wv, wo]
                        .into_iter()
                        .all(|name| self.gpu_tensors.contains_key(name))
                        && self.gpu_qk_norm.contains_key(q_norm)
                        && self.gpu_qk_norm.contains_key(k_norm)
                }
                Mixer::GatedDeltaNet(gdn) => {
                    // All quantized projections must be uploaded
                    [gdn.wqkv.as_str(), gdn.wgate.as_str(), gdn.ssm_beta.as_str(), gdn.ssm_alpha.as_str(), gdn.ssm_out.as_str()]
                        .into_iter()
                        .all(|name| self.gpu_tensors.contains_key(name))
                        // Plus all the small f32 state/weights per layer
                        && self.gpu_gdn_conv_weights.contains_key(&i)
                        && self.gpu_gdn_ssm_norm.contains_key(&i)
                        && self.gpu_gdn_ssm_a.contains_key(&i)
                        && self.gpu_gdn_ssm_dt_bias.contains_key(&i)
                        && self.gpu_gdn_conv_history.contains_key(&i)
                        && self.gpu_gdn_recurrent_state.contains_key(&i)
                }
            };
            let ffn_ready = [&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down]
                .into_iter()
                .all(|name| self.gpu_tensors.contains_key(name));
            if !mixer_ready || !ffn_ready {
                return None;
            }
        }
        Some(b)
    }

    /// Fully GPU-resident forward pass: the residual stream (`x`) stays on
    /// GPU as a `Handle` for the entire layer stack, and so does every
    /// mixer's own state now — attention's KV cache, short-conv's
    /// recurrent history, and GatedDeltaNet's conv/history/recurrence state
    /// are all persistent GPU buffers mutated in place. The only CPU round-trip
    /// left in the whole forward pass is the final logits readback.
    #[cfg(feature = "wgpu")]
    pub(crate) fn run_gpu_resident(
        &self,
        b: &crate::ops::wgpu_backend::WgpuBackend,
        token: u32,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let eps = self.cfg.layer_norm_rms_epsilon;
        let d_model = self.cfg.embedding_length as usize;

        let embd = self.weights.dequant_row(&self.token_embd, token as usize)?;
        let mut x_handle = b.upload_act(&embd);
        // Every `residual_add` below is immediately followed by an
        // `rms_norm` (the next mixer's/FFN's prenorm, or the final output
        // norm) — `launch_add_residual_rms_norm` fuses each such pair into
        // one dispatch, so `xn_handle` for a given iteration is always
        // produced by the *previous* residual-add rather than a standalone
        // norm call. The very first prenorm (before any residual exists
        // yet) is the one exception, computed once here.
        let mut xn_handle =
            b.launch_rms_norm(&x_handle, &self.gpu_attn_norms[&0], d_model, eps);

        for (i, layer) in self.layers.iter().enumerate() {
            let mixer_delta_handle = match &layer.mixer {
                Mixer::Attention {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    let hq = &self.gpu_tensors[wq];
                    let hk = &self.gpu_tensors[wk];
                    let hv = &self.gpu_tensors[wv];
                    let ho = &self.gpu_tensors[wo];
                    let n_heads = self.cfg.head_count as usize;
                    let n_kv_heads = self.cfg.head_count_kv as usize;
                    let head_dim = self.cfg.head_dim as usize;
                    let kv_dim = n_kv_heads * head_dim;

                    // QKV → QK-norm → RoPE → KV-cache write → attention →
                    // wo → residual-add, fully GPU-resident — the KV cache
                    // is a persistent GPU buffer mutated in place, so this
                    // layer never crosses back to CPU at all.
                    let (q_h, k_h, v_h) = b.launch_qkv(hq, hk, hv, &xn_handle);

                    let theta = self.cfg.rope_freq_base;
                    if let Some(qn) = q_norm {
                        b.launch_qk_norm_rope(&q_h, &self.gpu_qk_norm[qn], n_heads, head_dim, eps, head_dim, pos, theta);
                    } else {
                        b.launch_rope(&q_h, n_heads, head_dim, head_dim, pos, theta);
                    }
                    if let Some(kn) = k_norm {
                        b.launch_qk_norm_rope(&k_h, &self.gpu_qk_norm[kn], n_kv_heads, head_dim, eps, head_dim, pos, theta);
                    } else {
                        b.launch_rope(&k_h, n_kv_heads, head_dim, head_dim, pos, theta);
                    }

                    let k_cache = &self.gpu_kv_cache_k[&i];
                    let v_cache = &self.gpu_kv_cache_v[&i];
                    b.launch_kv_cache_write(k_cache, v_cache, &k_h, &v_h, pos, kv_dim);

                    let scores = self.gpu_attn_scores.as_ref().expect("ensure_gpu_kv_cache called");
                    let weights = self.gpu_attn_weights.as_ref().expect("ensure_gpu_kv_cache called");
                    let attn_out_handle = b.launch_attention(
                        &q_h,
                        k_cache,
                        v_cache,
                        scores,
                        weights,
                        pos,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                        self.gpu_kv_max_seq,
                    );

                    b.launch_only(ho, &attn_out_handle)
                }
                Mixer::ShortConv(sc) => {
                    let hin = &self.gpu_tensors[&sc.in_proj];
                    let hout = &self.gpu_tensors[&sc.out_proj];
                    let cw = &self.gpu_conv_weights[&i];
                    let hist = &self.gpu_conv_history[&i];

                    // in_proj → gate+conv → out_proj → residual-add, fully
                    // chained on GPU. The gate+conv math is data-parallel
                    // across channels (see `short_conv` kernel), and its
                    // recurrent history lives in a persistent GPU buffer
                    // mutated in place — no CPU round-trip for this mixer.
                    let bcx_handle = b.launch_only(hin, &xn_handle);
                    let conv_out_handle =
                        b.launch_short_conv(&bcx_handle, cw, hist, self.l_cache, d_model);
                    b.launch_only(hout, &conv_out_handle)
                }
                Mixer::GatedAttention {
                    wqg,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    let hqg = &self.gpu_tensors[wqg];
                    let hk = &self.gpu_tensors[wk];
                    let hv = &self.gpu_tensors[wv];
                    let ho = &self.gpu_tensors[wo];
                    let n_heads = self.cfg.head_count as usize;
                    let n_kv_heads = self.cfg.head_count_kv as usize;
                    let head_dim = self.cfg.head_dim as usize;
                    let kv_dim = n_kv_heads * head_dim;

                    // Fused Q+gate, K, V → split Q/gate → QK-norm → partial
                    // RoPE → KV-cache write → attention → sigmoid-gate →
                    // wo → residual-add, fully GPU-resident.
                    let (qg_raw_h, k_h, v_h) = b.launch_qkv(hqg, hk, hv, &xn_handle);
                    let (q_h, gate_h) = b.launch_split_qg(&qg_raw_h, head_dim, n_heads);

                    let theta = self.cfg.rope_freq_base;
                    b.launch_qk_norm_rope(&q_h, &self.gpu_qk_norm[q_norm], n_heads, head_dim, eps, self.n_rot, pos, theta);
                    b.launch_qk_norm_rope(&k_h, &self.gpu_qk_norm[k_norm], n_kv_heads, head_dim, eps, self.n_rot, pos, theta);

                    let k_cache = &self.gpu_kv_cache_k[&i];
                    let v_cache = &self.gpu_kv_cache_v[&i];
                    b.launch_kv_cache_write(k_cache, v_cache, &k_h, &v_h, pos, kv_dim);

                    let scores = self.gpu_attn_scores.as_ref().expect("ensure_gpu_kv_cache called");
                    let weights = self.gpu_attn_weights.as_ref().expect("ensure_gpu_kv_cache called");
                    let attn_out_handle = b.launch_attention(
                        &q_h,
                        k_cache,
                        v_cache,
                        scores,
                        weights,
                        pos,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                        self.gpu_kv_max_seq,
                    );

                    let gated_handle =
                        b.launch_sigmoid_mul(&attn_out_handle, &gate_h, n_heads * head_dim);
                    b.launch_only(ho, &gated_handle)
                }
                Mixer::GatedDeltaNet(gdn) => {
                    let dims = self.gdn_dims.expect("gdn_dims must be set for GatedDeltaNet layers");
                    let n_v_heads = dims.n_v_heads;
                    let n_k_heads = dims.n_k_heads;
                    let head_k_dim = dims.head_k_dim;
                    let head_v_dim = dims.head_v_dim;
                    let key_dim = dims.key_dim;
                    let conv_dim = dims.conv_dim;
                    let d_conv = dims.d_conv;

                    // GPU-resident Gated DeltaNet chain:
                    // qkv = W_qkv * xn  →  l2_norm(Q/K in-place)  →
                    // gate/decay = gdn_gate_decay(in-place)  →
                    // conv_out = causal_conv1d_silu(in-place history)  →
                    // gdn_recurrence(in-place state)  →
                    // gated RMSNorm (ssm_norm * silu(z))  →
                    // ssm_out  →  residual-add

                    // 1. Fused QKV + gate + beta + alpha projections (all read
                    // same xn) — four independent launches, kept as handles,
                    // no CPU round-trip.
                    let hqkv = &self.gpu_tensors[gdn.wqkv.as_str()];
                    let hgate = &self.gpu_tensors[gdn.wgate.as_str()];
                    let hbeta = &self.gpu_tensors[gdn.ssm_beta.as_str()];
                    let halpha = &self.gpu_tensors[gdn.ssm_alpha.as_str()];

                    let qkv_handle = b.launch_only(hqkv, &xn_handle);
                    let gate_out_handle = b.launch_only(hgate, &xn_handle);
                    let beta_handle = b.launch_only(hbeta, &xn_handle);
                    let alpha_handle = b.launch_only(halpha, &xn_handle);

                    // 2. Beta/decay gate computation (mutates in-place on GPU)
                    b.launch_gdn_gate_decay(
                        &beta_handle,
                        &alpha_handle,
                        &self.gpu_gdn_ssm_a[&i],
                        &self.gpu_gdn_ssm_dt_bias[&i],
                        n_v_heads,
                    );

                    // 4. Causal conv1d + SiLU (mutates persistent conv history)
                    let conv_handle = b.launch_causal_conv1d_silu(
                        &qkv_handle,
                        &self.gpu_gdn_conv_weights[&i],
                        &self.gpu_gdn_conv_history[&i],
                        conv_dim,
                        d_conv,
                    );

                    // L2-normalize each Q/K head (dim head_k_dim) on the conv
                    // output, in place — must happen after the conv, not before.
                    b.launch_l2_norm_heads(&conv_handle, 0, key_dim, n_k_heads, head_k_dim, eps, conv_dim);

                    // 5. Delta-rule recurrence (mutates persistent state)
                    let scale = 1.0 / (head_k_dim as f32).sqrt();
                    let gdn_out_handle = b.launch_gdn_recurrence(
                        &self.gpu_gdn_recurrent_state[&i],
                        &conv_handle,
                        &beta_handle,
                        &alpha_handle,
                        n_v_heads,
                        n_k_heads,
                        head_k_dim,
                        head_v_dim,
                        key_dim,
                        conv_dim,
                        scale,
                    );

                    // 6. Gated RMSNorm per head: ssm_norm * norm(out[h]) * silu(z[h]),
                    // in place — matches `cpu_path.rs`'s per-head loop.
                    b.launch_gdn_gated_norm(
                        &gdn_out_handle,
                        &self.gpu_gdn_ssm_norm[&i],
                        &gate_out_handle,
                        n_v_heads,
                        head_v_dim,
                        eps,
                    );

                    // 7. ssm_out projection
                    let hssm_out = &self.gpu_tensors[gdn.ssm_out.as_str()];
                    b.launch_only(hssm_out, &gdn_out_handle)
                }
            };

            // Mixer residual-add fused with the FFN prenorm — one dispatch.
            let ffn_norm_h = &self.gpu_ffn_norms[&i];
            let (new_x, xn2_handle) =
                b.launch_add_residual_rms_norm(&x_handle, &mixer_delta_handle, ffn_norm_h, d_model, eps);
            x_handle = new_x;

            // FFN, fully chained — still no read.
            let hg = &self.gpu_tensors[&layer.ffn_gate];
            let hu = &self.gpu_tensors[&layer.ffn_up];
            let hd = &self.gpu_tensors[&layer.ffn_down];
            let down_handle = b.ffn_chain_from_handle(hg, hu, hd, &xn2_handle);

            // FFN residual-add fused with the next prenorm — either the next
            // layer's attn-norm or, on the last layer, the final output norm
            // (so the separate final `rms_norm` below is never needed).
            let next_norm_h = self
                .gpu_attn_norms
                .get(&(i + 1))
                .unwrap_or_else(|| self.gpu_output_norm.as_ref().unwrap());
            let (new_x, next_xn) =
                b.launch_add_residual_rms_norm(&x_handle, &down_handle, next_norm_h, d_model, eps);
            x_handle = new_x;
            xn_handle = next_xn;
        }

        // lm_head — the one remaining read is the logits. `xn_handle` already
        // holds the output-norm-applied residual stream, fused into the last
        // layer's FFN residual-add above.
        let h_lm = &self.gpu_tensors[&self.lm_head];
        // f32-output matmul: logits stay full-precision for sampling/argmax
        // even though `xn_handle` is f16.
        let logits_handle = b.launch_only_f32out(h_lm, &xn_handle);
        Ok(b.read_handle(logits_handle, h_lm.out_dim()))
    }
}
