//! GPU-resident launch helpers: each takes/returns GPU handles with no CPU
//! round-trip, so a whole layer (or the whole forward pass) can chain
//! through them without ever reading an intermediate result back. Used by
//! `LlamaModel::run_gpu_resident`.

use super::{GpuWeightHandle, WgpuBackend};
use cubecl::wgpu::WgpuRuntime;

impl WgpuBackend {
    /// Launch a dequant matmul without reading the result back — caller
    /// batches multiple `launch_only` calls, then reads them all with one
    /// `client.read`, instead of paying a blocking round-trip per matmul.
    pub(crate) fn launch_only(
        &self,
        h: &GpuWeightHandle,
        x_handle: &cubecl::server::Handle,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        let out_handle = self.client.empty(h.out_dim * core::mem::size_of::<f32>());
        let grid_x = (h.out_dim as u32).min(65535);
        let grid_y = ((h.out_dim as u32) + grid_x - 1) / grid_x;
        unsafe {
            crate::ops::kernels::wgpu::matmul_dequant_wgpu::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(grid_x, grid_y, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(h.handle.clone(), h.out_dim * h.row_u32s),
                ArrayArg::from_raw_parts(x_handle.clone(), h.in_dim),
                ArrayArg::from_raw_parts(out_handle.clone(), h.out_dim),
                h.dtype as u32,
                h.in_dim,
                h.row_u32s,
                grid_x,
            );
        }
        out_handle
    }

    /// Launch the SwiGLU combine kernel reading two GPU-resident buffers
    /// directly (no CPU round-trip for `gate`/`up`).
    pub(crate) fn launch_silu_mul(
        &self,
        gate_handle: &cubecl::server::Handle,
        up_handle: &cubecl::server::Handle,
        len: usize,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        let out_handle = self.client.empty(len * core::mem::size_of::<f32>());
        unsafe {
            let threads = 64u32;
            let workgroups = (len as u32 + threads - 1) / threads;
            crate::ops::kernels::wgpu::silu_mul::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(gate_handle.clone(), len),
                ArrayArg::from_raw_parts(up_handle.clone(), len),
                ArrayArg::from_raw_parts(out_handle.clone(), len),
            );
        }
        out_handle
    }

    /// Launch RMSNorm: `out = (x / rms(x)) * weight`, reading `x` and
    /// `weight` directly from GPU handles — no CPU round-trip.
    pub(crate) fn launch_rms_norm(
        &self,
        x_handle: &cubecl::server::Handle,
        weight_handle: &cubecl::server::Handle,
        len: usize,
        eps: f32,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        let out_handle = self.client.empty(len * core::mem::size_of::<f32>());
        unsafe {
            crate::ops::kernels::wgpu::rms_norm::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(x_handle.clone(), len),
                ArrayArg::from_raw_parts(weight_handle.clone(), len),
                ArrayArg::from_raw_parts(out_handle.clone(), len),
                eps,
            );
        }
        out_handle
    }

    /// Fused `new_x = x + delta` then `normed = norm(new_x)` — one dispatch
    /// instead of a separate residual-add + `launch_rms_norm` pair. Returns
    /// `(new_x_handle, normed_handle)`.
    pub(crate) fn launch_add_residual_rms_norm(
        &self,
        x_handle: &cubecl::server::Handle,
        delta_handle: &cubecl::server::Handle,
        weight_handle: &cubecl::server::Handle,
        len: usize,
        eps: f32,
    ) -> (cubecl::server::Handle, cubecl::server::Handle) {
        use cubecl::prelude::*;
        let new_x_handle = self.client.empty(len * core::mem::size_of::<f32>());
        let normed_handle = self.client.empty(len * core::mem::size_of::<f32>());
        unsafe {
            crate::ops::kernels::wgpu::add_residual_rms_norm::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(x_handle.clone(), len),
                ArrayArg::from_raw_parts(delta_handle.clone(), len),
                ArrayArg::from_raw_parts(weight_handle.clone(), len),
                ArrayArg::from_raw_parts(new_x_handle.clone(), len),
                ArrayArg::from_raw_parts(normed_handle.clone(), len),
                eps,
            );
        }
        (new_x_handle, normed_handle)
    }

    /// Launch the short-conv gate+depthwise-conv kernel (LFM2 `ShortConv`
    /// mixer): reads `bcx` (already GPU-resident, the output of an
    /// `in_proj` launch) and the layer's static conv weights, mutates
    /// the layer's persistent `history` handle in place, and returns
    /// the `conv_out` handle — no CPU round-trip.
    pub(crate) fn launch_short_conv(
        &self,
        bcx_handle: &cubecl::server::Handle,
        weight_handle: &cubecl::server::Handle,
        history_handle: &cubecl::server::Handle,
        l: usize,
        d: usize,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        let out_handle = self.client.empty(d * core::mem::size_of::<f32>());
        unsafe {
            let threads = 64u32;
            let workgroups = (d as u32 + threads - 1) / threads;
            crate::ops::kernels::wgpu::short_conv::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(bcx_handle.clone(), 3 * d),
                ArrayArg::from_raw_parts(weight_handle.clone(), d * l),
                ArrayArg::from_raw_parts(history_handle.clone(), d * l.saturating_sub(1)),
                ArrayArg::from_raw_parts(out_handle.clone(), d),
                l,
            );
        }
        out_handle
    }

    /// Launch Q/K/V projections without reading the results back — the
    /// fully GPU-resident attention path keeps q/k/v as handles through
    /// RoPE, QK-norm, KV-cache write, and the attention kernel itself.
    pub(crate) fn launch_qkv(
        &self,
        hq: &GpuWeightHandle,
        hk: &GpuWeightHandle,
        hv: &GpuWeightHandle,
        x_handle: &cubecl::server::Handle,
    ) -> (
        cubecl::server::Handle,
        cubecl::server::Handle,
        cubecl::server::Handle,
    ) {
        let q = self.launch_only(hq, x_handle);
        let k = self.launch_only(hk, x_handle);
        let v = self.launch_only(hv, x_handle);
        (q, k, v)
    }

    /// Apply rotary position embedding in place to a GPU-resident Q or K
    /// buffer (`n_heads` heads of `head_dim` each). `n_rot` is the number of
    /// dims rotated per head (pass `head_dim` for full rotation, or fewer
    /// for Qwen3.5's `GatedAttention` partial RoPE).
    pub(crate) fn launch_rope(
        &self,
        handle: &cubecl::server::Handle,
        n_heads: usize,
        head_dim: usize,
        n_rot: usize,
        pos: usize,
        theta: f32,
    ) {
        use cubecl::prelude::*;
        let total = n_heads * (n_rot / 2);
        let threads = 64u32;
        let workgroups = (total as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::rope::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(handle.clone(), n_heads * head_dim),
                n_heads,
                head_dim,
                n_rot,
                pos,
                theta,
            );
        }
    }

    /// Deinterleave a fused `[Q(head_dim) | gate(head_dim)]`-per-head buffer
    /// (the output of Qwen3.5's `wqg` projection) into contiguous `q` and
    /// `gate` handles.
    pub(crate) fn launch_split_qg(
        &self,
        qg_raw_handle: &cubecl::server::Handle,
        head_dim: usize,
        n_heads: usize,
    ) -> (cubecl::server::Handle, cubecl::server::Handle) {
        use cubecl::prelude::*;
        let q_handle = self.client.empty(n_heads * head_dim * core::mem::size_of::<f32>());
        let gate_handle = self.client.empty(n_heads * head_dim * core::mem::size_of::<f32>());
        let threads = 64u32;
        let workgroups = ((n_heads * head_dim) as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::split_qg::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(qg_raw_handle.clone(), 2 * n_heads * head_dim),
                ArrayArg::from_raw_parts(q_handle.clone(), n_heads * head_dim),
                ArrayArg::from_raw_parts(gate_handle.clone(), n_heads * head_dim),
                head_dim,
            );
        }
        (q_handle, gate_handle)
    }

    /// `out[i] = a[i] * sigmoid(b[i])`, both already GPU-resident — used for
    /// Qwen3.5's `GatedAttention` output gate (`attn_out *= sigmoid(gate)`).
    pub(crate) fn launch_sigmoid_mul(
        &self,
        a_handle: &cubecl::server::Handle,
        b_handle: &cubecl::server::Handle,
        len: usize,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        let out_handle = self.client.empty(len * core::mem::size_of::<f32>());
        let threads = 64u32;
        let workgroups = (len as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::sigmoid_mul::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(a_handle.clone(), len),
                ArrayArg::from_raw_parts(b_handle.clone(), len),
                ArrayArg::from_raw_parts(out_handle.clone(), len),
            );
        }
        out_handle
    }

    /// Fused QK-norm + RoPE: one launch instead of a separate QK-norm followed
    /// by `launch_rope` — same math, one thread per head does the RMSNorm
    /// reduction then the rotation back to back.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_qk_norm_rope(
        &self,
        handle: &cubecl::server::Handle,
        weight_handle: &cubecl::server::Handle,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        n_rot: usize,
        pos: usize,
        theta: f32,
    ) {
        use cubecl::prelude::*;
        let threads = 64u32;
        let workgroups = (n_heads as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::qk_norm_rope::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(handle.clone(), n_heads * head_dim),
                ArrayArg::from_raw_parts(weight_handle.clone(), head_dim),
                n_heads,
                head_dim,
                eps,
                n_rot,
                pos,
                theta,
            );
        }
    }

    /// Append the current token's (already RoPE'd) K/V into the layer's
    /// persistent GPU-resident cache at slot `pos`. Mutates `k_cache`/
    /// `v_cache` in place.
    pub(crate) fn launch_kv_cache_write(
        &self,
        k_cache: &cubecl::server::Handle,
        v_cache: &cubecl::server::Handle,
        new_k: &cubecl::server::Handle,
        new_v: &cubecl::server::Handle,
        pos: usize,
        kv_dim: usize,
    ) {
        use cubecl::prelude::*;
        let threads = 64u32;
        let workgroups = (kv_dim as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::kv_cache_write::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(k_cache.clone(), pos * kv_dim + kv_dim),
                ArrayArg::from_raw_parts(v_cache.clone(), pos * kv_dim + kv_dim),
                ArrayArg::from_raw_parts(new_k.clone(), kv_dim),
                ArrayArg::from_raw_parts(new_v.clone(), kv_dim),
                pos,
            );
        }
    }

    /// Causal GQA attention against the layer's GPU-resident KV cache.
    /// Returns a fresh `out` handle (`n_heads * head_dim`); `scores` and
    /// `weights` are reused scratch buffers (`n_heads * max_seq` each).
    /// Three kernel launches in sequence: scores → softmax → output.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_attention(
        &self,
        q: &cubecl::server::Handle,
        k_cache: &cubecl::server::Handle,
        v_cache: &cubecl::server::Handle,
        scores: &cubecl::server::Handle,
        weights: &cubecl::server::Handle,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        max_seq: usize,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        let out_handle = self.client.empty(n_heads * head_dim * core::mem::size_of::<f32>());
        let sums_handle = self.client.empty(n_heads * core::mem::size_of::<f32>());
        let threads = 64u32;

        // 1. Q·K scaled dot-product: one thread per (head, kv-position).
        let seq_len = pos + 1;
        let wg_scores = (n_heads as u32 * seq_len as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::attention_scores::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(wg_scores, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(q.clone(), n_heads * head_dim),
                ArrayArg::from_raw_parts(k_cache.clone(), max_seq * n_kv_heads * head_dim),
                ArrayArg::from_raw_parts(scores.clone(), n_heads * max_seq),
                pos,
                head_dim,
                n_heads,
                n_kv_heads,
                max_seq,
            );
        }

        // 2. Numerically-stable softmax: one thread per head.
        let wg_softmax = (n_heads as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::attention_softmax::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(wg_softmax, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(scores.clone(), n_heads * max_seq),
                ArrayArg::from_raw_parts(weights.clone(), n_heads * max_seq),
                ArrayArg::from_raw_parts(sums_handle.clone(), n_heads),
                pos,
                n_heads,
                max_seq,
            );
        }

        // 3. Weighted sum of V: one thread per (head, output-dim).
        let wg_out = (n_heads as u32 * head_dim as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::attention_output::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(wg_out, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(v_cache.clone(), max_seq * n_kv_heads * head_dim),
                ArrayArg::from_raw_parts(weights.clone(), n_heads * max_seq),
                ArrayArg::from_raw_parts(sums_handle.clone(), n_heads),
                ArrayArg::from_raw_parts(out_handle.clone(), n_heads * head_dim),
                pos,
                head_dim,
                n_heads,
                n_kv_heads,
                max_seq,
            );
        }

        out_handle
    }

    /// Upload a plain f32 activation/weight vector to GPU (no packing —
    /// used for embeddings and norm vectors, which are always f32).
    pub(crate) fn upload_activation(&self, x: &[f32]) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        self.client.create_from_slice(f32::as_bytes(x))
    }

    /// Blocking read of a GPU handle back to a `Vec<f32>` of length `len`.
    pub(crate) fn read_handle(&self, h: cubecl::server::Handle, len: usize) -> Vec<f32> {
        use cubecl::prelude::*;
        let bytes = self.client.read_one_unchecked(h);
        f32::from_bytes(&bytes)[..len].to_vec()
    }

    /// Same chain as `matmul_dequant_ffn` (`gate`→`up`→SiLU-combine→`down`)
    /// but takes and returns GPU handles with no readback at all — used by
    /// the fully GPU-resident forward pass to keep `down`'s output on GPU
    /// for a subsequent residual-add.
    pub(crate) fn ffn_chain_from_handle(
        &self,
        h_gate: &GpuWeightHandle,
        h_up: &GpuWeightHandle,
        h_down: &GpuWeightHandle,
        x_handle: &cubecl::server::Handle,
    ) -> cubecl::server::Handle {
        let gate_handle = self.launch_only(h_gate, x_handle);
        let up_handle = self.launch_only(h_up, x_handle);
        let act_handle = self.launch_silu_mul(&gate_handle, &up_handle, h_gate.out_dim);
        self.launch_only(h_down, &act_handle)
    }

    /// L2-normalize `n_heads` segments of `head_dim` at each of two base
    /// offsets (Q range, K range), in place, in a single dispatch — Qwen3.5
    /// Gated DeltaNet's Q/K normalization (no learned weight, unlike
    /// `launch_qk_norm_rope`).
    pub(crate) fn launch_l2_norm_heads(
        &self,
        handle: &cubecl::server::Handle,
        base_offset: usize,
        base_offset2: usize,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        total_len: usize,
    ) {
        use cubecl::prelude::*;
        let threads = 64u32;
        let workgroups = ((2 * n_heads) as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::l2_norm_heads::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(handle.clone(), total_len),
                base_offset,
                base_offset2,
                n_heads,
                head_dim,
                eps,
            );
        }
    }

    /// Gated DeltaNet's output gated-RMSNorm: `weight * norm(x[h]) *
    /// silu(gate[h])` per head, in place on `handle`. `weight` is
    /// `[head_dim]`, reused across all `n_heads` segments of `handle`/`gate`.
    pub(crate) fn launch_gdn_gated_norm(
        &self,
        handle: &cubecl::server::Handle,
        weight_handle: &cubecl::server::Handle,
        gate_handle: &cubecl::server::Handle,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        use cubecl::prelude::*;
        let total_len = n_heads * head_dim;
        let threads = 64u32;
        let workgroups = (n_heads as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::gdn_gated_rms_norm::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(handle.clone(), total_len),
                ArrayArg::from_raw_parts(weight_handle.clone(), head_dim),
                ArrayArg::from_raw_parts(gate_handle.clone(), total_len),
                n_heads,
                head_dim,
                eps,
            );
        }
    }

    /// Gated DeltaNet's `beta`/`decay` gate computation, in place on the
    /// layer's raw projection outputs. `ssm_a`/`dt_bias` are small static
    /// per-layer vectors uploaded once.
    pub(crate) fn launch_gdn_gate_decay(
        &self,
        beta_raw_handle: &cubecl::server::Handle,
        alpha_raw_handle: &cubecl::server::Handle,
        ssm_a_handle: &cubecl::server::Handle,
        dt_bias_handle: &cubecl::server::Handle,
        n_v_heads: usize,
    ) {
        use cubecl::prelude::*;
        let threads = 64u32;
        let workgroups = (n_v_heads as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::gdn_gate_decay::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(beta_raw_handle.clone(), n_v_heads),
                ArrayArg::from_raw_parts(alpha_raw_handle.clone(), n_v_heads),
                ArrayArg::from_raw_parts(ssm_a_handle.clone(), n_v_heads),
                ArrayArg::from_raw_parts(dt_bias_handle.clone(), n_v_heads),
            );
        }
    }

    /// Gated DeltaNet's causal depthwise conv1d + SiLU, reading the raw
    /// `wqkv` projection output and mutating the layer's persistent
    /// `history` buffer in place. Returns the `conv_out` handle.
    pub(crate) fn launch_causal_conv1d_silu(
        &self,
        input_handle: &cubecl::server::Handle,
        weight_handle: &cubecl::server::Handle,
        history_handle: &cubecl::server::Handle,
        conv_dim: usize,
        d_conv: usize,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        let out_handle = self.client.empty(conv_dim * core::mem::size_of::<f32>());
        let threads = 64u32;
        let workgroups = (conv_dim as u32 + threads - 1) / threads;
        unsafe {
            crate::ops::kernels::wgpu::causal_conv1d_silu::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(workgroups, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(input_handle.clone(), conv_dim),
                ArrayArg::from_raw_parts(weight_handle.clone(), conv_dim * d_conv),
                ArrayArg::from_raw_parts(history_handle.clone(), conv_dim * d_conv.saturating_sub(1)),
                ArrayArg::from_raw_parts(out_handle.clone(), conv_dim),
                d_conv,
            );
        }
        out_handle
    }

    /// Gated DeltaNet's per-head delta-rule recurrence: one workgroup per
    /// v-head, one thread per output column (no cross-thread sync needed —
    /// see the kernel doc comment). Mutates the layer's persistent
    /// `state` buffer in place and returns the `out` handle (`value_dim`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_gdn_recurrence(
        &self,
        state_handle: &cubecl::server::Handle,
        conv_out_handle: &cubecl::server::Handle,
        beta_handle: &cubecl::server::Handle,
        decay_handle: &cubecl::server::Handle,
        n_v_heads: usize,
        n_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        key_dim: usize,
        conv_dim: usize,
        scale: f32,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;
        let value_dim = n_v_heads * head_v_dim;
        let out_handle = self.client.empty(value_dim * core::mem::size_of::<f32>());
        unsafe {
            crate::ops::kernels::wgpu::gdn_recurrence::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(n_v_heads as u32, 1, 1),
                CubeDim::new_1d(head_v_dim as u32),
                ArrayArg::from_raw_parts(state_handle.clone(), n_v_heads * head_k_dim * head_v_dim),
                ArrayArg::from_raw_parts(conv_out_handle.clone(), conv_dim),
                ArrayArg::from_raw_parts(beta_handle.clone(), n_v_heads),
                ArrayArg::from_raw_parts(decay_handle.clone(), n_v_heads),
                ArrayArg::from_raw_parts(out_handle.clone(), value_dim),
                n_k_heads,
                head_k_dim,
                head_v_dim,
                key_dim,
                scale,
            );
        }
        out_handle
    }
}
