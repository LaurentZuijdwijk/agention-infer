# Inference Pipeline

> ⚠️ **Partly out of date.** Some described mechanics (batched flash attention, MoE, TurboQuant KV,
> f16) are not yet implemented — the engine currently does single-token f32 forward with a naive
> attention path and an f32 KV cache. See [`docs/roadmap/`](roadmap/) for what's actually built and
> what's planned. Trust the code over this file where they disagree.

## The Six Core Operations

Every transformer forward pass is composed of exactly these six operations. All other complexity is sequencing and data routing.

### 1. matmul — Matrix-Vector Multiply

The dominant operation. 80%+ of all compute time.

```
W: [out_dim, in_dim]   quantized weight tensor
x: [in_dim]            f32 activation vector
y: [out_dim]           f32 output

y[i] = Σ_j  dequant(W[i,j]) * x[j]
```

Key: dequantize one row at a time directly into the dot product accumulator. Never allocate the full f32 matrix. A Q4_K row of 4096 values is only 2 KB — fits in L1 cache.

With SIMD (CPU):
```rust
fn dot_f32x8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = f32x8::ZERO;
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        acc += f32x8::from(ca) * f32x8::from(cb);
    }
    acc.reduce_add()  // + handle tail
}
```

### 2. rms_norm — Root Mean Square Normalization

Stabilizes activations between operations. Applied before attention and FFN in each layer.

```
rms  = sqrt( (1/d) × Σ x[i]² + eps )
y[i] = (x[i] / rms) × weight[i]
```

```rust
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let rms = (x.iter().map(|v| v*v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(weight).map(|(&xi, &wi)| wi * xi / rms).collect()
}
```

`eps` is typically `1e-5` or `1e-6`. Read from `{arch}.attention.layer_norm_rms_epsilon`.

### 3. rope — Rotary Position Embedding

Encodes token position by rotating Q and K vectors. Preserves relative position information in dot products.

For each pair of dimensions `(d, d + half_dim)` at position `pos`:
```
θ_d = pos / theta^(2d / head_dim)
q'[d]          = q[d] * cos(θ_d) - q[d + half_dim] * sin(θ_d)
q'[d + half_dim] = q[d] * sin(θ_d) + q[d + half_dim] * cos(θ_d)
```

Applied identically to K. Applied per attention head.

```rust
fn rope_head(h: &mut [f32], pos: usize, theta: f32) {
    let half = h.len() / 2;
    for d in 0..half {
        let freq = pos as f32 / theta.powf(2.0 * d as f32 / h.len() as f32);
        let (sin, cos) = freq.sin_cos();
        let x0 = h[d];
        let x1 = h[d + half];
        h[d]        = x0 * cos - x1 * sin;
        h[d + half] = x1 * cos + x0 * sin;
    }
}
```

**RoPE scaling for long context**: Models with context > their base training length use scaling to slow down rotations. Read `{arch}.rope.scaling.type` and `.factor`. YaRN scaling (used by Llama 3) applies different scaling to low vs high frequency dimensions.

### 4. softmax — Probability Normalization

Converts raw scores to probabilities summing to 1.0. Applied in two places: attention scores and final logits.

```
// Numerically stable: subtract max before exp
max  = max(x)
x[i] = exp(x[i] - max)
sum  = Σ x[i]
x[i] = x[i] / sum
```

```rust
fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    x.iter_mut().for_each(|v| *v = (*v - max).exp());
    let sum: f32 = x.iter().sum();
    x.iter_mut().for_each(|v| *v /= sum);
}
```

### 5. silu_mul — SwiGLU Gate

The FFN activation function in Llama-family models. Applied elementwise between gate and up projections.

```
silu(x) = x / (1 + e^-x)
output[i] = gate[i] * silu(up[i])
```

```rust
fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| g * (u / (1.0 + (-u).exp()))).collect()
}
```

### 6. add — Residual Connection

Elementwise addition. Applied after attention output and after FFN output. Each layer adds a correction to the residual stream rather than replacing it.

```rust
fn add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(x, y)| *x += *y);
}
```

---

## Full Forward Pass — Dense Model

```rust
fn forward(model: &Model, token: u32, pos: usize,
           kv: &mut KvCache) -> Vec<f32> {
    // 1. Embedding lookup
    let mut x = model.token_embd.row(token as usize);   // [d_model]

    // 2. Transformer layers
    for (i, layer) in model.layers.iter().enumerate() {

        // --- Attention ---
        let xn = rms_norm(&x, &layer.attn_norm, model.cfg.rms_eps);

        let mut q = matmul(&xn, &layer.wq);    // [n_heads * head_dim]
        let mut k = matmul(&xn, &layer.wk);    // [n_kv_heads * head_dim]
        let v     = matmul(&xn, &layer.wv);    // [n_kv_heads * head_dim]

        rope(&mut q, &mut k, pos, &model.cfg.rope);

        kv.write(i, pos, &k, &v);              // TurboQuant compression
        let (k_all, v_all) = kv.read_up_to(i, pos);

        let attn = attention(&q, &k_all, &v_all, pos, &model.cfg);
        let attn_out = matmul(&attn, &layer.wo);
        add(&mut x, &attn_out);                // residual

        // --- FFN (SwiGLU) ---
        let xn = rms_norm(&x, &layer.ffn_norm, model.cfg.rms_eps);

        let gate = matmul(&xn, &layer.ffn_gate);
        let up   = matmul(&xn, &layer.ffn_up);
        let h    = silu_mul(&gate, &up);
        let ffn_out = matmul(&h, &layer.ffn_down);
        add(&mut x, &ffn_out);                 // residual
    }

    // 3. Final norm + project to vocab
    let xn = rms_norm(&x, &model.output_norm, model.cfg.rms_eps);
    matmul(&xn, &model.lm_head)               // [vocab_size] logits
}
```

---

## Attention Kernel

### Standard (short context)

```
scores[j] = dot(q, k_all[j]) / sqrt(head_dim)   for j in 0..=pos
scores    = softmax(scores)
output    = Σ scores[j] * v_all[j]
```

Applied per head. With GQA, Q heads are grouped to share KV heads:
```
head 0..group_size-1     → use kv_head 0
head group_size..2×gs-1  → use kv_head 1
...
```

### Flash Attention (long context)

Tiles the computation to avoid materializing the full `[seq_len, seq_len]` score matrix. Required for context > ~8K tokens.

Outer loop over KV tiles, inner loop over Q tiles. Maintains running softmax statistics `(m, l)` that are updated incrementally without seeing all scores simultaneously.

```rust
fn flash_attention(q: &[f32], kv_cache: &KvCache, pos: usize,
                   head_dim: usize, n_heads: usize) -> Vec<f32> {
    const TILE: usize = 64;
    let mut out = vec![0f32; n_heads * head_dim];

    for head in 0..n_heads {
        let kv_head = head / (n_heads / n_kv_heads);
        let q_head = &q[head * head_dim..(head+1) * head_dim];

        let mut m = f32::NEG_INFINITY;    // running max
        let mut l = 0f32;                 // running sum of exp
        let mut acc = vec![0f32; head_dim];

        for tile_start in (0..=pos).step_by(TILE) {
            let tile_end = (tile_start + TILE).min(pos + 1);

            // compute scores for this tile
            let scores: Vec<f32> = (tile_start..tile_end).map(|j| {
                let k_j = kv_cache.read_k(kv_head, j); // dequantizes here
                dot(q_head, &k_j) / (head_dim as f32).sqrt()
            }).collect();

            // update running softmax
            let tile_max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let m_new = m.max(tile_max);

            let scale_old = (m - m_new).exp();
            let tile_exp: Vec<f32> = scores.iter().map(|&s| (s - m_new).exp()).collect();
            let tile_sum: f32 = tile_exp.iter().sum();

            l = scale_old * l + tile_sum;

            // accumulate into output
            acc.iter_mut().enumerate().for_each(|(d, a)| {
                let v_contrib: f32 = tile_exp.iter().enumerate().map(|(j, &w)| {
                    let v_j = kv_cache.read_v(kv_head, tile_start + j);
                    w * v_j[d]
                }).sum();
                *a = scale_old * *a + v_contrib;
            });

            m = m_new;
        }

        // normalize
        let out_head = &mut out[head * head_dim..(head+1) * head_dim];
        for (o, a) in out_head.iter_mut().zip(acc.iter()) {
            *o = a / l;
        }
    }
    out
}
```

### Local/Global Attention (Gemma 3 style)

Most layers use local (sliding window) attention, a few use global (full context):

```rust
fn attention_for_layer(layer_idx: usize, q: &[f32], kv_cache: &KvCache,
                        pos: usize, cfg: &ModelConfig) -> Vec<f32> {
    let is_global = cfg.global_attention_layers.contains(&layer_idx);
    let window = if is_global { pos + 1 } else { cfg.local_window_size.min(pos + 1) };
    let kv_start = if is_global { 0 } else { pos + 1 - window };
    flash_attention_range(q, kv_cache, kv_start, pos, cfg)
}
```

---

## MoE Forward Pass

Replaces the dense FFN block when `model.is_moe()`:

```rust
fn moe_ffn(x: &[f32], layer: &MoeLayer,
           trace: Option<&mut LayerTrace>) -> Vec<f32> {
    // 1. Router
    let scores  = matmul(x, &layer.router_weight);   // [n_experts]
    let probs   = softmax_copy(&scores);
    let top_k   = top_k_indices(&probs, layer.n_active);  // e.g. 8 of 256

    // 2. Optional trace
    if let Some(t) = trace {
        t.router_scores    = scores.clone();
        t.selected_experts = top_k.clone();
        t.selected_weights = top_k.iter().map(|&i| probs[i]).collect();
    }

    // 3. Expert dispatch — schedule by memory offset for locality
    let mut sorted_experts = top_k.clone();
    sorted_experts.sort_by_key(|&i| layer.expert_offset(i));

    let mut out = vec![0f32; x.len()];
    for &expert_idx in &sorted_experts {
        let w = probs[expert_idx];
        let expert_out = dense_ffn(x, &layer.experts[expert_idx]);
        for (o, e) in out.iter_mut().zip(expert_out.iter()) {
            *o += w * e;
        }
    }

    // 4. Shared experts (DeepSeek style) — always run
    if let Some(shared) = &layer.shared_expert {
        let shared_out = dense_ffn(x, shared);
        for (o, s) in out.iter_mut().zip(shared_out.iter()) {
            *o += s;
        }
    }

    out
}
```

Expert locality scheduling: sort selected experts by their byte offset in the weight file before dispatching. On a 100 GB model file, sequential reads are dramatically faster than random access, even on NVMe and even with mmap page caching.

---

## Decode Loop

```rust
pub fn generate(model: &Model, tokens: &[u32], config: &GenerateConfig,
                kv: &mut KvCache) -> impl Iterator<Item = TokenEvent> {
    // Prefill: process prompt tokens
    let mut pos = 0;
    for &token in &tokens[..tokens.len() - 1] {
        forward(model, token, pos, kv);   // compute KV, discard logits
        pos += 1;
    }

    // Decode: generate new tokens
    let mut token = *tokens.last().unwrap();
    std::iter::from_fn(move || {
        let logits = forward(model, token, pos, kv);
        pos += 1;

        let next = sampler.sample(&logits, &config.sampler);

        // Check stop conditions
        if next == model.eos_token { return Some(TokenEvent::End(StopReason::Eos)); }
        if config.stop_sequences.iter().any(|s| matches_suffix(&history, s)) {
            return Some(TokenEvent::End(StopReason::StopSequence));
        }
        if pos >= config.max_tokens { return Some(TokenEvent::End(StopReason::MaxTokens)); }

        token = next;
        Some(TokenEvent::Token(next))
    })
}
```

`TokenEvent` distinguishes thinking tokens (inside `<think>...</think>`) from output tokens, which is required for models like MiniMax M2.7 that use interleaved thinking.

---

## Speculative Decoding Loop

```rust
pub fn generate_speculative(target: &Model, draft: &dyn DraftSource,
                             tokens: &[u32], config: &GenerateConfig,
                             kv: &mut KvCache) -> impl Iterator<Item = TokenEvent> {
    // prefill same as above
    let mut pos = prefill(target, tokens, kv);
    let mut token = *tokens.last().unwrap();

    std::iter::from_fn(move || {
        // 1. Draft n tokens cheaply
        let drafts = draft.propose(token, pos, config.spec_n);

        // 2. Verify in one parallel pass
        let verified = target.verify_parallel(token, &drafts, pos, kv);

        // 3. Accept longest valid prefix
        let accept_len = drafts.iter().zip(verified.iter())
            .take_while(|(d, v)| d == v)
            .count()
            .min(drafts.len());

        // 4. Emit accepted tokens
        pos += accept_len + 1;
        token = if accept_len < drafts.len() {
            verified[accept_len]   // first rejected position gets target's token
        } else {
            verified[accept_len]   // one more token from target after full acceptance
        };

        Some(TokenEvent::Tokens(verified[..=accept_len].to_vec()))
    })
}
```

`DraftSource` implementations:
- `SmallModelDraft` — separate loaded model
- `LayerSkipDraft` — same model, skip layers 20..60
- `EagleDraft` — trained EAGLE-3 head
- `MtpDraft` — use MTP heads if model has them

---

## Sampler

Applied to logits after the final forward pass:

```rust
pub fn sample(logits: &mut Vec<f32>, cfg: &SamplerConfig,
              recent_tokens: &[u32]) -> u32 {
    // 1. Repetition penalty
    for &tok in recent_tokens.iter().rev().take(cfg.rep_penalty_window) {
        logits[tok as usize] /= cfg.rep_penalty;
    }

    // 2. Temperature
    if cfg.temperature > 0.0 {
        logits.iter_mut().for_each(|l| *l /= cfg.temperature);
    } else {
        // Greedy: return argmax directly
        return argmax(logits) as u32;
    }

    // 3. Softmax
    softmax(logits);

    // 4. Min-P filtering (preferred over top-p for modern models)
    if cfg.min_p > 0.0 {
        let max_p = logits.iter().cloned().fold(0f32, f32::max);
        let threshold = cfg.min_p * max_p;
        logits.iter_mut().for_each(|p| if *p < threshold { *p = 0.0 });
        renormalize(logits);
    }

    // 5. Top-K filtering
    if cfg.top_k > 0 {
        keep_top_k(logits, cfg.top_k);
        renormalize(logits);
    }

    // 6. Sample from distribution
    multinomial_sample(logits)
}
```

Default sampler config for quality output:
```
temperature: 0.7
min_p:       0.05
top_k:       0       (disabled, use min_p instead)
top_p:       1.0     (disabled)
rep_penalty: 1.1
rep_window:  64
```
