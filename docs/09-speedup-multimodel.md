# Speedup Techniques & Multi-Model Serving

## Speculative Decoding

### Concept

Normal decode: one forward pass → one token. Speculative decode: draft N tokens cheaply, verify all N in one parallel pass. Output is mathematically identical to non-speculative decoding.

```
Draft N=5:    [t1, t2, t3, t4, t5]   (cheap, fast, sometimes wrong)
Verify:       target model processes [input, t1, t2, t3, t4, t5] in one pass
Accept prefix: if t1 correct, t2 correct, t3 wrong → emit [t1, t2], redo from t3
```

Speedup when draft acceptance rate is high (easy tokens, familiar patterns). No speedup on hard/surprising tokens. Never worse than non-speculative in terms of output quality — rejection just falls back to the target's token.

### Draft Sources

All implement `DraftSource` trait:

```rust
pub trait DraftSource {
    fn propose(&mut self, token: u32, pos: usize, n: usize,
               kv: &KvCache) -> Vec<u32>;
}
```

**1. Small model draft** (separate loaded model)
```
Llama 3.2 1B proposes → Llama 3.1 8B verifies
Qwen2.5 1.5B proposes → Qwen2.5 7B verifies
Typical acceptance: 70–85% for related model families
Speedup: 2–3×
Memory: +1–1.5 GB for draft model
```

**2. Self-speculative** (same model, skip layers)
```
Skip layers 20–60 for the draft pass
Verification uses all layers
No extra memory
Acceptance: ~65–75%
Speedup: 1.5–2×
```

**3. EAGLE-3** (trained draft head)
```
Tiny head (~277 MB) trained on target model's layer representations
Conditions on early + middle + late layer hidden states
Acceptance: 80–90% (best among methods)
Speedup: 2–4×
Requires training: ~hours on RTX 4090 for a new model
```

**4. MTP heads** (model was trained with multi-token prediction)
```
Free — heads are already part of the model
DeepSeek V3, some Qwen2.5 variants
No extra memory, no extra training
Acceptance: depends on model
Speedup: 1.5–3×
```

### Verification Step

Target model processes draft tokens in parallel via causal masking:

```rust
fn verify_parallel(model: &Model, context: &[u32], drafts: &[u32],
                   pos: usize, kv: &mut KvCache) -> Vec<u32> {
    // Build input: context + drafts
    // Run single forward pass with causal attention
    // Extract token predictions at each draft position
    // Compare to draft tokens
    let logits_sequence = model.forward_sequence(context, drafts, pos, kv);
    logits_sequence.iter()
        .map(|logits| sampler.greedy(logits))   // greedy for verification
        .collect()
}
```

---

## Layer Skipping (Self-Speculative)

Some tokens don't need all 80 layers. Skip middle layers for easy tokens.

```rust
pub struct LayerSkipDraft {
    skip_range: Range<usize>,   // e.g. 20..60
}

impl DraftSource for LayerSkipDraft {
    fn propose(&mut self, token: u32, pos: usize, n: usize,
               kv: &KvCache) -> Vec<u32> {
        let mut draft_tokens = Vec::with_capacity(n);
        let mut t = token;
        for _ in 0..n {
            let logits = model.forward_skipping(t, pos, &self.skip_range, kv);
            t = sampler.sample(&logits);
            draft_tokens.push(t);
        }
        draft_tokens
    }
}
```

Layer skipping can also be used independently of speculation as an "easy token fast path":
- Run a confidence classifier after layer 16
- If confidence is high → exit early, skip remaining layers
- If low → run all layers

---

## TurboQuant KV Cache

See `07-turbo-quant-kv-cache.md` for full details.

Summary: 3-bit KV cache with Walsh-Hadamard rotation + Lloyd-Max quantization. Near-lossless at 5× compression. Default format — not optional.

---

## Multi-Model Serving

### Unified Memory Pool Advantage

On discrete GPU: loading multiple models means swapping VRAM. On Strix Halo's 96 GB unified pool, multiple small models fit simultaneously with zero swap cost:

```
Resident simultaneously (~10 GB total):
  Phi-3.5 mini 3B     2 GB    fast/simple queries
  Qwen2.5 7B          4 GB    general queries
  Qwen2.5-Coder 7B    4 GB    code queries
  Router model        0.1 GB  query classifier

On-demand (swap the above when needed):
  MiniMax M2.7        93 GB   complex/long queries
```

### Router

A small classification model that reads the query and selects which model to use:

```rust
pub struct ModelRouter {
    classifier: SmallModel,   // e.g. a fine-tuned Phi-3.5 mini
    models: HashMap<ModelId, Arc<Model>>,
}

impl ModelRouter {
    pub fn route(&self, query: &str) -> ModelId {
        let features = self.classifier.classify(query);
        // features: {is_code, is_complex, is_short, estimated_tokens}
        match features {
            f if f.is_code           => ModelId::Coder,
            f if f.estimated_tokens > 50000 => ModelId::Large,
            f if f.is_short          => ModelId::Fast,
            _                        => ModelId::General,
        }
    }
}
```

The router classifier can itself be a simple fine-tuned model or a rule-based system (detect code blocks, estimate complexity from question words).

### Cascading

Try small first, escalate on low confidence:

```rust
pub struct CascadeRunner {
    models: Vec<(Arc<Model>, f32)>,   // (model, confidence_threshold)
}

impl CascadeRunner {
    pub async fn run(&self, prompt: &str) -> (String, ModelId) {
        for (model, threshold) in &self.models {
            let (response, confidence) = model.generate_with_confidence(prompt);
            if confidence >= *threshold {
                return (response, model.id());
            }
            // Append response as context for next model? Or discard?
        }
        // Last model always returns (no threshold)
        let (response, _) = self.models.last().unwrap().0.generate(prompt);
        (response, self.models.last().unwrap().0.id())
    }
}
```

Confidence estimation options:
- Token probability: `exp(logprob_sum / n_tokens)` — cheap
- Self-consistency: sample twice, check agreement — more reliable
- Trained verifier — most accurate, requires training

### Self-MoA (Multi-Sample Synthesis)

Sample the same model multiple times, synthesize a better answer:

```rust
pub struct SelfMoa {
    model: Arc<Model>,
    n_samples: usize,   // typically 3
}

impl SelfMoa {
    pub fn generate(&self, prompt: &str) -> String {
        // Generate n independent samples
        let samples: Vec<String> = (0..self.n_samples)
            .map(|_| self.model.generate(prompt, &SamplerConfig {
                temperature: 0.8,   // some variation between samples
                ..Default::default()
            }))
            .collect();

        // Synthesize: feed all samples back to same model as aggregator
        let synthesis_prompt = format!(
            "You have been asked: {}\n\n\
             Here are {} draft responses:\n{}\n\n\
             Synthesize the best answer, combining insights from all drafts:",
            prompt,
            self.n_samples,
            samples.iter().enumerate()
                .map(|(i, s)| format!("Draft {}: {}", i+1, s))
                .collect::<Vec<_>>().join("\n\n")
        );

        self.model.generate(&synthesis_prompt, &SamplerConfig {
            temperature: 0.3,   // lower temperature for synthesis
            ..Default::default()
        })
    }
}
```

Research finding: Self-MoA often outperforms multi-model MoA because quality consistency within one good model beats averaging with weaker models.

---

## Prefix Caching

Avoid recomputing KV cache for shared prefixes (system prompts, conversation history).

```rust
pub struct PrefixCache {
    // Key: hash of token sequence
    // Value: KV cache snapshot at that position
    entries: LruCache<u64, KvCacheSnapshot>,
    max_entries: usize,
}

impl PrefixCache {
    pub fn lookup(&self, tokens: &[u32]) -> Option<(usize, &KvCacheSnapshot)> {
        // Find longest cached prefix
        for len in (1..=tokens.len()).rev() {
            let hash = hash_tokens(&tokens[..len]);
            if let Some(snapshot) = self.entries.get(&hash) {
                return Some((len, snapshot));
            }
        }
        None
    }

    pub fn store(&mut self, tokens: &[u32], kv: &KvCache, up_to: usize) {
        let hash = hash_tokens(&tokens[..up_to]);
        self.entries.put(hash, kv.snapshot(up_to));
    }
}
```

In the generate loop:
```rust
// Check cache before prefill
if let Some((cached_len, snapshot)) = prefix_cache.lookup(&tokens) {
    kv.restore(snapshot);
    // only prefill tokens[cached_len..] instead of all tokens
    prefill_from(model, &tokens[cached_len..], cached_len, kv);
} else {
    prefill(model, &tokens, kv);
    prefix_cache.store(&tokens, kv, tokens.len());
}
```

Impact: for a 2000-token system prompt reused across turns, prefix caching reduces prefill from 2000+ tokens to just the new user message. 90%+ latency reduction on first token.

---

## Speedup Stack — Combined Effect

On Strix Halo running M2.7:

```
Baseline (CPU only, naive):         ~2 tok/s
+ ROCm basic kernels:               ~8 tok/s
+ Fused dequant + APU tuning:       ~20 tok/s
+ TurboQuant (3-bit KV):            ~22 tok/s  (same speed, 5× more context)
+ Expert locality scheduling:       ~24 tok/s
+ Speculative (EAGLE head, 3× acc): ~60 tok/s effective
+ Prefix caching (long sessions):   first token much faster

For 7B model:
Baseline:                           ~5 tok/s
+ ROCm fused kernels:               ~35 tok/s
+ Speculative decoding:             ~80 tok/s effective
+ Prefix caching:                   ~5ms first token (vs ~200ms)
```

These stack multiplicatively. Speculative decoding doesn't change kernel efficiency — it changes how many tokens you get per forward pass. Bandwidth optimization doesn't change speculation — it makes each forward pass faster.

---

## Build Order for These Features

```
Phase 4:  Basic decode loop (no cache, no speculation)
Phase 5:  KV cache (f16, no compression)
Phase 6:  TurboQuant KV cache                     ← enables long context
Phase 7:  Speculative decoding, small model draft  ← 2-3× speedup
Phase 8:  EAGLE-3 draft head support              ← 3-4× speedup
Phase 9:  Prefix caching                          ← latency on turns
Phase 10: Router + multi-model serving            ← quality + efficiency
Phase 11: Self-MoA                                ← quality on hard queries
Phase 12: Cascading                               ← cost efficiency
```

Each phase adds concrete value independently. A user with Phase 6 complete has a working long-context engine. Phase 7 makes it fast. Phases 10–12 make it smart about when to use which model.
