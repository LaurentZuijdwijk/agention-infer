# MoE Models & Routing Visualization

## Supported MoE Architectures

| Model | Experts | Active | Shared | Architecture |
|---|---|---|---|---|
| Mixtral 8×7B | 8 | 2 | 0 | mixtral |
| Mixtral 8×22B | 8 | 2 | 0 | mixtral |
| Qwen2.5 MoE | 64 | 8 | 0 | qwen2moe |
| DeepSeek V3 | 256 | 8 | 1 | deepseek2 |
| MiniMax M2.7 | 256 | 8 | 0 | minimax-m2 |
| Phi-3.5 MoE | 16 | 2 | 0 | phi3 |

---

## How MoE Changes the Forward Pass

Only the FFN block changes. Attention is identical to dense models.

```
Dense FFN:
  gate  = matmul(x, ffn_gate)     [ffn_dim]
  up    = matmul(x, ffn_up)       [ffn_dim]
  h     = gate * silu(up)         [ffn_dim]
  out   = matmul(h, ffn_down)     [d_model]
  (one set of weights per layer)

MoE FFN:
  scores         = matmul(x, router)         [n_experts]
  probs          = softmax(scores)
  top_k_indices  = argtop_k(probs, k)
  out = Σ probs[i] * expert_ffn(x, i)   for i in top_k_indices
  (256 sets of weights per layer, only 8 run per token)
```

### Memory Layout in GGUF

llama.cpp packs all expert tensors for a layer into a single 3D tensor:

```
blk.{i}.ffn_gate_inp.weight    [n_experts, d_model]         router weights
blk.{i}.ffn_gate_exps.weight   [n_experts, ffn_dim, d_model] all gate experts
blk.{i}.ffn_up_exps.weight     [n_experts, ffn_dim, d_model]
blk.{i}.ffn_down_exps.weight   [n_experts, d_model, ffn_dim]
```

For M2.7 (256 experts, d_model=7168, ffn_dim=2048):
```
ffn_gate_exps: [256, 2048, 7168] at IQ4_XS ≈ 18.4 GB per tensor
               × 3 tensors per layer × 80 layers ≈ most of the 108 GB
```

Accessing expert `i`: slice `w[i * ffn_dim * d_model .. (i+1) * ffn_dim * d_model]`

---

## Tracing Infrastructure

The trace system captures routing decisions without impacting the main inference path when disabled.

```rust
#[derive(Debug, Clone)]
pub struct LayerTrace {
    pub layer_idx:        usize,
    pub token_pos:        usize,
    pub token_id:         u32,
    pub token_str:        String,
    pub router_scores:    Vec<f32>,     // [n_experts] raw logits
    pub router_probs:     Vec<f32>,     // [n_experts] after softmax
    pub selected_experts: Vec<usize>,   // [k] indices
    pub selected_weights: Vec<f32>,     // [k] their softmax weights
    pub expert_byte_offsets: Vec<u64>,  // [k] for locality analysis
}

#[derive(Debug)]
pub struct ForwardTrace {
    pub model_name: String,
    pub n_experts:  usize,
    pub n_active:   usize,
    pub layers:     Vec<Vec<LayerTrace>>,  // [n_layers][n_tokens]
}

impl ForwardTrace {
    pub fn expert_activation_matrix(&self) -> Vec<Vec<f32>> {
        // Returns [n_tokens][n_experts] activation frequency
        // Value = sum of routing weights that expert received for this token
    }

    pub fn expert_specialization(&self, texts: &[String]) -> ExpertProfile {
        // Aggregate which token types each expert activates for
    }

    pub fn router_entropy_by_layer(&self) -> Vec<f32> {
        // Per-layer: average entropy of routing distribution
        // Low entropy = confident routing, few experts dominate
        // High entropy = uncertain routing, many experts similar
    }
}
```

The MoE forward pass checks for an active trace context with zero overhead when disabled:

```rust
fn moe_ffn(x: &[f32], layer: &MoeLayer,
           ctx: &mut InferenceContext) -> Vec<f32> {
    let scores = matmul(x, &layer.router_weight);
    let probs  = softmax_copy(&scores);
    let top_k  = top_k_indices(&probs, layer.n_active);

    // Trace: compile-time no-op when tracing disabled
    if let Some(trace) = ctx.trace.as_mut() {
        trace.record_layer(LayerTrace {
            layer_idx: layer.idx,
            router_scores: scores.clone(),
            router_probs:  probs.clone(),
            selected_experts: top_k.clone(),
            selected_weights: top_k.iter().map(|&i| probs[i]).collect(),
            ..ctx.current_token_info()
        });
    }

    // Normal dispatch
    dispatch_experts(x, layer, &top_k, &probs)
}
```

---

## The `trace` Binary

A standalone CLI tool that runs inference with tracing enabled and outputs structured data:

```bash
# JSON output for tooling
trace --model mixtral-8x7b.gguf \
      --prompt "def fibonacci(n):" \
      --format json > routing.json

# Terminal heatmap
trace --model mixtral-8x7b.gguf \
      --prompt "def fibonacci(n):" \
      --format terminal

# Compare two prompts
trace --model mixtral-8x7b.gguf \
      --prompt-a "def fibonacci(n):" \
      --prompt-b "Dear Sir, I am writing to" \
      --format diff
```

### Terminal Heatmap Output

```
MoE Routing — Mixtral 8×7B — "def fibonacci(n):"

Token        L0          L1          L2    ...  L31
             01234567    01234567    01234567
"def"        ..█.....    .█......    ....█...    ██......
" fib"       ..█.....    .█......    ...█....    .█......
"onacci"     ..█.....    ..█.....    ....█...    █.......
"("          ......█.    ......█.    ......██    ......█.
"n"          ...█....    ....█...    .....█..    .....█..
")"          ......█.    ......█.    ......██    ......█.
":"          .......█    .......█    .......█    .......█

Legend: █ = primary expert (weight > 0.6)
        ▓ = secondary expert (weight > 0.3)
        . = not selected

Expert specialization:
  Expert 2: code keywords (def, class, import)
  Expert 7: punctuation and delimiters
  Expert 5: identifiers
```

### JSON Schema

```json
{
  "model": "mixtral-8x7b-instruct",
  "n_experts": 8,
  "n_active": 2,
  "n_layers": 32,
  "tokens": [
    {
      "pos": 0,
      "id": 1984,
      "text": "def",
      "layers": [
        {
          "layer": 0,
          "router_scores": [-0.4, 1.2, 2.8, 0.1, -0.9, 0.3, -1.1, 0.7],
          "router_probs":  [0.05, 0.12, 0.61, 0.04, 0.01, 0.08, 0.01, 0.08],
          "selected": [2, 1],
          "weights":  [0.61, 0.12]
        }
      ]
    }
  ]
}
```

---

## Expert Pruning Analysis

The trace output enables expert pruning decisions:

```rust
pub fn analyze_expert_importance(trace: &ForwardTrace,
                                  calibration_texts: &[String]) -> ExpertImportance {
    let mut expert_activation_count = vec![0usize; trace.n_experts];
    let mut expert_total_weight = vec![0f32; trace.n_experts];

    for layer_traces in &trace.layers {
        for token_trace in layer_traces {
            for (&expert, &weight) in token_trace.selected_experts.iter()
                                        .zip(token_trace.selected_weights.iter()) {
                expert_activation_count[expert] += 1;
                expert_total_weight[expert] += weight;
            }
        }
    }

    ExpertImportance {
        activation_frequency: expert_activation_count,
        average_weight: expert_total_weight,
        // Low frequency + low weight = candidate for pruning
    }
}
```

This is how domain-specialized MoE variants are created: run the original model on a domain-specific corpus (code, medical, legal), identify which experts activate predominantly for that domain, prune the rest.

---

## MoE Interleaved Thinking (MiniMax M2.7)

M2.7 uses interleaved thinking — `<think>...</think>` blocks can appear throughout the response, not just at the start. This requires special handling in the token stream:

```rust
#[derive(Debug, Clone)]
pub enum TokenEvent {
    Output(u32),              // normal output token
    ThinkStart,               // <think> detected
    ThinkToken(u32),          // token inside thinking block
    ThinkEnd,                 // </think> detected
    End(StopReason),
}

pub enum StopReason {
    Eos,
    StopSequence(String),
    MaxTokens,
    ContextFull,
}
```

Important: thinking tokens must be preserved in conversation history for optimal performance on subsequent turns. They are not display artifacts — they represent the model's reasoning state.

The trace tool marks thinking tokens distinctly in the routing visualization, often showing different expert activation patterns during thinking vs output phases. This is itself an interesting interpretability signal.

---

## Routing Quality Metrics

Exposed by the trace tool and optionally logged during inference:

```
Router entropy:
  H = -Σ p_i log(p_i) for selected experts
  Low → model is confident about routing (well-trained)
  High → model is uncertain, many experts similar

Load balance:
  Fraction of tokens routed to each expert over time
  Well-trained: roughly uniform (no expert starved)
  Poorly trained: some experts dominate, others idle

Expert specialization score:
  KL divergence between expert's token-type distribution
  and the overall token-type distribution
  High → expert specializes in specific token types
  Low  → expert is generalist

Routing consistency:
  For the same token in similar contexts,
  do the same experts fire?
  High consistency → routing is stable and meaningful
```
