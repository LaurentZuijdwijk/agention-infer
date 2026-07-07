# Build Roadmap

## Guiding Philosophy

Build correct before fast. Build simple before complete. Each phase ships something testable and useful. No phase exists solely as infrastructure for a later phase — every milestone produces a tool you can actually run against real models.

```
Phase 1:  gguf-info CLI                    → inspect any GGUF file
Phase 2:  CubeCL ops + dequant (CPU)       → correct kernels, run on CPU now GPU later
Phase 3:  tokenizer                        → encode/decode text
Phase 4:  single forward pass              → first token generation
Phase 5:  KV cache + decode loop           → full response generation
Phase 6:  MoE support                      → Mixtral, M2.7
Phase 7:  TurboQuant KV cache             → long context on M2.7
Phase 8:  CubeCL GPU promotion            → swap CPU client for ROCm, no rewrites
Phase 9:  speculative decoding            → 2-4× throughput
Phase 10: MoE routing visualization       → expert analysis tool
Phase 11: multi-model server              → OpenAI-compatible API
```

---

## Phase 1 — GGUF Inspection Tool

**Goal**: `gguf-info` CLI that correctly parses and displays any GGUF file.

**Milestone**: Run against Llama 3.2 1B, Qwen2.5 7B, and Mixtral 8×7B. Output matches llama.cpp's metadata display.

### 1a: Parser

Files: `src/error.rs`, `src/types.rs`, `src/parser.rs`, `src/loader.rs`

- [ ] `GgufError` enum with `thiserror`
- [ ] `GgmlType` enum for all quantization types
- [ ] `MetadataValue` enum for all 13 GGUF value types
- [ ] `TensorInfo` struct (name, shape, dtype, offset)
- [ ] `GgufFile` struct (version, metadata HashMap, tensor Vec)
- [ ] `parse(&[u8]) -> Result<GgufFile>` — pure, no I/O
- [ ] `load_mmap(path) -> Result<Mmap>` — only I/O function
- [ ] Unit tests with hand-crafted byte fixtures (no real files)

### 1b: Typed Model View

Files: `src/model.rs`

- [ ] `ModelInfo::from_gguf()` — typed accessors for common fields
- [ ] `MoeInfo` — expert_count, expert_used_count, ffn_length, shared_count
- [ ] Architecture prefix handling (llama., qwen2., minimax-m2., etc.)
- [ ] Memory budget calculator — `max_context_at_kv_bits(system_ram)`
- [ ] License field extraction

### 1c: CLI

Files: `src/bin/gguf-info.rs`

- [ ] `clap` argument parsing (`model` path, `--tensors` flag, `--verbose`)
- [ ] Summary display: architecture, params, quantization, context, MoE info
- [ ] Memory budget display: what context length fits at 16/8/3-bit KV
- [ ] Tensor list display (`--tensors`): name, shape, dtype, size
- [ ] Quantization distribution summary for dynamic quant models
- [ ] Multi-file GGUF detection and reporting

### 1d: Multi-file support

Files: `src/loader.rs`

- [ ] Detect split files (`-00001-of-00004` suffix pattern)
- [ ] `MultiFileMmap` struct — virtual address space across files
- [ ] Tensor offset resolution across file boundaries

**Test**: `gguf-info MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf` correctly reports 256 experts, 80 layers, memory budget.

---

## Phase 2 — Math Primitives (CubeCL from day one)

**Goal**: All 6 core operations implemented as `#[cube]` kernels, validated
against numpy reference, running on CPU via CubeCL's CPU backend.

**Why CubeCL in Phase 2, not Phase 8**: Writing ops as plain Rust then
rewriting as `#[cube]` in Phase 8 means doing the work twice. CubeCL's CPU
backend runs on any machine, produces correct output, and the same kernels
promote to ROCm/Vulkan in Phase 8 with zero changes. Write once, validate on
CPU, deploy on GPU.

**Milestone**: Python script computes matmul/norm/softmax/rope on known inputs.
CubeCL CPU backend produces matching results to 1e-4 tolerance.

### 2a: CubeCL Setup

Files: `Cargo.toml`, `src/ops/mod.rs`

- [ ] Add `cubecl`, `cubecl-wgpu` to Cargo.toml (Vulkan fallback, works everywhere)
- [ ] Add `cubecl-hip` under `cfg(target_os = "linux")` (ROCm, Phase 8 ready)
- [ ] `ComputeClient` factory — auto-selects CPU backend for now
- [ ] `GpuBuffer` wrapper — typed handle to CubeCL device memory
- [ ] `Backend` trait with `client()` accessor

### 2b: Dequantization

Files: `src/quant/`

Dequantization runs on CPU as plain Rust — it feeds data into CubeCL kernels
but doesn't need to be a `#[cube]` function itself. The fused dequant+matmul
kernel in 2c handles the GPU path.

- [ ] `F32` — memcpy/cast
- [ ] `F16` — `half::f16::to_f32()`
- [ ] `BF16` — bfloat16 to f32
- [ ] `Q8_0` — 34 bytes → 32 f32 values
- [ ] `Q4_K` — 140 bytes → 256 f32 values (superblock + scale unpacking)
- [ ] `Q6_K` — 210 bytes → 256 f32 values
- [ ] `IQ4_XS` — codebook lookup (verify codebook against llama.cpp source)
- [ ] `IQ3_XXS` — codebook lookup
- [ ] `MXFP4` — E2M1 + E8M0 scale, bit manipulation

### 2c: CubeCL Kernels

Files: `src/ops/kernels.rs`

All ops written as `#[cube]` functions. Run on CPU client now, GPU later —
no code changes needed to promote.

- [ ] `matmul_f32` — baseline f32 matmul, validates kernel infrastructure
- [ ] `matmul_q8_0` — fused dequant Q8_0 + dot product in one kernel
- [ ] `matmul_q4_k` — fused dequant Q4_K superblock + dot product
- [ ] `matmul_iq4_xs` — fused codebook lookup + dot product
- [ ] `rms_norm` — numerically stable, reads weight array
- [ ] `softmax` — numerically stable (max subtraction before exp)
- [ ] `rope` — per-head rotation, YaRN scaling flag
- [ ] `silu_mul` — elementwise gate * silu(up)
- [ ] `add` — in-place residual addition

### 2d: CubeCL Autotuning Hook

Files: `src/ops/mod.rs`

- [ ] Call `client.autotune()` at engine startup
- [ ] Cache tuning results to disk (`~/.cache/gguf-rs/autotune-{device}.json`)
- [ ] Log chosen configuration (workgroup size, vectorization width) at startup
- [ ] Skip autotuning if cache exists and device matches

**Why now**: Autotuning is a one-line addition on the `ComputeClient`. Adding
it in Phase 2 means every subsequent phase benefits from it automatically —
including the CPU backend, which the autotuner will vectorize correctly.

### 2e: Kernel Validation

Files: `tests/ops_tests.rs`, `tests/fixtures/generate.py`

- [ ] Generate numpy reference values: `python tests/fixtures/generate.py`
- [ ] `matmul_f32` vs numpy `@` operator to 1e-5
- [ ] `matmul_q8_0` — dequant round-trip + matmul vs reference to 1e-3
- [ ] `rms_norm` vs numpy reference to 1e-5
- [ ] `softmax` vs numpy, sum-to-one, numerical stability with large inputs
- [ ] `rope` vs numpy reference to 1e-4
- [ ] `silu_mul` vs numpy reference to 1e-5
- [ ] No NaN/Inf on zero inputs for any kernel
- [ ] CPU client output == Vulkan client output (if Vulkan available)

**Test**: `cargo test` passes with zero model files on any developer machine.

---

## Phase 3 — Tokenizer

**Goal**: Encode and decode text using vocabulary from GGUF metadata.

**Milestone**: `echo "Hello world" | tokenize --model llama.gguf | detokenize --model llama.gguf` outputs "Hello world" unchanged.

Files: `src/tokenizer.rs`

- [ ] Load vocabulary from `tokenizer.ggml.tokens` and `tokenizer.ggml.scores`
- [ ] BPE merge algorithm
- [ ] Special token handling (BOS, EOS, PAD, UNK)
- [ ] Encode: `String → Vec<u32>`
- [ ] Decode: `Vec<u32> → String`
- [ ] Chat template rendering (basic Jinja2 subset — if/for/set)
- [ ] Interleaved thinking token detection (`<think>`, `</think>`)

---

## Phase 4 — Single Forward Pass (CPU)

**Goal**: Run one complete forward pass through a real model and produce logits.

**Milestone**: On Llama 3.2 1B Q8_0, given token "Hello", produce logits. Top token matches llama.cpp output at temperature=0.

Files: `src/model/llama.rs`, `src/attention.rs`

### 4a: Dense model

- [ ] Weight loading from tensor map by name
- [ ] Embedding lookup (`token_embd.weight` row access)
- [ ] Single layer forward: `attn_norm → q/k/v matmuls → rope → attention → o_proj → add → ffn_norm → gate/up → silu_mul → down → add`
- [ ] GQA broadcast (n_heads / n_kv_heads grouping)
- [ ] Full stack: 32 layers → output_norm → lm_head → logits

### 4b: Attention kernel

- [ ] Naive attention (full score matrix, no caching) — for testing
- [ ] Flash attention (tiled, no full matrix) — for production

### 4c: Sampler

Files: `src/sampler.rs`

- [ ] Greedy decode (argmax)
- [ ] Temperature scaling
- [ ] Min-P filtering
- [ ] Top-K filtering
- [ ] Repetition penalty
- [ ] Multinomial sampling

**Test**: At temperature=0, output token matches llama.cpp on same model + prompt.

---

## Phase 5 — KV Cache & Decode Loop

**Goal**: Generate a full response (not just one token).

**Milestone**: `engine --model llama.gguf --prompt "Tell me about" --max-tokens 100` generates coherent 100-token response.

Files: `src/kv_cache.rs`, decode loop in model

- [ ] `KvCache` with f16 storage (no TurboQuant yet)
- [ ] Write K/V at each position during forward pass
- [ ] Read K/V for attention over full context
- [ ] Decode loop: prefill → autoregressive decode → stop on EOS
- [ ] Stop sequence detection
- [ ] `TokenEvent` streaming (token-by-token output, not buffered)
- [ ] Max context enforcement

---

## Phase 6 — MoE Support

**Goal**: Run Mixtral 8×7B and MiniMax M2.7.

**Milestone**: Mixtral produces correct output. M2.7 (at IQ3_XXS) runs and produces coherent text.

Files: `src/model/moe.rs`, `src/model/minimax.rs`

- [ ] MoE forward pass: router → top-k → expert dispatch → weighted sum
- [ ] Shared experts (DeepSeek style)
- [ ] Expert locality scheduling (sort by byte offset)
- [ ] Multi-file tensor loading for M2.7
- [ ] `minimax-m2` architecture handler (may differ from llama in attn details)
- [ ] Interleaved thinking token stream for M2.7

---

## Phase 7 — TurboQuant KV Cache

**Goal**: 3-bit KV cache enabling long context on M2.7.

**Milestone**: M2.7 generates coherent text at 32K context within 96 GB.

Files: `src/kv_cache.rs` (replace f16 storage)

- [ ] Walsh-Hadamard Transform (in-place, power-of-2 head_dim)
- [ ] Lloyd-Max 2-bit codebook (precomputed for standard normal)
- [ ] `KvCache::write` — WHT + quantize + pack
- [ ] `KvCache::read` — unpack + dequantize + inverse WHT
- [ ] Memory budget enforcement: refuse to start if context > available memory
- [ ] QJL residual correction (optional, behind flag)
- [ ] Benchmark: measure quality vs 16-bit KV on needle-in-haystack

---

## Phase 8 — CubeCL GPU Promotion

**Goal**: The kernels written in Phase 2 run on the ROCm GPU. No kernel
rewrites — just swap the `ComputeClient` from CPU to ROCm.

**Milestone**: 7B model at 30+ tok/s. M2.7 at 15+ tok/s. GPU output matches
CPU output token-for-token (greedy).

Files: `src/ops/mod.rs` (backend selection), `src/loader.rs` (unified memory)

### 8a: ROCm Client

- [ ] Detect ROCm at startup (`HSA_OVERRIDE_GFX_VERSION=11.0.0` for gfx1151)
- [ ] `RocmBackend::new(device_id)` — creates ROCm `ComputeClient`
- [ ] Autotuning runs on GPU at first launch, caches wave32-optimal config
- [ ] GPU client replaces CPU client in `create_backend()` auto path

### 8b: Unified Memory

- [ ] `hipMallocManaged` for activation buffers (already in CubeCL ROCm backend)
- [ ] Verify mmap'd weight pages accessible from GPU without `hipMemcpy`
- [ ] Weight tensor `data` pointer passed directly to `#[cube]` kernels
- [ ] Profile: confirm zero copies for weight access on Strix Halo

### 8c: Validation

- [ ] Each `#[cube]` kernel: GPU output matches CPU output to 1e-3 tolerance
- [ ] Golden test: GPU greedy decode matches CPU greedy decode token-for-token
- [ ] No GPU memory leaks across 100+ forward passes (track alloc/free)
- [ ] Benchmark: record tok/s at this point as v0.1 baseline

### 8d: Vulkan Fallback

- [ ] `VulkanBackend` via `cubecl-wgpu` — same kernels, different client
- [ ] Test on any Vulkan-capable GPU (even integrated)
- [ ] Documents Windows path for future users

---

## Phase 9 — Speculative Decoding

**Goal**: 2–4× effective token throughput.

**Milestone**: 7B model with 1B draft at 60+ effective tok/s.

Files: `src/speculative.rs`

- [ ] `DraftSource` trait
- [ ] `SmallModelDraft` — separate loaded model proposes tokens
- [ ] `LayerSkipDraft` — same model, skip middle layers
- [ ] Parallel verification in target model
- [ ] Accept/reject with target's token on first mismatch
- [ ] Metrics: acceptance rate, effective tok/s vs non-speculative

---

## Phase 10 — MoE Routing Visualization

**Goal**: `trace` binary showing expert activation patterns.

**Milestone**: Running `trace` on Mixtral with a code prompt shows expert specialization pattern.

Files: `src/bin/trace.rs`, trace infrastructure in `src/model/moe.rs`

- [ ] `LayerTrace` and `ForwardTrace` structs
- [ ] Tracing hook in MoE forward pass (zero overhead when disabled)
- [ ] JSON export of full routing trace
- [ ] Terminal heatmap renderer (`ratatui`)
- [ ] Expert activation frequency analysis
- [ ] Router entropy per layer per token
- [ ] Two-prompt diff mode

---

## Phase 11 — Multi-Model & Server

**Goal**: OpenAI-compatible HTTP API. Multiple models served simultaneously.

**Milestone**: `curl http://localhost:8080/v1/chat/completions -d '{"model": "qwen7b", ...}'` returns correct response.

Files: `src/bin/engine.rs`, `src/multi_model.rs`

- [ ] `axum`-based HTTP server
- [ ] OpenAI `/v1/chat/completions` compatible endpoint
- [ ] Multiple models loaded into unified memory pool
- [ ] Simple rule-based router (code → coder model, etc.)
- [ ] Cascading runner with confidence threshold
- [ ] Self-MoA for high-stakes queries
- [ ] Prefix caching
- [ ] Streaming SSE response

---

## Testing Strategy

```
Unit tests (no real model files, no GPU):
  parser_tests.rs    — hand-crafted GGUF byte fixtures
  quant_tests.rs     — dequant round-trip: quant → dequant ≈ original
  ops_tests.rs       — each #[cube] op vs numpy reference (CPU client)
  turbo_quant_tests  — WHT properties, compression ratio, cosine similarity

Integration tests (require model files, gated behind --features integration):
  forward_pass_tests — full forward pass vs llama.cpp, temperature=0
  tokenizer_tests    — encode → decode round-trip, matches llama.cpp

Performance benchmarks (manual, before releases):
  bench/matmul.rs    — tok/s per quant type, CPU and GPU
  bench/attention.rs — attention throughput vs context length
  bench/e2e.rs       — end-to-end tok/s on each target model
```

Golden test: the most valuable integration test. At temperature=0 (greedy),
output token sequence must match llama.cpp token-for-token on the same model,
prompt, and context length. Any divergence = a bug in math, quantization,
or architecture. Run on both CPU client and ROCm client — outputs must match.

See `11-testing.md` for full test implementation details.

---

## Dependency Summary

```toml
[dependencies]
# Core
memmap2   = "0.9"        # memory-mapped files
half      = "2.3"        # f16 / bf16 types
bytemuck  = "1.14"       # safe byte casting

# GPU compute — one kernel set, all hardware (no C++ required)
cubecl      = { version = "0.3", features = ["std"] }
cubecl-wgpu = "0.3"      # Vulkan fallback (everywhere)

# Error handling
thiserror = "1.0"        # typed errors for the library
anyhow    = "1.0"        # ergonomic errors for binaries

# CLI
clap = { version = "4.4", features = ["derive"] }

# Server (Phase 11)
axum       = "0.7"
tokio      = { version = "1", features = ["full"] }
serde      = { version = "1", features = ["derive"] }
serde_json = "1"

# TUI (Phase 10)
ratatui = "0.26"

# ROCm backend (Linux only)
[target.'cfg(target_os = "linux")'.dependencies]
cubecl-hip = "0.3"

# Metal backend (macOS — bonus portability)
[target.'cfg(target_os = "macos")'.dependencies]
cubecl-metal = "0.3"
```

No `[build-dependencies]`. No `hipcc`. No C++ toolchain required.
