# Architecture

> ⚠️ **Partly aspirational / out of date.** This document describes an intended design; parts of it
> do not match the tree (e.g. the crate layout lists files like `attention.rs`, `kv_cache.rs`,
> `moe.rs`, `engine.rs` that don't exist; the compute stack is hand-written **wgpu/WGSL + CPU**, not
> the pure-CubeCL-ROCm path described; ROCm/HIP is stubbed to CPU fallback). Trust the code and
> [`docs/roadmap/`](roadmap/) over this file. Kept for design intent; a full rewrite is deferred.

## Guiding Principles

**Parsing is pure.** The GGUF parser takes `&[u8]` and returns a typed data structure. No I/O inside the parser. Fully testable without touching disk.

**Zero copy.** Tensor data is never copied out of the mmap. A `Tensor` is a name, a shape, a dtype, and a pointer into the mmap. Dequantization happens block-by-block at compute time.

**One kernel, every backend.** GPU kernels are written once in Rust using the `#[cube]` macro (CubeCL). The same source compiles to ROCm HIP, CPU SIMD, Vulkan, and Metal. No C++, no FFI, no separate codebases to maintain.

**Backends are swappable.** The forward pass calls `backend.matmul(...)`. Whether that dispatches to a CubeCL CPU kernel, a ROCm HIP kernel, or a Vulkan compute shader is invisible to the model logic.

**Correctness before speed.** The CPU backend is always present, always correct. GPU backends are validated against it before use.

**Fail loudly.** Corrupt files, unsupported quantization types, missing metadata — all produce typed errors with clear messages, not panics or silent wrong output.

---

## GPU Compute: CubeCL

The original plan was to write HIP kernels in C++ and call them via FFI. We use CubeCL instead.

CubeCL is a multi-platform high-performance GPU compute language extension for Rust. You annotate Rust functions with `#[cube]` and the CubeCL compiler translates them to HIP, PTX, WGSL, or MSL depending on the target backend.

```rust
// Write once in Rust
#[cube(launch)]
fn matmul_q8_0<F: Float>(
    x:   &Array<F>,
    w:   &Array<u8>,     // Q8_0 packed blocks
    out: &mut Array<F>,
    #[comptime] n_blocks: u32,
) {
    let row = ABSOLUTE_POS_X;
    let mut sum = F::new(0.0);
    for b in 0..n_blocks {
        let delta = unpack_f16(&w[row * n_blocks * 34 + b * 34]);
        for i in 0u32..32u32 {
            sum += delta * (w[row * n_blocks * 34 + b * 34 + 2 + i] as i8) as F
                        * x[b * 32 + i];
        }
    }
    out[row] = sum;
}

// Run on CPU
let client = CubeCL::cpu_client();
matmul_q8_0::launch(&client, cube_count, cube_dim, x_arg, w_arg, out_arg, n_blocks);

// Run on ROCm — identical call
let client = CubeCL::hip_client(0);
matmul_q8_0::launch(&client, cube_count, cube_dim, x_arg, w_arg, out_arg, n_blocks);
```

### Why CubeCL Changes Everything

**No C++ compiler dependency.** No `hipcc`, no `build.rs` invoking external tools, no FFI bindings. Pure Rust.

**Automatic hardware tuning.** CubeCL benchmarks kernel configurations at startup and picks the fastest for your specific device. On Strix Halo (gfx1151), this automatically discovers optimal wave32 workgroup sizes — the single biggest performance gap in llama.cpp's ROCm backend.

**CPU + GPU from same source.** The CPU backend uses CubeCL's MLIR/LLVM compiler with automatic SIMD vectorization — the same kernel that runs on GPU runs on CPU with full AVX2 utilization. One correctness reference, not two separate implementations.

**Free portability.** Swap `cubecl-hip` for `cubecl-wgpu` and the same kernels run on Vulkan (Windows, Linux, Android). Swap for `cubecl-metal` and they run on Apple Silicon. Your engine targets every unified memory APU platform from one codebase.

**State-of-the-art matmul.** CubeCL's matmul engine matches cuBLAS and CUTLASS performance, including the batched vector-matrix product kernel used in LLM decode — optimized specifically for this workload with tensor core support.

---

## Crate Layout

```
gguf-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs                  ← public API surface
│   ├── error.rs                ← GgufError enum (thiserror)
│   ├── types.rs                ← GgmlType, MetadataValue, TensorInfo, GgufFile
│   ├── parser.rs               ← parse(&[u8]) -> Result<GgufFile>  pure
│   ├── loader.rs               ← load_mmap(path) -> Result<Mmap>   I/O only here
│   ├── model.rs                ← ModelInfo, MoeInfo — typed metadata view
│   ├── tokenizer.rs            ← BPE tokenizer from GGUF vocab
│   ├── quant/
│   │   ├── mod.rs              ← dequant dispatch
│   │   ├── q8_0.rs
│   │   ├── q4_k.rs
│   │   ├── q6_k.rs
│   │   ├── iq4_xs.rs
│   │   ├── iq3_xxs.rs
│   │   └── mxfp4.rs
│   ├── ops/
│   │   ├── mod.rs              ← Backend trait + ComputeClient wrapper
│   │   └── kernels.rs          ← ALL kernels as #[cube] Rust functions
│   │                              runs on CPU, ROCm, Vulkan, Metal
│   ├── attention.rs            ← flash attention (calls kernels.rs)
│   ├── kv_cache.rs             ← KvCache + TurboQuant compression
│   ├── model/
│   │   ├── mod.rs              ← Model trait
│   │   ├── llama.rs            ← Llama / Mistral / Qwen forward pass
│   │   ├── moe.rs              ← MoE layer, expert dispatch
│   │   └── minimax.rs          ← MiniMax M2.7 specifics
│   ├── sampler.rs              ← temperature, top-p, min-p, rep penalty
│   ├── speculative.rs          ← speculative decoding, EAGLE draft
│   └── multi_model.rs          ← router, cascading, Self-MoA
└── src/bin/
    ├── gguf-info.rs            ← inspection CLI
    ├── engine.rs               ← inference server (OpenAI-compatible API)
    └── trace.rs                ← MoE routing visualizer
└── tests/
    ├── parser_tests.rs         ← hand-built byte fixtures, no real files
    ├── quant_tests.rs          ← dequant round-trip validation
    ├── ops_tests.rs            ← matmul/norm/rope vs numpy reference
    └── golden_tests.rs         ← full model output vs llama.cpp reference
```

Note: no `kernels/` directory of `.hip` files. All compute is in `src/ops/kernels.rs`.

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Binaries                                               │
│  gguf-info    engine (HTTP API)    trace (MoE viz)      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Orchestration                                          │
│  multi_model.rs  — router, cascading, Self-MoA          │
│  speculative.rs  — draft/verify loop                    │
│  sampler.rs      — temperature, top-p, min-p            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Model Forward Pass                                     │
│  model/llama.rs  — transformer layers                   │
│  model/moe.rs    — expert routing + dispatch            │
│  attention.rs    — flash attention, local/global        │
│  kv_cache.rs     — TurboQuant compressed cache          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Compute — CubeCL                                       │
│  ops/kernels.rs  — #[cube] Rust kernels                 │
│                    matmul (fused dequant)                │
│                    rms_norm, rope, softmax, silu_mul     │
│                    flash attention tiles                 │
│                    autotuned per device at startup       │
│                                                         │
│  Backend selection at runtime:                          │
│    cubecl-hip   → ROCm (Linux, Strix Halo)             │
│    cubecl-wgpu  → Vulkan (Windows, Linux fallback)      │
│    cubecl-metal → Metal (Apple Silicon)                 │
│    cubecl cpu   → CPU SIMD via MLIR/LLVM                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  File / Memory Layer                                    │
│  loader.rs       — mmap into unified memory             │
│  parser.rs       — bytes → GgufFile (pure)              │
│  types.rs        — GgufFile, TensorInfo, MetadataValue  │
│  model.rs        — ModelInfo, MoeInfo                   │
└─────────────────────────────────────────────────────────┘
```

---

## The Backend Trait

```rust
pub trait Backend: Send + Sync {
    fn client(&self) -> &ComputeClient;
    fn name(&self) -> &str;

    // Convenience wrappers around CubeCL kernel launches
    fn matmul(&self, out: &mut GpuBuffer, x: &GpuBuffer, w: &Tensor);
    fn rms_norm(&self, out: &mut GpuBuffer, x: &GpuBuffer, weight: &GpuBuffer, eps: f32);
    fn rope(&self, q: &mut GpuBuffer, k: &mut GpuBuffer, pos: usize, cfg: &RopeConfig);
    fn softmax(&self, x: &mut GpuBuffer);
    fn silu_mul(&self, out: &mut GpuBuffer, gate: &GpuBuffer, up: &GpuBuffer);
    fn add(&self, a: &mut GpuBuffer, b: &GpuBuffer);
}

// All backends share the same kernel source — only ComputeClient differs
pub struct CpuBackend   { client: ComputeClient }
pub struct RocmBackend  { client: ComputeClient }
pub struct VulkanBackend { client: ComputeClient }

// Backend selection at startup
pub fn create_backend(preference: BackendPreference) -> Box<dyn Backend> {
    match preference {
        BackendPreference::Auto => {
            // Try ROCm → Vulkan → CPU in order
            if rocm_available() { return Box::new(RocmBackend::new(0)); }
            if vulkan_available() { return Box::new(VulkanBackend::new()); }
            Box::new(CpuBackend::new())
        }
        BackendPreference::Cpu    => Box::new(CpuBackend::new()),
        BackendPreference::Rocm   => Box::new(RocmBackend::new(0)),
        BackendPreference::Vulkan => Box::new(VulkanBackend::new()),
    }
}
```

---

## Unified Memory on Strix Halo

CubeCL's `hipMallocManaged` unified memory integration means weights read
directly from mmap'd pages — no staging copies:

```rust
// loader.rs — weights mmap'd into unified address space
pub fn load_unified(path: &Path) -> Result<UnifiedMmap> {
    let file = File::open(path)?;
    // On Strix Halo: this memory is physically accessible by both CPU and GPU
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(UnifiedMmap(mmap))
}

// ops/kernels.rs — GPU reads directly from mmap pointer
#[cube(launch)]
fn matmul_q4_k<F: Float>(
    x:   &Array<F>,
    w:   &Array<u8>,   // ← pointer into mmap, no copy needed on APU
    out: &mut Array<F>,
    // ...
) { ... }
```

On discrete GPUs, `w` would need to be copied to VRAM first. On Strix Halo, it's already there.

---

## Cargo.toml

```toml
[package]
name    = "gguf-rs"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "gguf-info"
path = "src/bin/gguf-info.rs"

[[bin]]
name = "engine"
path = "src/bin/engine.rs"

[[bin]]
name = "trace"
path = "src/bin/trace.rs"

[dependencies]
# File loading
memmap2   = "0.9"
half      = "2.3"          # f16 / bf16
bytemuck  = "1.14"         # safe byte casting

# GPU compute — one set of kernels, multiple backends
cubecl    = { version = "0.3", features = ["std"] }

# Error handling
thiserror = "1.0"          # typed errors for lib
anyhow    = "1.0"          # ergonomic errors for bins

# CLI
clap = { version = "4.4", features = ["derive"] }

# Server (Phase 11)
axum       = "0.7"
tokio      = { version = "1", features = ["full"] }
serde      = { version = "1", features = ["derive"] }
serde_json = "1"

# TUI (Phase 10)
ratatui = "0.26"

# ROCm backend (Linux)
[target.'cfg(target_os = "linux")'.dependencies]
cubecl-hip  = "0.3"

# Vulkan backend (everywhere)
[dependencies]
cubecl-wgpu = "0.3"

# Metal backend (macOS — bonus portability)
[target.'cfg(target_os = "macos")'.dependencies]
cubecl-metal = "0.3"
```

No `[build-dependencies]`. No `hipcc`. No C++ toolchain required.

---

## Data Flow — Single Token Decode

```
token: u32
    │
    ▼
embed(token, &token_embd)           → x: GpuBuffer [d_model]
    │
    ▼  ×n_layers
┌─────────────────────────────────────┐
│  xn = backend.rms_norm(x, norm)    │
│  q  = backend.matmul(xn, wq)       │
│  k  = backend.matmul(xn, wk)       │
│  v  = backend.matmul(xn, wv)       │
│  backend.rope(q, k, pos, cfg)      │
│  kv_cache.write(layer, pos, k, v)  │  ← TurboQuant compression
│  (k_all, v_all) = kv_cache.read()  │  ← decompress on read
│  attn = flash_attention(q,k_all,v) │  ← CubeCL tiled kernel
│  out  = backend.matmul(attn, wo)   │
│  backend.add(x, out)               │  ← residual
│                                    │
│  xn   = backend.rms_norm(x, norm) │
│  [dense or MoE FFN]                │
│  backend.add(x, ffn_out)           │  ← residual
└─────────────────────────────────────┘
    │
    ▼
xn     = backend.rms_norm(x, output_norm)
logits = backend.matmul(xn, lm_head)    → [vocab_size]
    │
    ▼
sampler(logits)                          → next_token: u32
```

---

## Error Design

Two-tier error handling:

```rust
// Library (types.rs, parser.rs, model.rs, ops/):
//   Precise typed errors — callers can match on them
#[derive(Error, Debug)]
pub enum GgufError {
    InvalidMagic,
    UnsupportedVersion(u32),
    UnexpectedEof(usize),
    UnknownMetadataType(u32),
    UnknownTensorType(u32),
    InvalidString(usize),
    MissingMetadata(String),
    WrongMetadataType(String, &'static str),
    UnsupportedArchitecture(String),
    MultiFileMismatch,
    BackendError(String),
}

// Binaries (bin/gguf-info.rs, bin/engine.rs):
//   anyhow::Result everywhere. GgufError converts automatically via From.
//   Print and exit. No matching needed.
```
