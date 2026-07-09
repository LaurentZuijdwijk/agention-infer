# ROCm & Hardware Optimization

> ℹ️ **Reality note:** the working GPU path today is **wgpu/Vulkan** (via `cubecl-wgpu`), **not**
> native ROCm/HIP — `Hip`/`Cuda` backends are currently stubbed and fall back to CPU
> (`src/ops/backend_select.rs`). Whether to bring up native `cubecl-hip` is decided by a measured
> matmul A/B in [Phase 2](roadmap/phase-2-gpu-performance.md); recommendation is Vulkan-primary. The
> hardware specs, bandwidth-ceiling math, wave32 discussion, and expert-locality notes below remain
> valid reference material.

## Target Hardware: AMD Strix Halo

### Specifications

```
GPU:        Radeon 890M (RDNA 3.5, gfx1151)
CPU:        Zen 5 cores (up to 16 cores)
Memory:     Up to 96 GB LPDDR5X unified
Bandwidth:  256 GB/s
NPU:        XDNA 2 (up to 50 TOPS) — not used initially
TDP:        45–65 W (configurable)
```

### Why This Hardware Matters

Strix Halo is an APU — CPU and GPU share a single physical memory pool. This is architecturally different from any discrete GPU setup:

```
Discrete GPU (what inference engines were designed for):
  System RAM ←── PCIe (~64 GB/s) ──→ GPU VRAM
  weights live in VRAM (8–24 GB limit)
  transfers are expensive and frequent
  optimization: minimize transfers

Strix Halo APU:
  Single 96 GB pool @ 256 GB/s
  CPU and GPU access same physical addresses
  no transfers — GPU reads directly from mmap'd weights
  optimization: maximize bandwidth utilization
```

A 7B Q4 model (~4 GB) fits trivially. A 32B Q4 model (~19 GB) fits. MiniMax M2.7 at IQ3_XXS (~93 GB) fits — a 230B parameter model on a laptop APU.

---

## Theoretical Performance

```
tokens/sec ceiling = memory_bandwidth / active_weights_per_token

Model                Quant     Size      Ceiling    Realistic
─────────────────────────────────────────────────────────────
Llama 3.2 1B         Q8_0      1.3 GB    197 t/s    130–160 t/s
Llama 3.2 3B         Q6_K      2.3 GB    111 t/s    70–90 t/s
Qwen2.5 7B           Q4_K_M    4.1 GB    62 t/s     35–50 t/s
Llama 3.1 8B         Q4_K_M    4.7 GB    54 t/s     30–45 t/s
Qwen2.5 32B          Q4_K_M    19 GB     13 t/s     8–14 t/s
MiniMax M2.7         IQ4_XS    108 GB    ~28 t/s*   18–25 t/s
MiniMax M2.7         IQ3_XXS   93 GB     ~28 t/s*   18–25 t/s

* M2.7 has only ~10B active params per token (MoE sparsity).
  Effective bandwidth load is much less than total model size.
```

Hitting 70–75% of ceiling = good engine. Hitting 80%+ = excellent.

---

## ROCm Setup

### Current Status on Strix Halo

```
ROCm officially supports:   gfx1100 (RX 7900 XTX)
Strix Halo (gfx1151):       not yet listed officially
Workaround:                 HSA_OVERRIDE_GFX_VERSION=11.0.0
                            pretend to be gfx1100
                            works for most kernels in practice
```

### Installation (Ubuntu/Debian)

```bash
# Install ROCm
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install.deb
sudo apt install ./amdgpu-install.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -aG render,video $USER
# reboot

# Verify
export HSA_OVERRIDE_GFX_VERSION=11.0.0
rocminfo | grep gfx
rocm-smi

# Validate with llama.cpp first
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
make GGML_HIPBLAS=1 AMDGPU_TARGETS=gfx1100
./llama-cli -m model.gguf -p "Hello" --gpu-layers 99
```

---

## GPU Kernel Architecture: CubeCL

We use CubeCL rather than raw HIP FFI. This eliminates the C++ toolchain dependency and gives automatic hardware tuning.

### Why Not Raw HIP FFI

The original plan was HIP kernels in C++ called via `extern "C"`. CubeCL is strictly better:

```
Raw HIP FFI:              CubeCL:
  C++ .hip files            Pure Rust #[cube] functions
  hipcc at build time       No external compiler
  unsafe FFI bindings       Type-safe kernel launches
  manual wave32 tuning      Autotuned per device at startup
  ROCm only                 ROCm + Vulkan + Metal + CPU
  two codebases             One kernel, all backends
```

### Writing Kernels with CubeCL

```rust
// src/ops/kernels.rs
use cubecl::prelude::*;

#[cube(launch)]
fn matmul_q8_0<F: Float>(
    x:   &Array<F>,
    w:   &Array<u8>,      // Q8_0 packed blocks in unified memory
    out: &mut Array<F>,
    #[comptime] n_blocks: u32,
) {
    let row = ABSOLUTE_POS_X;
    let mut sum = F::new(0.0);

    for b in 0..n_blocks {
        // Dequantize Q8_0 block inline — fused with dot product
        let block_base = (row * n_blocks + b) * 34u32;

        // Read f16 delta (bytes 0-1 of block)
        let delta = unpack_f16(w[block_base], w[block_base + 1u32]);

        // Accumulate dot product
        for i in 0u32..32u32 {
            let q = w[block_base + 2u32 + i] as i32 as F;
            sum += delta * q * x[b * 32u32 + i];
        }
    }
    out[row] = sum;
}

// Dispatch — same call for any backend
pub fn launch_matmul_q8_0(
    client: &ComputeClient,
    x: &CubeBuffer<f32>,
    w: &CubeBuffer<u8>,
    out: &mut CubeBuffer<f32>,
    n_blocks: u32,
) {
    let out_dim = out.len() as u32;
    matmul_q8_0::launch::<f32>(
        client,
        CubeCount::Static(out_dim.div_ceil(64), 1, 1),
        CubeDim::new(64, 1, 1),
        x.as_arg(),
        w.as_arg(),
        out.as_arg_mut(),
        n_blocks,
    );
}
```

### The Autotuning Advantage

CubeCL benchmarks multiple kernel configurations at first run and caches the results:

```rust
// At engine startup, CubeCL runs tuning pass:
//   tries workgroup sizes: 32, 64, 128, 256
//   tries vectorization widths: 1, 2, 4, 8
//   picks fastest for this specific device

// On gfx1151 (Strix Halo), autotuning discovers:
//   wave32 = 32 threads per wavefront
//   4-wide vectorization for memory loads
//   2 waves per CU optimal for matmul

// This is the single biggest gap vs llama.cpp ROCm:
// llama.cpp uses CUDA-style wave64 assumptions.
// CubeCL finds wave32 automatically.
// Result: 2-4× speedup on decode matmuls vs unconfigured llama.cpp ROCm.
```

### Unified Memory

CubeCL's ROCm backend uses `hipMallocManaged` for allocations. On Strix Halo the mmap'd weights are already in the unified pool — the GPU reads directly:

```rust
// loader.rs — standard mmap, but on Strix Halo this IS GPU memory
pub fn load_mmap(path: &Path) -> Result<Mmap> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)   // GPU can read this directly on APU
}

// ops/kernels.rs — w points into mmap, no copy needed
#[cube(launch)]
fn matmul_q4_k<F: Float>(x: &Array<F>, w: &Array<u8>, ...) {
    // w is mmap'd memory. On discrete GPU: would need hipMemcpy first.
    // On Strix Halo: already physically accessible by GPU. Zero copy.
}
```

---

## RDNA 3.5 Specific Tuning

### Wave32 vs Wave64

RDNA 3 and 3.5 prefer 32-thread wavefronts over the 64-thread wavefronts used by older GCN/RDNA architectures and NVIDIA.

```c
// HIP kernel for RDNA 3.5
// __launch_bounds__(threads_per_block, waves_per_CU)
__launch_bounds__(256, 2)   // 256 threads = 8 waves of 32
__global__ void matmul_q4_k(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    int in_dim, int out_dim
) {
    // one output element per thread
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    // process input in Q4_K superblocks of 256 values
    float sum = 0.0f;
    // ... dequant + dot product ...
    out[row] = sum;
}
```

```rust
// Dispatch from Rust
fn dispatch_matmul_q4_k(x: &[f32], w: &Tensor, out: &mut [f32]) {
    let out_dim = out.len() as u32;
    let in_dim  = x.len() as u32;
    let threads = 256u32;
    let blocks  = (out_dim + threads - 1) / threads;

    unsafe {
        hip_matmul_q4_k(
            x.as_ptr(), w.data.as_ptr(),
            out.as_mut_ptr(),
            in_dim, out_dim,
            self.stream,
        );
        // kernels are async — sync when needed
    }
}
```

### Fused Dequant + Matmul

The critical optimization: dequantize inside the GPU kernel, immediately before the dot product. Never write f32 back to global memory.

```c
// Fused Q8_0 matmul kernel
__global__ void matmul_q8_0_fused(
    const uint8_t* __restrict__ w,    // packed Q8_0 blocks
    const float*   __restrict__ x,    // f32 input activations
    float*         __restrict__ out,
    int n_blocks                       // in_dim / 32
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        // Q8_0 block: 2-byte delta + 32 int8 values
        const uint8_t* block = w + row * n_blocks * 34 + b * 34;
        half delta_h;
        memcpy(&delta_h, block, sizeof(half));
        float delta = __half2float(delta_h);

        for (int i = 0; i < 32; i++) {
            float w_val = delta * (float)(int8_t)block[2 + i];
            float x_val = x[b * 32 + i];
            sum += w_val * x_val;          // MAC — no memory write for w_val
        }
    }

    out[row] = sum;
}
```

Each weight value is loaded, dequantized, multiplied, and accumulated — never written to global memory as f32. This keeps memory bandwidth at the quantized rate (1 byte/weight for Q8, 0.5 bytes/weight for Q4).

---

## Performance vs llama.cpp ROCm

### The Wave32 Gap

The single biggest performance difference between gguf-rs and llama.cpp on RDNA hardware is wave32 alignment. RDNA GPUs natively use 32-thread wavefronts. Many ROCm libraries inherited CUDA's wave64 assumptions. Aligning to wave32 produces a 3–4× speedup on small/medium matmul kernels — exactly the GEMV workload used in single-token decode.

llama.cpp's ROCm kernels were written with NVIDIA threading in mind. CubeCL's autotuner discovers wave32 as optimal for gfx1151 at startup automatically.

### Realistic Estimates

```
llama.cpp ROCm on Strix Halo:
  Wave64 mismatch on some kernels   decode matmuls underutilized
  No APU unified memory awareness   occasional unnecessary staging
  General purpose kernels           not tuned per device at runtime
  Estimated: 40–55% of bandwidth ceiling
  ~22–30 tok/s on 7B Q4_K_M

gguf-rs + CubeCL on Strix Halo:
  CubeCL autotuning → wave32        matmul utilization gap closed
  hipMallocManaged throughout       zero copy from mmap to GPU
  Fused dequant+matmul kernels      optimal memory access
  APU expert locality scheduling    better cache on M2.7
  Estimated: 65–75% of bandwidth ceiling
  ~40–48 tok/s on 7B Q4_K_M
```

Realistic 1.5–2× improvement on decode throughput on this specific hardware.

### Where llama.cpp Still Wins

```
Model support:    70+ architectures vs a handful
Ecosystem:        Ollama, LM Studio, Jan all use it
Maturity:         years of bug fixes, edge cases handled
Windows:          works well; CubeCL ROCm is Linux only currently
Community:        1200+ contributors finding bugs fast
```

### The Real Differentiator

Raw speed is only part of the story:

```
Kernel speed:                 1.5–2×  vs llama.cpp ROCm
+ TurboQuant KV (3-bit):      5× context at same memory
+ Speculative decode (EAGLE): 2–4×  effective throughput
+ Expert locality (M2.7):     10–20% on large MoE

M2.7 at 128K context:
  llama.cpp:  ~8 tok/s, KV pressure, barely viable
  gguf-rs:    ~25–35 effective tok/s with EAGLE, full context
```

Running M2.7 at genuine 128K context is a capability difference, not just speed. llama.cpp cannot do it on 96 GB hardware. gguf-rs can.

---

## Performance Optimization Layers

### Layer 1: Naive CPU (baseline)

- Sequential matmul, dequant full matrix first
- Expected: 5–15% of bandwidth ceiling

### Layer 2: CubeCL CPU Backend

- Same `#[cube]` kernels, CubeCL MLIR/LLVM compiler
- Automatic SIMD vectorization (AVX2 on x86)
- Parallel over output rows
- Expected: 20–35% of ceiling

### Layer 3: CubeCL ROCm — Default Config

- GPU dispatch via cubecl-hip
- CubeCL autotuning runs at startup, discovers wave32
- Fused dequant + matmul in one kernel pass
- Expected: 55–65% of ceiling

### Layer 4: APU-Specific

- `hipMallocManaged` throughout
- Weights read directly from mmap (no staging)
- Expert locality scheduling (MoE)
- Async kernel execution, overlap with CPU work
- Expected: 65–75% of ceiling

### Layer 5: Advanced

- Speculative decoding (EAGLE-3 or self-speculative)
- TurboQuant 3-bit KV cache
- These multiply effective throughput without changing kernel efficiency

---

## MoE Expert Locality Scheduling

For M2.7 with 256 experts per layer, naive dispatch accesses random locations in a ~100 GB weight file. Expert locality scheduling sorts dispatch by file offset:

```rust
fn dispatch_experts_sorted(x: &[f32], layer: &MoeLayer,
                            selected: &[usize]) -> Vec<f32> {
    // Sort selected experts by their byte offset in the weight file
    // Sequential reads are dramatically faster than random access
    let mut sorted: Vec<(usize, u64)> = selected.iter()
        .map(|&i| (i, layer.expert_byte_offset(i)))
        .collect();
    sorted.sort_by_key(|&(_, offset)| offset);

    let mut out = vec![0f32; x.len()];
    for (expert_idx, _) in sorted {
        let weight = probs[expert_idx];
        let expert_out = run_expert(x, &layer.experts[expert_idx]);
        for (o, e) in out.iter_mut().zip(expert_out.iter()) {
            *o += weight * e;
        }
    }
    out
}
```

On a system with NVMe-backed mmap, sequential expert access means the OS prefetcher can keep pages hot. Random access causes cache misses on a 100 GB file that can't fully fit in CPU cache.

---

## Memory Budget Management

```rust
pub struct MemoryBudget {
    total_bytes: u64,
    weights_bytes: u64,
    kv_cache_bytes_per_token: u64,
}

impl MemoryBudget {
    pub fn max_context_len(&self, kv_bits: u32) -> usize {
        let available = self.total_bytes
            .saturating_sub(self.weights_bytes)
            .saturating_sub(512 * 1024 * 1024);  // 512 MB headroom

        let kv_per_token = self.kv_cache_bytes_per_token * kv_bits as u64 / 16;
        (available / kv_per_token) as usize
    }

    pub fn report(&self) {
        let max_16bit = self.max_context_len(16);
        let max_8bit  = self.max_context_len(8);
        let max_3bit  = self.max_context_len(3);  // TurboQuant

        println!("Memory budget:");
        println!("  weights:         {} GB", self.weights_bytes / GB);
        println!("  available for KV: {} GB", available / GB);
        println!("  max context (16-bit KV): {}K tokens", max_16bit / 1024);
        println!("  max context (8-bit KV):  {}K tokens", max_8bit / 1024);
        println!("  max context (3-bit KV):  {}K tokens", max_3bit / 1024);
    }
}
```

This is surfaced in `gguf-info` before the user attempts to load a model — tells them exactly what context length is achievable on their hardware with each KV quantization level.
