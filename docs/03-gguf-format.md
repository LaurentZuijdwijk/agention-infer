# GGUF Format & Parsing

## What GGUF Is

GGUF (GPT-Generated Unified Format) is a self-contained binary format for LLM weights, invented by llama.cpp in August 2023. It replaced a succession of earlier formats (GGML, GGMF, GGJT) by solving their core problems: external file dependencies, hardcoded hyperparameters, and non-extensibility.

A GGUF file contains everything needed to run a model:
- Model architecture parameters (layer count, head count, context length, etc.)
- The complete tokenizer vocabulary and merge rules
- All weight tensors, quantized
- Chat template (Jinja2 string)
- Licensing information

No external config files. No Python code. One file.

---

## Binary Layout

```
┌──────────────────────────────────────┐  offset 0
│  Magic: "GGUF"  (4 bytes, 0x46554747)│
│  Version: u32   (2 or 3)             │
│  tensor_count: u64                   │
│  metadata_kv_count: u64              │
├──────────────────────────────────────┤
│  METADATA SECTION                    │
│  ┌────────────────────────────────┐  │
│  │ key: string (u64 len + bytes)  │  │
│  │ value_type: u32                │  │
│  │ value: <typed>                 │  │
│  └────────────────────────────────┘  │
│  × metadata_kv_count                 │
├──────────────────────────────────────┤
│  TENSOR INFO SECTION                 │
│  ┌────────────────────────────────┐  │
│  │ name: string                   │  │
│  │ n_dims: u32                    │  │
│  │ dims: [u64; n_dims]            │  │
│  │ ggml_type: u32                 │  │
│  │ byte_offset: u64               │  │
│  └────────────────────────────────┘  │
│  × tensor_count                      │
├──────────────────────────────────────┤
│  <padding to 32-byte alignment>      │
├──────────────────────────────────────┤
│  TENSOR DATA                         │
│  (raw bytes, back-to-back)           │
│  (each tensor 32-byte aligned)       │
└──────────────────────────────────────┘
```

All integers are little-endian. Strings are `u64` length followed by UTF-8 bytes — no null terminator. Tensor byte offsets are relative to the start of the tensor data section (after alignment padding), not to the start of the file.

---

## Metadata Value Types

```
0   UINT8
1   INT8
2   UINT16
3   INT16
4   UINT32
5   INT32
6   FLOAT32
7   BOOL        (1 byte, 0 or 1)
8   STRING      (u64 length + UTF-8 bytes)
9   ARRAY       (u32 elem_type + u64 count + [values...])
10  UINT64
11  INT64
12  FLOAT64
```

Arrays can nest. An `ARRAY` of `STRING` is common for vocabulary tokens. An `ARRAY` of `FLOAT32` is common for BPE scores.

Note: GGUF is inconsistent about integer widths. Some fields that logically should be `UINT64` are stored as `UINT32` depending on the model exporter. The typed accessor layer must handle both:

```rust
pub fn get_u64(&self, key: &str) -> Result<u64> {
    match self.metadata.get(key) {
        Some(MetadataValue::U32(v)) => Ok(*v as u64),  // common case
        Some(MetadataValue::U64(v)) => Ok(*v),
        _ => Err(...)
    }
}
```

---

## Key Metadata Fields

### General

```
general.architecture        String    "llama", "qwen2", "minimax-m2", ...
general.name                String    human-readable model name
general.license             String    license identifier
general.file_type           u32       quantization type of majority of tensors
```

### Architecture-Specific (prefix = general.architecture value)

```
{arch}.context_length               u32/u64   max sequence length
{arch}.embedding_length             u32/u64   d_model
{arch}.block_count                  u32/u64   number of transformer layers
{arch}.attention.head_count         u32/u64   n_heads (Q heads)
{arch}.attention.head_count_kv      u32/u64   n_kv_heads (GQA)
{arch}.feed_forward_length          u32/u64   FFN intermediate dim (dense)
{arch}.rope.freq_base               f32       RoPE theta (default 10000)
{arch}.rope.scaling.type            String    "yarn", "linear", etc.
{arch}.rope.scaling.factor          f32

{arch}.attention.layer_norm_rms_epsilon  f32  RMSNorm epsilon
```

### MoE-Specific

```
{arch}.expert_count                 u32/u64   total experts per layer
{arch}.expert_used_count            u32/u64   top-K active per token
{arch}.expert_feed_forward_length   u32/u64   FFN dim per expert
{arch}.expert_shared_count          u32/u64   always-active shared experts
```

### Tokenizer

```
tokenizer.ggml.model                String    "llama", "gpt2", "rwkv"
tokenizer.ggml.tokens               String[]  vocabulary — one string per token
tokenizer.ggml.scores               f32[]     BPE merge scores
tokenizer.ggml.token_type           i32[]     0=normal 1=unknown 2=control 3=user
tokenizer.ggml.bos_token_id         u32       begin-of-sequence token ID
tokenizer.ggml.eos_token_id         u32       end-of-sequence token ID
tokenizer.ggml.padding_token_id     u32
tokenizer.chat_template             String    Jinja2 template string
```

---

## Tensor Naming Conventions

### Dense Models (Llama family)

```
token_embd.weight               [vocab_size, d_model]
output_norm.weight              [d_model]
output.weight                   [vocab_size, d_model]   (may be absent → tie with token_embd)

blk.{i}.attn_norm.weight        [d_model]
blk.{i}.attn_q.weight           [n_heads * head_dim, d_model]
blk.{i}.attn_k.weight           [n_kv_heads * head_dim, d_model]
blk.{i}.attn_v.weight           [n_kv_heads * head_dim, d_model]
blk.{i}.attn_output.weight      [d_model, n_heads * head_dim]
blk.{i}.ffn_norm.weight         [d_model]
blk.{i}.ffn_gate.weight         [ffn_dim, d_model]
blk.{i}.ffn_up.weight           [ffn_dim, d_model]
blk.{i}.ffn_down.weight         [d_model, ffn_dim]
```

### MoE Models

```
blk.{i}.ffn_gate_inp.weight     [n_experts, d_model]         router
blk.{i}.ffn_gate_exps.weight    [n_experts, ffn_dim, d_model] all gate experts packed
blk.{i}.ffn_up_exps.weight      [n_experts, ffn_dim, d_model]
blk.{i}.ffn_down_exps.weight    [n_experts, d_model, ffn_dim]

# Shared experts (DeepSeek style):
blk.{i}.ffn_gate_shexp.weight   [ffn_dim, d_model]
blk.{i}.ffn_up_shexp.weight     [ffn_dim, d_model]
blk.{i}.ffn_down_shexp.weight   [d_model, ffn_dim]
```

Note: llama.cpp packs all expert tensors for a layer into a single 3D tensor (`_exps` suffix) rather than separate tensors per expert index. The first dimension is `n_experts`. Verify against actual model files.

### Weight Tying

If `output.weight` is absent from the tensor list, the embedding table (`token_embd.weight`) is used for both embedding lookup and the final LM head projection (transposed). This is called weight tying.

```rust
let lm_head = model.tensors.get("output.weight")
    .unwrap_or_else(|| model.tensors.get("token_embd.weight").unwrap());
```

---

## Multi-File GGUF

Models larger than ~50 GB are split across multiple files:

```
MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf   ← header + tensor info + some data
MiniMax-M2.7-UD-IQ4_XS-00002-of-00004.gguf   ← tensor data only
MiniMax-M2.7-UD-IQ4_XS-00003-of-00004.gguf   ← tensor data only
MiniMax-M2.7-UD-IQ4_XS-00004-of-00004.gguf   ← tensor data only
```

Only the first file contains the full GGUF header and metadata. Subsequent files contain only tensor data, with byte offsets continuing from where the first file ended.

The loader must:
1. Parse header from file 1 only
2. Build a combined virtual address space from all mmaps
3. Resolve tensor offsets against the correct physical file

```rust
pub struct MultiFileMmap {
    mmaps: Vec<(Mmap, u64)>,    // (mmap, cumulative_start_offset)
}

impl MultiFileMmap {
    pub fn slice(&self, offset: u64, len: usize) -> &[u8] {
        // find which file contains this offset, return slice
    }
}
```

---

## Memory Mapping Strategy

On Strix Halo, mmap is not just a convenience — it is the correct architecture:

```rust
pub fn load_mmap(path: &Path) -> Result<Mmap> {
    let file = File::open(path)?;
    // SAFETY: we trust the file won't be modified while mapped.
    // On Strix Halo, this memory is in the unified pool.
    // The GPU can read directly from these addresses.
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}
```

Pages are loaded on demand by the OS. First access to any tensor page causes a page fault — the OS loads that page from NVMe. Subsequent accesses (same and future tokens) hit the page cache. For repeated inference, the entire model eventually lives in the page cache without ever being explicitly "loaded."

The GPU (ROCm on Strix Halo) can access the same physical pages the CPU mapped. No `hipMemcpy` needed. The GPU reads weights directly from the unified memory pool.

---

## Supported Architectures

| Architecture string | Examples |
|---|---|
| `llama` | Llama 2, Llama 3, Mistral, Phi-3 |
| `qwen2` | Qwen2, Qwen2.5, Qwen2.5-Coder |
| `qwen2moe` | Qwen2.5 MoE |
| `minimax-m2` | MiniMax M2.7 |
| `deepseek2` | DeepSeek V2, V3 |
| `gemma3` | Gemma 3 |
| `gemma3n` | Gemma 3n (MatFormer + PLE) |
| `mixtral` | Mixtral 8×7B, 8×22B |

Architecture detection:

```rust
let arch = gguf.get_string("general.architecture")?;
let model: Box<dyn Model> = match arch {
    "llama"      => Box::new(LlamaModel::from_gguf(&gguf, backend)?),
    "qwen2"      => Box::new(LlamaModel::from_gguf(&gguf, backend)?),  // compatible
    "qwen2moe"   => Box::new(MoeModel::from_gguf(&gguf, backend)?),
    "minimax-m2" => Box::new(MinimaxModel::from_gguf(&gguf, backend)?),
    "mixtral"    => Box::new(MoeModel::from_gguf(&gguf, backend)?),
    other        => return Err(GgufError::UnsupportedArchitecture(other.into())),
};
```
