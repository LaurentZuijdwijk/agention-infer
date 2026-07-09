//! Reusable greedy-generation runner shared by binaries (`bench`) and tests
//! (`golden`, CPU↔GPU cross-check). Deliberately dtype-agnostic: it drives the
//! public `Model::forward` loop with argmax decoding so the same code path
//! validates correctness and measures throughput.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::loader::load;
use crate::model::{create_model_with_backend, KvCache};
use crate::ops::{create_backend, BackendPreference};
use crate::sampler::argmax;
use crate::tokenizer::Tokenizer;

/// Result of a greedy generation run.
pub struct RunResult {
    /// Tokenized prompt (as our tokenizer produced it).
    pub prompt_ids: Vec<u32>,
    /// Greedy-generated token ids (excludes the prompt). Stops early at EOS
    /// when `stop_at_eos` is set.
    pub token_ids: Vec<u32>,
    /// Decoded text of `token_ids`.
    pub text: String,
    /// Wall time to run the prompt through prefill (one forward per token).
    pub prefill: Duration,
    /// Wall time to generate `token_ids`.
    pub decode: Duration,
    /// Name of the backend that actually ran (e.g. "cpu", "wgpu").
    pub backend: String,
    /// Whether decoding stopped because EOS was hit.
    pub stopped_eos: bool,
}

impl RunResult {
    pub fn prefill_tok_s(&self) -> f64 {
        self.prompt_ids.len() as f64 / self.prefill.as_secs_f64().max(1e-9)
    }
    pub fn decode_tok_s(&self) -> f64 {
        self.token_ids.len() as f64 / self.decode.as_secs_f64().max(1e-9)
    }
}

/// Load `path`, run `prompt` through prefill, then greedily decode up to
/// `max_new` tokens. `stop_at_eos` halts at the model's EOS token (use `false`
/// for benchmarking a fixed token count).
pub fn greedy_run(
    path: &Path,
    backend_pref: BackendPreference,
    prompt: &str,
    max_new: usize,
    stop_at_eos: bool,
) -> Result<RunResult> {
    let (gguf, mmap) = load(path)?;
    let data = &mmap[gguf.data_offset as usize..];
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let backend = create_backend(backend_pref);
    let backend_name = backend.name().to_string();
    let mut model = create_model_with_backend(&gguf, data, Some(backend))?;
    let cfg = model.config().clone();

    let prompt_ids = tokenizer.encode(prompt);
    if prompt_ids.is_empty() {
        return Err(crate::error::GgufError::BackendError(
            "prompt encoded to zero tokens".into(),
        ));
    }

    let max_seq = (prompt_ids.len() + max_new).min(cfg.context_length as usize);
    let mut kv = KvCache::new(
        cfg.block_count as usize,
        cfg.head_count_kv as usize,
        cfg.head_dim as usize,
        max_seq,
    );

    // ── Prefill ──────────────────────────────────────────────────────────
    // One batched pass over the whole prompt (one weight read per layer),
    // rather than a `forward` per token.
    let t_prefill = Instant::now();
    let mut logits = model.forward_batch(&prompt_ids, 0, &mut kv)?;
    let mut pos = prompt_ids.len();
    let prefill = t_prefill.elapsed();

    // ── Greedy decode ────────────────────────────────────────────────────
    let mut token_ids = Vec::with_capacity(max_new);
    let mut bytes = Vec::new();
    let mut stopped_eos = false;
    let t_decode = Instant::now();
    while token_ids.len() < max_new && pos < max_seq {
        let next = argmax(&logits) as u32;
        if stop_at_eos && Some(next) == tokenizer.eos_token_id {
            stopped_eos = true;
            break;
        }
        token_ids.push(next);
        bytes.extend_from_slice(&tokenizer.token_bytes(next));
        logits = model.forward(next, pos, &mut kv)?;
        pos += 1;
    }
    let decode = t_decode.elapsed();

    Ok(RunResult {
        prompt_ids,
        token_ids,
        text: String::from_utf8_lossy(&bytes).into_owned(),
        prefill,
        decode,
        backend: backend_name,
        stopped_eos,
    })
}
