# Phase 3 — Model + quant + tokenizer breadth (STUB)

**Status:** gated on Phase 2. Do not start until Phase 2 lands competitive numbers and we decide to
continue. This is a stub — flesh it into a full execution doc when the gate opens.

## Goal

Run the models that justify 128 GB — especially **MoE** — not "all of llama.cpp." Breadth is a means
to running the select target set, not a goal in itself.

## Rough task list

- **MoE forward pass** (the flagship gap): router → top-k → expert dispatch → weighted sum; shared
  experts (DeepSeek style); **expert-locality scheduling** (sort dispatch by byte offset,
  `docs/06-rocm-hardware.md:390`). Add `src/model/moe.rs` or a `Mixer::Moe` variant; dispatch in
  `create_model_with_backend` (`src/model/mod.rs:463`). Unlocks Mixtral, Qwen3-MoE/235B,
  GPT-OSS-120B, DeepSeek, MiniMax.
- **Quant coverage:** dequant + fused GPU kernels for **MXFP4** (GPT-OSS), **IQ4_XS / IQ3_XXS / IQ2**
  (IQ MoE + long-context), **Q3_K**, legacy **Q4_0/Q4_1/Q5_1**. Extend `src/quant/`,
  `GPU_DEQUANT_DTYPES` (`src/ops/mod.rs:24`), `WeightMap::dequant_tensor` (`src/model/mod.rs:287`).
  Port bit-tricks from llama.cpp's Vulkan shaders.
- **Tokenizers:** **SentencePiece/Unigram** (Llama-1/2, Mistral, Gemma) and **tiktoken**-style; lift
  the gpt2-only guard at `src/tokenizer.rs:53`. Round-trip tests vs llama.cpp.
- **Chat-template engine:** render GGUF `tokenizer.chat_template` (minja-style Jinja subset); replace
  the hardcoded ChatML in `src/bin/generate.rs:126`. Built-in fallbacks per family.
- **Dense architectures as needed:** Gemma (norm quirks, logit soft-cap), Phi, StableLM.

## Done-criteria (draft)

Golden parity on a Qwen3-MoE/Mixtral model + a GPT-OSS (MXFP4) model + an SPM model (Gemma/Mistral);
tokenizer round-trip tests pass. See `docs/08-moe-routing.md` and `docs/04-quantization.md` for
reference material.
