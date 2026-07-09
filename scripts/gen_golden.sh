#!/usr/bin/env bash
# Generate golden fixtures from llama.cpp's `llama-simple` (raw greedy
# completion, no chat template, no sampler config — pure argmax). Our golden
# test (tests/golden.rs) replays the same prompt through our engine's greedy
# decode and asserts the generated text matches token-for-token.
#
# Usage:
#   LLAMA_SIMPLE=~/Projects/llama-cpp-turboquant/build/bin/llama-simple \
#     scripts/gen_golden.sh
#
# Re-run whenever the model set or the reference llama.cpp version changes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${GGUF_MODELS_DIR:-$ROOT/models}"
OUT_DIR="$ROOT/tests/fixtures/golden"
LLAMA_SIMPLE="${LLAMA_SIMPLE:-$HOME/Projects/llama-cpp-turboquant/build/bin/llama-simple}"
LLAMA_REPO="${LLAMA_REPO:-$HOME/Projects/llama-cpp-turboquant}"
N_PREDICT="${N_PREDICT:-24}"

if [[ ! -x "$LLAMA_SIMPLE" ]]; then
  echo "llama-simple not found at $LLAMA_SIMPLE (set LLAMA_SIMPLE)" >&2
  exit 1
fi
BUILD="$(git -C "$LLAMA_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
mkdir -p "$OUT_DIR"

# Models to cover (basename without .gguf). Add rows as the set grows.
MODELS=(
  "Qwen3.5-9B-Q4_K_M"
  "Qwen3-1.7B-Q8_0"
  "LFM2.5-1.2B-Instruct-Q4_K_M"
  "LFM2.5-1.2B-Instruct-Q6_K"
)

# prompt-id -> prompt text (single line). Chosen to be *confident* — greedy
# decoding is deterministic across engines only where the top logit has a clear
# margin. Open-ended prompts (e.g. "The capital of France is") diverge at
# near-tie branches and are unsuitable as a strict cross-engine gate; see
# docs/roadmap/phase-0-correctness-harness.md.
declare -A PROMPTS=(
  ["count"]="Here is a list of numbers: 1, 2, 3,"
  ["days"]="The days of the week are Monday, Tuesday, Wednesday,"
  ["alpha"]="a b c d e f g"
  ["evens"]="2, 4, 6, 8, 10,"
)

for model in "${MODELS[@]}"; do
  gguf="$MODELS_DIR/$model.gguf"
  if [[ ! -f "$gguf" ]]; then
    echo "skip (missing): $gguf" >&2
    continue
  fi
  for pid in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$pid]}"
    echo "generating $model / $pid ..." >&2
    raw="$(timeout 180 "$LLAMA_SIMPLE" -m "$gguf" -n "$N_PREDICT" "$prompt" </dev/null 2>/dev/null || true)"
    # Completion = everything after the FIRST occurrence of the prompt echo.
    # (llama-simple prints <BOS_piece> + prompt + completion.)
    completion="${raw#*"$prompt"}"
    if [[ "$completion" == "$raw" ]]; then
      echo "  WARN: prompt echo not found in output for $model/$pid; storing raw" >&2
    fi
    out="$OUT_DIR/${model}__${pid}.txt"
    {
      echo "# model: $model.gguf"
      echo "# prompt_id: $pid"
      echo "# prompt: $prompt"
      echo "# n_predict: $N_PREDICT"
      echo "# tool: llama-simple"
      echo "# llama_build: $BUILD"
      echo ""
      printf '%s' "$completion"
    } >"$out"
    echo "  wrote $out" >&2
  done
done
echo "done." >&2
