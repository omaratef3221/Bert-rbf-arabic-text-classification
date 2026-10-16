#!/usr/bin/env bash
# All API-LLM runs for one model: E1 zero-shot (primary prompt on the FULL test set, other prompts on the fixed subset),
# E5 few-shot k=1/2, E6b/E7 controls and the determinism repeat (all on the subset).
#   bash run_llm_api.sh <provider> <model-id> [workers] [subset-size]
#   e.g. bash run_llm_api.sh openrouter openai/gpt-4.1 8 1400
#   FULL_PROMPTS="en_v2 en_v1" bash run_llm_api.sh together deepseek-ai/DeepSeek-V4-Flash-0731 8 1400   # second full-set prompt = subset-vs-full check
# subset-size 0 = everything on the full test set. Keys come from .env here or /root/.api_keys.env on the GPU box.
# Every llm_eval.py run is resumable (per-article JSONL), so re-running this script only fills in what is missing.
set -uo pipefail
cd "$(dirname "$0")"
PROVIDER="$1"; MODEL="$2"; WORKERS="${3:-8}"; SUBSET="${4:-0}"
FULL_PROMPTS="${FULL_PROMPTS:-en_v2}"
[[ -f /root/.api_keys.env ]] && set -a && source /root/.api_keys.env && set +a
export WANDB_PROJECT="${WANDB_PROJECT:-arabic-news-revision}"
export SANAD_ROOT="${SANAD_ROOT:-/workspace/data/sanad/1}"
PY="${PY:-/venv/main/bin/python}"
SAFE=$(echo "$MODEL" | sed -E 's/[^A-Za-z0-9.-]+/_/g')
mkdir -p logs results
LOG="logs/llm_${PROVIDER}_${SAFE}.log"
SUB=(); [[ "$SUBSET" != "0" ]] && SUB=(--subset "$SUBSET")
EXTRA=(); [[ -n "${OR_PROVIDER:-}" ]] && EXTRA=(--or-provider "$OR_PROVIDER")   # openrouter upstream pinning, e.g. OR_PROVIDER=anthropic
run () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; $PY llm_eval.py --provider "$PROVIDER" --model "$MODEL" --workers "$WORKERS" ${EXTRA[@]+"${EXTRA[@]}"} "$@" >> "$LOG" 2>&1 || echo "[FAIL] $*" | tee -a "$LOG"; }

for p in $FULL_PROMPTS; do run --prompt "$p"; done                                                   # E1 primary (full test set)
for p in en_v1 en_v3 ar_v1 ar_v2; do [[ " $FULL_PROMPTS " == *" $p "* ]] || run --prompt "$p" ${SUB[@]+"${SUB[@]}"}; done   # E1 variants
run --prompt en_v2 --shots 1 ${SUB[@]+"${SUB[@]}"}                        # E5: k=1 (7 exemplars)
run --prompt en_v2 --shots 2 ${SUB[@]+"${SUB[@]}"}                        # E5: k=2 (14 exemplars)
run --prompt en_v2 --truncate-words 100 ${SUB[@]+"${SUB[@]}"}             # E6b: input-length control
run --prompt en_v2 --strip-dateline ${SUB[@]+"${SUB[@]}"}                 # E7: lexical-shortcut control
run --prompt en_v2 --tag-suffix rep2 ${SUB[@]+"${SUB[@]}"}                # determinism check: identical config, second pass
echo "[$(date -u +%FT%TZ)] ALL DONE $PROVIDER $MODEL subset=$SUBSET" | tee -a "$LOG"
