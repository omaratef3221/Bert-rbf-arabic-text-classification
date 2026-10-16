#!/usr/bin/env bash
# All API-LLM runs for one model (E1 zero-shot x5 prompts, E5 few-shot k=1/3, E6b/E7 controls, determinism repeat).
#   bash run_llm_api.sh <provider> <model-id> [workers]
#   e.g. bash run_llm_api.sh together deepseek-ai/DeepSeek-V4-Flash-0731 8
# Keys are read from .env in this folder or from the environment (sourced from /root/.api_keys.env on the GPU box).
# Every llm_eval.py run is resumable (per-article JSONL), so re-running this script only fills in what is missing.
set -uo pipefail
cd "$(dirname "$0")"
PROVIDER="$1"; MODEL="$2"; WORKERS="${3:-8}"
[[ -f /root/.api_keys.env ]] && set -a && source /root/.api_keys.env && set +a
export WANDB_PROJECT="${WANDB_PROJECT:-arabic-news-revision}"
export SANAD_ROOT="${SANAD_ROOT:-/workspace/data/sanad/1}"
PY="${PY:-/venv/main/bin/python}"
SAFE=$(echo "$MODEL" | sed -E 's/[^A-Za-z0-9.-]+/_/g')
mkdir -p logs results
LOG="logs/llm_${PROVIDER}_${SAFE}.log"
run () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; $PY llm_eval.py --provider "$PROVIDER" --model "$MODEL" --workers "$WORKERS" "$@" >> "$LOG" 2>&1 || echo "[FAIL] $*" | tee -a "$LOG"; }

for p in en_v2 en_v1 en_v3 ar_v1 ar_v2; do run --prompt $p; done          # E1: zero-shot, five prompts (primary first)
run --prompt en_v2 --shots 1                                                # E5: few-shot k=1 (7 exemplars)
run --prompt en_v2 --shots 3                                                # E5: few-shot k=3 (21 exemplars)
run --prompt en_v2 --truncate-words 100                                     # E6b: input-length control
run --prompt en_v2 --strip-dateline                                         # E7: lexical-shortcut control
run --prompt en_v2 --tag-suffix rep2                                        # determinism check: identical config, second pass
echo "[$(date -u +%FT%TZ)] ALL DONE $PROVIDER $MODEL" | tee -a "$LOG"
