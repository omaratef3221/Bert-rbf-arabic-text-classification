#!/usr/bin/env bash
# Run from the Mac when every sequence on the GPU box has finished. Pulls everything off the box, verifies it,
# builds the tables, commits, and (only with DESTROY=1) destroys the vast.ai instance.
#   bash finalize_box.sh                 # sync + verify + tables + commit
#   DESTROY=1 bash finalize_box.sh       # same, then destroy the instance after the checks pass
set -uo pipefail
cd "$(dirname "$0")"
HOST="${HOST:-root@194.228.55.129}"; PORT="${PORT:-31447}"; INSTANCE="${INSTANCE:-56283733}"
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=30"
PY="${PY:-python}"

echo "== 1. state on the box =="
$SSH $HOST 'cd /workspace/revision; echo "markers: $(ls results/markers | wc -l)"; grep -c "ALL DONE" logs/llm_*.log | tr "\n" " "; echo; grep -h "\[FAIL" logs/run_all.log logs/llm_*.log 2>/dev/null; tmux ls 2>/dev/null' 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)"

echo "== 2. sync results, logs, wandb, LoRA adapters =="
mkdir -p results logs wandb_box
rsync -az -e "ssh -p $PORT -o BatchMode=yes" --exclude "emb_cache/" --exclude "lora/*/checkpoint*" "$HOST:/workspace/revision/results/" ./results/
rsync -az -e "ssh -p $PORT -o BatchMode=yes" "$HOST:/workspace/revision/logs/" ./logs/
rsync -az -e "ssh -p $PORT -o BatchMode=yes" "$HOST:/workspace/revision/wandb/" ./wandb_box/ 2>/dev/null || true

echo "== 3. verify =="
REMOTE=$($SSH $HOST 'cd /workspace/revision; echo "$(wc -l < results/supervised/runs.jsonl) $(ls results/*.summary.json | wc -l) $(ls results/supervised/*.pred.csv | wc -l) $(ls results/lora/*/budget.json 2>/dev/null | wc -l) $(ls results/lora/*/adapter/adapter_model.safetensors 2>/dev/null | wc -l)"' 2>/dev/null | tail -1)
LOCAL="$(wc -l < results/supervised/runs.jsonl | tr -d ' ') $(ls results/*.summary.json | wc -l | tr -d ' ') $(ls results/supervised/*.pred.csv | wc -l | tr -d ' ') $(ls results/lora/*/budget.json 2>/dev/null | wc -l | tr -d ' ') $(ls results/lora/*/adapter/adapter_model.safetensors 2>/dev/null | wc -l | tr -d ' ')"
echo "remote (runs.jsonl lines, LLM summaries, pred csvs, lora budgets, adapters): $REMOTE"
echo "local  (same):                                                           $LOCAL"
if [[ "$REMOTE" != "$LOCAL" ]]; then echo "MISMATCH - not destroying"; exit 1; fi

echo "== 4. tables =="
$PY analysis.py --results-dir results 2>&1 | grep -vE "Warning|warn" | tail -60

echo "== 5. commit =="
cd .. && git add revision/results revision/analysis.py revision/EXPERIMENT_LOG.md revision/finalize_box.sh 2>/dev/null; git -c user.name="omaratef3221" -c user.email="omaratef3221@gmail.com" commit -q -m "results: all GPU and LLM runs synced from the vast.ai box; tables

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push origin main 2>&1 | tail -1; cd revision

if [[ "${DESTROY:-0}" == "1" ]]; then
  echo "== 6. destroy instance $INSTANCE =="
  ~/miniconda/bin/vastai destroy instance "$INSTANCE" && sleep 10 && ~/miniconda/bin/vastai show instances --raw | grep -c "\"id\": $INSTANCE" | sed 's/^0$/instance gone/; s/^[1-9].*/STILL LISTED/'
fi
echo "finalize done $(date -u +%FT%TZ)"
