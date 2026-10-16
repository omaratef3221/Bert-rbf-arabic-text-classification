#!/usr/bin/env bash
# All GPU experiments for the revision (E2, E3, E6a, E7, E8, E4 and the open-weight zero-shot of E1), in order.
# Runs on the vast.ai A100. Every step is idempotent at the file level: finished runs are listed in
# results/supervised/runs.jsonl / results/*.summary.json, and a step is skipped when its marker file exists.
#
#   bash run_all_gpu.sh            # full sequence, logs to logs/<step>.log and W&B project $WANDB_PROJECT
#   STEP=lc_dense bash run_all_gpu.sh   # one step only
set -uo pipefail
cd "$(dirname "$0")"
export WANDB_PROJECT="${WANDB_PROJECT:-arabic-news-revision}"
export SANAD_ROOT="${SANAD_ROOT:-/workspace/data/sanad/1}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results/markers
# three environments on the vast.ai box: /venv/main (torch, supervised), /venv/vllm (vLLM server + harness), /venv/lora (Unsloth/PEFT)
PY_MAIN="${PY_MAIN:-/venv/main/bin/python}"; PY_VLLM="${PY_VLLM:-/venv/vllm/bin/python}"; PY_LORA="${PY_LORA:-/venv/lora/bin/python}"
PY="$PY_MAIN"

run_step () {  # name, command...
  local name="$1"; shift
  if [[ -n "${STEP:-}" && "$STEP" != "$name" ]]; then return 0; fi
  if [[ -f "results/markers/$name.done" ]]; then echo "[skip] $name"; return 0; fi
  echo "[start] $name  $(date -u +%FT%TZ)" | tee -a logs/run_all.log
  if "$@" > "logs/$name.log" 2>&1; then
    touch "results/markers/$name.done"; echo "[done ] $name  $(date -u +%FT%TZ)" | tee -a logs/run_all.log
  else
    echo "[FAIL ] $name  exit=$?  see logs/$name.log" | tee -a logs/run_all.log
  fi
}

# 0. integrity: the split file must hash to the value built on the laptop
run_step split_check $PY - <<'EOF'
import hashlib, json, sys
h = hashlib.sha1(open("splits/sanad_alkhaleej_split_v1.csv", "rb").read()).hexdigest()
s = json.load(open("splits/sanad_alkhaleej_split_v1_summary.json"))
print("split sha1", h, "expected", s["sha1_of_split_file"]); assert h == s["sha1_of_split_file"]
from data_split import load_split
df = load_split(); print(df.split.value_counts().to_dict()); print("text sha1 verified for", len(df), "docs")
EOF

# 1. smoke tests (tiny, catches environment problems before the long runs)
run_step smoke_dense    $PY train_supervised.py --head dense    --smoke 30 --n-per-class 10 all --seeds 0 --wandb-project ""
run_step smoke_svm      $PY train_supervised.py --head svm      --smoke 30 --wandb-project ""
run_step smoke_finetune $PY train_supervised.py --head finetune --smoke 8 --epochs 1 --max-len 64 --batch-size 8 --wandb-project ""
rm -rf results/supervised results/emb_cache   # drop smoke outputs so runs.jsonl only holds real runs

# 2. supervised heads: full data, both splits, paper config (E1-supervised, E8)
run_step dense_full          $PY train_supervised.py --head dense --n-per-class all --seeds 0 1 2 --max-len 150
run_step svm_full            $PY train_supervised.py --head svm   --n-per-class all --seeds 0 1 2 --max-len 150
run_step dense_full_paperlr  $PY train_supervised.py --head dense --n-per-class all --seeds 0 --max-len 150 --lr 5e-5 --epochs 10 --patience 100
run_step dense_legacy        $PY train_supervised.py --head dense --n-per-class all --seeds 0 1 2 --max-len 150 --split-col split_legacy
run_step svm_legacy          $PY train_supervised.py --head svm   --n-per-class all --seeds 0 1 2 --max-len 150 --split-col split_legacy

# 3. learning curves (E2) at 150 and 512 wordpieces (E6a)
run_step lc_dense_150  $PY train_supervised.py --head dense --n-per-class 10 50 100 500 1000 all --seeds 0 1 2 --max-len 150
run_step lc_svm_150    $PY train_supervised.py --head svm   --n-per-class 10 50 100 500 1000 all --seeds 0 1 2 --max-len 150
run_step lc_dense_512  $PY train_supervised.py --head dense --n-per-class 10 50 100 500 1000 all --seeds 0 1 2 --max-len 512
run_step svm_full_512  $PY train_supervised.py --head svm   --n-per-class all --seeds 0 1 2 --max-len 512

# 4. dateline intervention (E7)
run_step dense_dateline_test   $PY train_supervised.py --head dense --n-per-class all --seeds 0 1 2 --max-len 150 --strip-dateline test_only
run_step dense_dateline_both   $PY train_supervised.py --head dense --n-per-class all --seeds 0 1 2 --max-len 150 --strip-dateline train_and_test

# 5. full fine-tuning of CAMeLBERT (E3)
run_step ft_100   $PY train_supervised.py --head finetune --n-per-class 100  --seeds 0 1 2 --max-len 512 --epochs 5 --lr 2e-5 --patience 2 --batch-size 32
run_step ft_1000  $PY train_supervised.py --head finetune --n-per-class 1000 --seeds 0 1 2 --max-len 512 --epochs 3 --lr 2e-5 --patience 1 --batch-size 32
run_step ft_all   $PY train_supervised.py --head finetune --n-per-class all  --seeds 0     --max-len 512 --epochs 3 --lr 2e-5 --patience 1 --batch-size 32

# 6. open-weight model: zero-shot with the same harness (E1), then QLoRA (E4)
export PY_VLLM PY_LORA
run_step qwen_zeroshot bash -c '
  $PY_VLLM -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192 --dtype bfloat16 --gpu-memory-utilization 0.85 > logs/vllm.log 2>&1 &
  VPID=$!; for i in $(seq 1 120); do curl -s localhost:8000/v1/models >/dev/null && break; sleep 5; done
  for p in en_v1 en_v2 en_v3 ar_v1 ar_v2; do $PY_VLLM llm_eval.py --provider vllm --model Qwen/Qwen2.5-7B-Instruct --prompt $p --workers 16 --wandb-project $WANDB_PROJECT; done
  $PY_VLLM llm_eval.py --provider vllm --model Qwen/Qwen2.5-7B-Instruct --prompt en_v2 --shots 1 --workers 16 --wandb-project $WANDB_PROJECT
  $PY_VLLM llm_eval.py --provider vllm --model Qwen/Qwen2.5-7B-Instruct --prompt en_v2 --shots 3 --workers 16 --wandb-project $WANDB_PROJECT
  kill $VPID; sleep 15'
run_step lora_100   $PY_LORA lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --n-per-class 100  --epochs 3 --eval-after-train
run_step lora_1000  $PY_LORA lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --n-per-class 1000 --epochs 2 --eval-after-train
run_step lora_all   $PY_LORA lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --n-per-class all  --epochs 1 --eval-after-train

# 7. tables
run_step tables $PY analysis.py --results-dir results
echo "ALL DONE $(date -u +%FT%TZ)" | tee -a logs/run_all.log
