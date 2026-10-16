#!/usr/bin/env bash
# One-time environment setup on the vast.ai A100 box. Idempotent; safe to re-run.
#   bash setup_gpu.sh            (expects /workspace/revision = this folder, /workspace/sanad_alkhaleej_v1.tar.gz = dataset)
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p /workspace/data/sanad /workspace/hf logs results
export HF_HOME=/workspace/hf

echo "== machine =="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /workspace | tail -1; nproc; free -g | head -2
echo "== python =="; python --version; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())" || true

echo "== data =="
if [[ ! -d /workspace/data/sanad/1/Culture ]]; then
  tar -xzf /workspace/sanad_alkhaleej_v1.tar.gz -C /workspace/data/sanad && echo "dataset extracted"
fi
echo "files: $(find /workspace/data/sanad/1 -name '*.txt' | wc -l)  (expected 45500)"
export SANAD_ROOT=/workspace/data/sanad/1

echo "== core packages =="
pip install -q --upgrade pip
pip install -q pandas numpy scikit-learn scipy transformers accelerate datasets kagglehub tqdm python-dotenv wandb openai anthropic google-genai httpx 2>&1 | tail -2
python -c "import transformers, sklearn, pandas, wandb; print('transformers', transformers.__version__, 'sklearn', sklearn.__version__, 'wandb', wandb.__version__)"

echo "== cuML (GPU SVM; optional, falls back to scikit-learn) =="
if ! python -c "import cuml" 2>/dev/null; then
  pip install -q --extra-index-url=https://pypi.nvidia.com "cuml-cu12" 2>&1 | tail -1 || echo "cuML not installed -> sklearn SVC fallback"
fi
python -c "import cuml; print('cuml', cuml.__version__)" 2>/dev/null || echo "cuml unavailable (sklearn fallback will be used)"

echo "== vLLM + PEFT/TRL/bitsandbytes (+ unsloth) =="
python -c "import vllm" 2>/dev/null || pip install -q vllm 2>&1 | tail -1
pip install -q peft trl bitsandbytes 2>&1 | tail -1
python -c "import unsloth" 2>/dev/null || pip install -q "unsloth" 2>&1 | tail -1 || echo "unsloth not installed -> peft fallback"
python - <<'EOF'
import importlib
for m in ["torch", "vllm", "peft", "trl", "bitsandbytes", "unsloth", "cuml"]:
    try:
        mod = importlib.import_module(m); print(f"{m:12s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"{m:12s} MISSING ({type(e).__name__})")
import torch; print("torch cuda", torch.version.cuda, "bf16", torch.cuda.is_bf16_supported())
EOF

echo "== W&B =="
if [[ -n "${WANDB_API_KEY:-}" ]]; then python -c "import wandb; wandb.login(key='$WANDB_API_KEY', relogin=True)" && echo "wandb login ok"; else echo "WANDB_API_KEY not set"; fi

echo "== model weights (pre-download so the runs do not wait on the network) =="
python - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ["CAMeL-Lab/bert-base-arabic-camelbert-msa", "Qwen/Qwen2.5-7B-Instruct"]:
    p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges*", "vocab*"])
    print("cached", repo, "->", p)
EOF

echo "== split integrity =="
python - <<'EOF'
import hashlib, json
h = hashlib.sha1(open("splits/sanad_alkhaleej_split_v1.csv", "rb").read()).hexdigest()
s = json.load(open("splits/sanad_alkhaleej_split_v1_summary.json"))
print("split sha1", h, "OK" if h == s["sha1_of_split_file"] else "MISMATCH")
from data_split import load_split
df = load_split(); print(df.split.value_counts().to_dict(), "| text sha1 verified for", len(df), "docs")
EOF
echo "setup finished $(date -u +%FT%TZ)"
