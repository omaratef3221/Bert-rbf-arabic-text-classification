# revision/ — code for the major revision of NLP-D-26-00114

Everything here is self-contained and does not modify the original `train.py` pipeline.
See `REVISION_PLAN.md` for the audit, the reviewer-point mapping and the full run plan.

| File | Purpose |
|---|---|
| `data_split.py` | Builds the single canonical split (`splits/sanad_alkhaleej_split_v1.csv`): sorted deterministic load, exact dedup, TF-IDF near-duplicate clusters, cluster-aware stratified 70/15/15 split, legacy split, learning-curve ranks, few-shot exemplar ranks. `load_split()` joins the text back from the Kaggle dump and verifies sha1s. |
| `train_supervised.py` | Frozen CAMeLBERT + dense / RBF-SVM heads, full fine-tuning, learning curve, early stopping, 150 vs 512 tokens, dateline intervention, legacy split; writes per-doc predictions and a metrics line with bootstrap CIs and timings. |
| `llm_eval.py` | Zero-/few-shot API and local LLM harness: deterministic decoding, pinned snapshots, raw-output logging, audited deterministic label mapping (strict / lenient), retries, resume, summaries. `--provider mock` runs offline. |
| `lora_finetune.py` | QLoRA adaptation of an open-weight LLM (default Qwen2.5-7B-Instruct) on the same train split; writes `budget.json` (params, tokens, GPU-hours, VRAM). |
| `analysis.py` | Main table, learning curve, prompt robustness, mapping audit, McNemar / paired bootstrap, accuracy-by-similarity bins. |
| `requirements.txt` | Dependencies. |

Quick checks that need no GPU or API key:

```bash
python data_split.py                 # ~2 min on a laptop; prints and saves the split summary
python llm_eval.py --selftest        # label mapper unit test
python llm_eval.py --provider mock --model mock --prompt ar_v1 --shots 1 --max-docs 70
python train_supervised.py --head dense --smoke 40 --device cpu   # tiny end-to-end run (downloads CAMeLBERT)
```

Results land in `results/` (`supervised/runs.jsonl`, `*.jsonl` + `*.summary.json` per LLM run,
`lora/<run>/budget.json`, `tables/*.csv`). API keys are read from `.env` (never commit it).
