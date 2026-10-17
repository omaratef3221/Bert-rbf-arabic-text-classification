# Experiment log and handoff — NLP-D-26-00114 revision

This file is written so that anyone (including a Claude session in the browser that has only this
repository and the W&B project) can continue the revision: find every result, know how it was
produced, and write the paper from it. Read this first, then `REVISION_PLAN.md` (what and why),
`MODEL_SNAPSHOTS.md` (which LLMs and settings), `STATUS_AND_TODO.md` (open items).

Last updated: 16 Oct 2026 (see §8 for the dated status log).

---

## 1. Context in five lines

- Paper: *Arabic News Classification: Lightweight Supervised Models vs. Zero-Shot Large Language Models*, Natural Language Processing Journal, manuscript NLP-D-26-00114, major revision due **22 Nov 2026**. Authors: Omar Elgendy, Ali Bou Nassif, Ashraf Elnagar (University of Sharjah). Submitted PDF: `paper v1.pdf` in the repo root (not in git).
- Task: 7-way single-label topic classification of SANAD AlKhaleej articles (Culture, Finance, Medical, Politics, Religion, Sports, Tech; 6,500 each).
- Systems: frozen CAMeLBERT-MSA [CLS] embedding + dense head or RBF-SVM (paper); full CAMeLBERT fine-tuning (new); zero-/few-shot API LLMs (GPT-4.1, Claude Haiku 4.5, Gemini Flash, DeepSeek V4 Flash); open-weight Qwen2.5-7B-Instruct zero-shot and QLoRA (new).
- The submitted LLM numbers cannot be reused: different random test samples per model, non-deterministic decoding, and the Claude 75 % / Religion-recall-0 result was a parsing bug (`max_tokens=1`, mapper without a Religion branch). Everything is recomputed on one split.
- Code: GitHub `omaratef3221/Bert-rbf-arabic-text-classification`, folder `revision/`. W&B: https://wandb.ai/omaratef3221/arabic-news-revision.

## 2. Where everything is

| What | Where |
|---|---|
| Canonical split (all systems) | `revision/splits/sanad_alkhaleej_split_v1.csv` (+ `_summary.json`, sha1 `951575cb…`). Columns: `doc_id`, `category`, `split` (dedup, cluster-aware), `split_legacy` (paper-style), `dup_group`, `dup_group_size`, `max_cos_other`, `n_words`, `text_sha1`, `lc_rank_s0..2` (learning-curve / few-shot ranks). Text is re-read from the Kaggle dump `haithemhermessi/sanad-dataset` by `data_split.load_split()`. |
| Supervised results | `revision/results/supervised/runs.jsonl` (one JSON line per run: config, accuracy, macro-F1, 95 % bootstrap CIs, per-class P/R/F1, confusion matrix, timings, trainable params, epoch history) and `revision/results/supervised/<run>.pred.csv` (doc_id, gold, pred). |
| LLM results | `revision/results/<provider>__<model>__<prompt>__k<shots>[__truncN].jsonl` (one line per article: raw output, mapped label, mapping method, finish reason, tokens, served model id, latency, attempts, error) + `.meta.json` (exact prompt text, exemplar ids, decoding settings, date) + `.summary.json` (strict and lenient metrics, mapping audit, unmapped examples, confusion). |
| LoRA budget | `revision/results/lora/<run>/budget.json` (trainable params, tokens, GPU-hours, peak VRAM, LoRA config); the adapter itself is not committed. Its test predictions are in `results/vllm__lora__…jsonl`. |
| Tables for the paper | `revision/results/tables/*.csv` produced by `python analysis.py --results-dir results`: `main_table.csv`, `learning_curve.csv`, `prompt_robustness.csv`, `fewshot.csv`, `mapping_audit.csv`, `pairwise_mcnemar.csv`. |
| Logs of the GPU box | `revision/logs/<step>.log` and `revision/logs/run_all.log` (not committed; copied to OneDrive at the end). |
| W&B | project `arabic-news-revision`, run names identical to the result file names; supervised runs grouped by `<head>_len<L>_<split>`, LLM runs grouped by `<provider>_<model>`. |

Naming of supervised runs: `<head>__<model>__len<maxlen>__n<per-class>__s<seed>__<split-col>__dl-<dateline>[__trunc<N>]`,
e.g. `dense__CAMeL-Lab_bert-base-arabic-camelbert-msa__len150__nall__s0__split__dl-none`.

## 3. The split (E0), already final

45,500 articles; 23 exact duplicates removed; 183 near-duplicate clusters at TF-IDF cosine ≥ 0.8
covering 1,403 articles (largest: templated daily market reports in Finance, 376 / 259 / 188
articles). Clusters are allocated whole, largest first, to the most under-filled split, so no
near-duplicate crosses train/val/test. Resulting counts (train / val / test): Culture 4547/974/974,
Finance 4548/975/975, Medical 4550/975/975, Politics 4549/975/975, Religion 4541/973/973, Sports
4549/975/975, Tech 4549/975/975. Zero test articles have a cosine ≥ 0.8 neighbour outside the test
set (the paper-style split had 212). Leakage effect measured on the submitted models: dropping test
articles with cosine ≥ 0.5 to a non-test article moves dense 96.82 → 96.48 and SVM 97.23 → 96.90.

## 4. Experiments and the paper element each one feeds

| ID | Experiment | Script / step in `run_all_gpu.sh` | Paper element |
|---|---|---|---|
| E1 | Zero-shot LLMs, 4 APIs + Qwen2.5-7B, 5 prompts each, full test set, T = 0 | `llm_eval.py`; Qwen via `qwen_zeroshot` | main table; prompt-robustness table (mean ± std, min–max); mapping-audit table |
| E2 | Supervised learning curve, dense and SVM, n ∈ {10, 50, 100, 500, 1000, all} per class, 3 seeds, 150 and 512 wordpieces | `lc_dense_150`, `lc_svm_150`, `lc_dense_512` | learning-curve figure (where supervised overtakes zero-/few-shot) |
| E3 | Full fine-tuning of CAMeLBERT at n ∈ {100, 1000, all} | `ft_100`, `ft_1000`, `ft_all` | main table row + learning-curve line; settles the frozen-vs-fine-tuned question |
| E4 | QLoRA on Qwen2.5-7B-Instruct at n ∈ {100, 1000, all}, with budget | `lora_100`, `lora_1000`, `lora_all` | main table row "task-adapted LLM"; budget table (params, tokens, GPU-h) |
| E5 | Few-shot k = 1 and 2 per class, primary prompt `en_v2`, all LLMs | `llm_eval.py --shots` | few-shot rows; gain over zero-shot |
| E6 | Input-length control: BERT 150 vs 512 (from E2); best LLM on first 100 words | `lc_dense_512`, `svm_full_512`; `llm_eval.py --truncate-words 100` | replaces mechanistic claims |
| E7 | Dateline intervention: dense head with test-only and train+test dateline removal; best LLM test-only | `dense_dateline_test`, `dense_dateline_both`; `llm_eval.py` on stripped text (TODO flag) | lexical-shortcut analysis (Reviewer 2, point 7) |
| E8 | Paper-style split replication, dense and SVM | `dense_legacy`, `svm_legacy` | shows the effect of deduplication |
| E9 | Statistics: bootstrap CIs, McNemar, prompt spread | `analysis.py` | CIs on every number, p-values for headline pairs |
| — | Paper-config dense head (lr 5e-5, 10 epochs) | `dense_full_paperlr` | continuity with the submitted numbers |

Supervised runs on the new split use early stopping on validation loss (patience 5, max 50 epochs,
lr 1e-3, AdamW, weight decay 0.01, batch 32) for the dense head; SVM: RBF, C = 9, gamma = scale
(cuML if installed, else scikit-learn); full fine-tuning: lr 2e-5, linear schedule with 6 % warm-up,
AMP, early stopping on val loss, max_len 512.

## 5. LLM settings (see `MODEL_SNAPSHOTS.md` for the rationale)

| Role | Model id | Provider | Status 16 Oct 2026 |
|---|---|---|---|
| GPT | `gpt-4.1-2025-04-14` | OpenAI | waiting for a valid key |
| Claude | `claude-haiku-4-5-20251001` | Anthropic | verified (smoke tests passed) |
| Gemini | `gemini-3.5-flash` (fallback `gemini-3.8-flash`) | Google AI Studio | waiting for credits |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Flash-0731`, reasoning off | Together AI serverless | verified |
| Open-weight | `Qwen/Qwen2.5-7B-Instruct` (bf16, vLLM; QLoRA 4-bit r=16) | vast.ai A100 | pending |

Decoding for every API model: temperature 0, top-p 1, max 16 output tokens, seed 42 where
supported, thinking/reasoning disabled, 60 s timeout, retries only on 429/5xx/timeouts. Prompt
caching is always on (shared prefix = instruction + labels + examples; Anthropic explicit
`cache_control`, OpenAI `prompt_cache_key`, Gemini explicit context cache; cache reads logged per
request, see `MODEL_SNAPSHOTS.md` §4). OpenRouter is supported as an alternative route (§3b there). Prompts
`en_v1` (paper wording), `en_v2` (primary), `en_v3`, `ar_v1`, `ar_v2`; full text in `llm_eval.py`.
Few-shot exemplars: the first k per class by `lc_rank_s0` in the train split, truncated to 120 words,
shuffled with a fixed seed. Scoring: strict mapping (7 labels + Arabic names), unmapped / ambiguous /
empty outputs count as errors; lenient mapping (synonyms such as Economy → Finance) reported as a
sensitivity.

## 6. Infrastructure

| Machine | Use | Details |
|---|---|---|
| MacBook (OneDrive copy of the repo) | split building, API LLM runs, statistics, writing | Python 3.12 venv; API keys in `revision/.env` (git-ignored); W&B key in `~/.netrc` |
| vast.ai instance `ssh -p 31447 root@194.228.55.129` (container `1a06c5e7698f`) | all GPU work (E2, E3, E4, E6a, E7, E8, Qwen zero-shot) | NVIDIA A100-PCIE-40GB, driver 595.104, CUDA 13.2, 64 CPUs, 503 GB RAM, 152 GB disk (not a persistent volume: results are rsynced to the Mac every 10 min and pushed to git). Envs: `/venv/main` (torch 2.14.1+cu132, transformers 5.20.1, scikit-learn 1.9.1, pandas 3.0.6, wandb 0.30) for the supervised runs; `/venv/vllm` (vLLM) for Qwen zero-shot; `/venv/lora` (Unsloth, TRL, PEFT, bitsandbytes) for QLoRA. Code `/workspace/revision`, data `/workspace/data/sanad/1` (45,500 files, tarball sha1 `ae7dd22b…`), HF cache `/workspace/hf` (CAMeLBERT + Qwen2.5-7B-Instruct pre-downloaded). A first instance (`38.64.63.43`) was abandoned: its outbound network dropped all HTTP responses. |

The GPU sequence is `revision/run_all_gpu.sh` (idempotent: each step writes `results/markers/<step>.done`
and is skipped on re-run; `STEP=<name> bash run_all_gpu.sh` runs one step). It is launched under
`tmux` session `rev` so it survives SSH disconnects: `tmux attach -t rev` to watch, `tail -f logs/run_all.log` for the step list.

## 7. How to turn results into the paper

1. Run `python analysis.py --results-dir results --pairs "<dense pred.csv>,<best LLM jsonl>" ...` → `results/tables/`.
2. Main table = `main_table.csv` rows: BERT-dense (mean ± std over seeds, len 150), BERT-svm, BERT-finetune, Qwen zero-shot, Qwen-LoRA, four API zero-shot (mean ± std over the 5 prompts) and few-shot rows; CIs from `acc_lo/acc_hi`.
3. Learning-curve figure from `learning_curve.csv` (x = labelled articles per class, log scale; lines dense-150, dense-512, svm, finetune, LoRA points at 100/1000/all; horizontal bands for zero-shot and few-shot LLM means).
4. Prompt robustness from `prompt_robustness.csv`; mapping audit from `mapping_audit.csv` (report exact / normalized / Arabic / contains / unmapped shares per model, with the raw examples).
5. Significance: `pairwise_mcnemar.csv` (McNemar exact p and paired bootstrap CI of the accuracy difference) for dense vs best LLM, LoRA vs dense, few-shot vs zero-shot.
6. Qualitative tables: sample disagreements from the `.pred.csv` / `.jsonl` files of the final runs only (join on `doc_id`).
7. Text changes: `REVISION_PLAN.md` §5; response letter: `REVISION_PLAN.md` §1 (A1–A11) with the new numbers.

## 7b. Final numbers (16–17 Oct 2026 runs; new deduplicated split; accuracy, mean over seeds)

Supervised (test n = 6,822; 3 seeds unless noted; `results/supervised/runs.jsonl`):

| System | 10 / class | 50 | 100 | 500 | 1000 | all (31,833) |
|---|---|---|---|---|---|---|
| Frozen CAMeLBERT + dense head, 150 tokens | 82.5 | 90.9 | 92.9 | 95.0 | 95.4 | **96.4** (paper config lr 5e-5: 96.8) |
| Frozen CAMeLBERT + RBF-SVM, 150 tokens | 81.5 | 90.4 | 92.4 | 95.3 | 96.0 | **97.1** |
| Frozen CAMeLBERT + dense head, 512 tokens | 88.4 | 94.2 | 95.1 | 96.3 | 96.7 | **97.6** |
| Frozen CAMeLBERT + RBF-SVM, 512 tokens | | | | | | **98.1** |
| Full fine-tuning of CAMeLBERT, 512 tokens (109 M params) | | | 96.6 | | 98.4 | **98.9** (1 seed) |

Paper-style random split: dense 96.8, SVM 97.2 (vs 96.4 / 97.1 deduplicated: leakage worth ≈ 0.1–0.4 points).
Dateline intervention (dense, 150): 96.4 → 96.5 with datelines removed from test only, 96.4 removed from train and test: the dateline is not a shortcut.
Religion recall: dense 0.95, SVM 0.97, fine-tuned 0.99.

LLMs, zero-shot primary prompt `en_v2`, full test set (n = 6,822; `results/*__en_v2__k0.summary.json`):

| Model | Accuracy | Macro-F1 | Religion recall | Unmapped |
|---|---|---|---|---|
| GPT-4.1 (`openai/gpt-4.1` via OpenRouter → OpenAI) | **92.8** | 92.7 | 0.76 | 0.00 % |
| Claude Haiku 4.5 (via OpenRouter → Anthropic) | **92.0** | 91.9 | 0.74 | 0.04 % |
| Gemini 3.8 Flash (via OpenRouter → Google AI Studio flex, minimal reasoning) | **91.5** | 91.4 | 0.72 | 0.06 % |
| DeepSeek V4 Flash 0731 (Together) | **91.5** | 91.4 | 0.71 | 0.01 % |
| Qwen2.5-7B-Instruct, bf16, vLLM | **85.2** (89.2 with `en_v1`) | — | — | 0.6 % |
| Qwen2.5-7B-Instruct + QLoRA, 100 per class (0.29 GPU-h) | **95.7** | 95.7 | 0.89 | 0.13 % |
| Qwen2.5-7B-Instruct + QLoRA, 1000 per class (1.48 GPU-h) | **98.4** | 98.4 | 0.97 | 0.01 % |
| Qwen2.5-7B-Instruct + QLoRA, all 31,833 (3.27 GPU-h, 20.2 M tokens) | **98.5** | 98.5 | 0.98 | 0.00 % |

(QLoRA rows: adapters evaluated through vLLM with a 16 k context, `results/vllm__lora_*__en_v2__k0.*`; Qwen zero-shot few-shot on the full set: k = 1 90.5 %, k = 2 87.8 % with 1.0 % unmapped.)

Subset-700 sweeps (the same 700 articles for every model; `results/tables/prompt_robustness.csv`, `fewshot.csv`, `controls.csv`):

| Model | 5 prompts, zero-shot: mean ± std (min–max) | Primary prompt | k = 1 | k = 2 | Repeat run | No dateline | First 100 words |
|---|---|---|---|---|---|---|---|
| GPT-4.1 | 95.2 ± 0.5 (94.7–95.9) | 95.4 | 94.4 | 94.4 | 95.4 | 95.4 | 94.4 |
| Claude Haiku 4.5 | 92.9 ± 0.7 (92.1–93.9) | 93.9 | 92.9 | 93.4 | 93.9 | 93.7 | 93.3 |
| Gemini 3.8 Flash | 93.1 ± 0.3 (92.7–93.4) | 93.4 | 94.9 | 95.3 | 93.4 | 93.6 | 92.4 |
| DeepSeek V4 Flash | 92.2 ± 1.5 (89.9–93.7) | 93.3 | 94.3 | 94.4 | 93.4 | 92.9 | 92.6 |
| Qwen2.5-7B zero-shot | 87.4 ± 2.5 (84.9–90.3) | 86.6 | 91.6 | 88.9 | | | |

Reading: prompt wording moves the strong API models by 0.3–0.7 points std (DeepSeek 1.5, Qwen 2.5); the repeat run at temperature 0 reproduces the primary run to within 0.1 point for every model; removing the dateline changes accuracy by at most 0.4 points (so it is not a shortcut for LLMs either); truncating the article to its first 100 words costs 0.6–1.0 points; few-shot examples help the weaker models (Qwen +5.0, Gemini +1.9, DeepSeek +1.1 at k = 2) and do not help GPT-4.1 or Claude (−1.0 / −0.4). The subset is slightly easier than the full test set for every model (e.g. DeepSeek 93.3 vs 91.5), which is why the full-set primary numbers, not the subset numbers, go in the main table.

## 8. Status log

| Date (UTC) | Event |
|---|---|
| 2026-10-12 | Audit of the submitted pipeline; leakage analysis; canonical split built; scripts written and smoke-tested on CPU. |
| 2026-10-14 | Live-API check of model availability; Claude Haiku 4.5 and DeepSeek V4 Flash verified; OpenAI keys invalid; Gemini account without credits. |
| 2026-10-16 | W&B project `arabic-news-revision` created; opt-in W&B logging added to all scripts and tested offline; `run_all_gpu.sh` written; dataset packaged for the GPU box (`sanad_alkhaleej_v1.tar.gz`, 54 MB, sha1 `ae7dd22b…`). Code pushed to GitHub. First vast.ai host (`38.64.63.43`) had no working egress and was replaced. |
| 2026-10-16 18:49 UTC | GPU box `194.228.55.129:31447` set up (see §6); split checksum verified on the box (train 31,833 / val 6,822 / test 6,822 after dedup); GPU smoke tests of dense, SVM and fine-tune heads passed; **`run_all_gpu.sh` launched in tmux session `rev`** (steps in the order of §4: full-data heads → legacy split → learning curves → dateline → full fine-tuning → Qwen zero-shot → QLoRA → tables). Results stream to W&B and are rsynced to `revision/results/` on the Mac every 10 minutes. First numbers (new split, mean over seeds): dense 96.4 %, SVM 97.1 %, paper-config dense 96.8 %; legacy split 96.8 / 97.2 %; learning curve dense 82.5 / 92.9 / 95.4 % at 10 / 100 / 1000 per class. |
| 2026-10-16 19:05 UTC | **DeepSeek V4 Flash (Together) launched** (tmux `llm`, `run_llm_api.sh`, 8 workers). Full-set primary prompt `en_v2`: accuracy 91.5 %, macro-F1 91.4 %, Religion recall 0.71, 1 unmapped output in 6,822. Design changed at 19:30 to save cost: primary prompt (and `en_v1` as the subset-vs-full check) on the full set, all other runs on the fixed 1,400 subset; few-shot is k = 1 and k = 2 (author decision). |
| 2026-10-16 19:41 UTC | **GPT-4.1 and Claude Haiku 4.5 launched via OpenRouter** (tmux `gpt`, `claude`; upstream pinned to OpenAI / Anthropic, fallbacks off; subset 700 for sweeps; 6 workers each). Cache pass-through verified (Claude k = 2: 71 % cached tokens). OpenRouter credit $55, key limit $50. |
| 2026-10-16 21:05 UTC | **All API-LLM sequences finished** (DeepSeek 20:06, Claude 20:11, GPT-4.1 20:18, Gemini 20:58 UTC): 40 API runs + 4 Qwen zero-shot runs so far, no failed runs; single-article API errors in 4 runs (1 of 6,822 each, scored as errors). OpenRouter usage $50.13 (key limit $50 reached at the very end; the remaining $4.87 credit is unusable until the key limit is raised). Full-set primary zero-shot: GPT-4.1 92.8 %, Claude Haiku 4.5 92.0 %, Gemini 3.8 Flash 91.5 %, DeepSeek V4 Flash 91.5 %, Qwen2.5-7B 85.2 % (en_v2; 89.2 % with en_v1). Subset-700 sweeps complete for all four API models (see `results/tables/`). |
| 2026-10-17 05:27 UTC | **All experiments complete; vast.ai instance 56283733 destroyed** after the final sync was verified against the box (83 supervised runs, 50 LLM runs, 3 QLoRA budgets and adapters; results, logs, wandb run folders and adapters are on OneDrive under `revision/`, everything except adapters/logs/wandb also in git). Final QLoRA accuracies (vLLM evaluation): 95.7 / 98.4 / 98.5 % at 100 / 1000 / all per class; Qwen k = 2: 87.8 %. GPU rental ≈ 11 h ≈ $8.5; API spend ≈ $53 (OpenRouter $50.13 + Together ≈ $3). Next: tables → figures → manuscript text and response letter (`REVISION_PLAN.md` §5). |
| 2026-10-17 03:52 UTC | **GPU sequence finished** (`run_all_gpu.sh`: 28 of 29 steps done; the `tables` step failed only because the box had a stale `analysis.py`, tables are built on the Mac). Timings: full CAMeLBERT fine-tuning 100 / 1000 / all per class in 7 / 12 / 13 min; Qwen2.5-7B zero-shot (5 prompts + few-shot) 1 h 53 min via vLLM; QLoRA (r = 16, all linear layers, 40.4 M trainable params = 0.82 %, 4-bit, Unsloth) 0.29 / 1.48 / 3.27 GPU-hours for 100 / 1000 / all per class (1.3 / 8.9 / 20.2 M training tokens, 9.3 GB peak VRAM). The Qwen few-shot step used the pre-decision k = 3 (4.6 % context overflows at 8 k tokens): moved to `results/superseded/`; k = 2 re-run with a 16 k context at 04:02. The direct Hugging Face evaluation of the three adapters right-truncated long prompts (2.2 % unmapped outputs = article continuations, mostly Religion): moved to `results/superseded/`; adapters re-evaluated through vLLM (16 k context, no truncation) as `vllm__lora_n100 / lora_n1000 / lora_all`. |
| 2026-10-16 19:45 UTC | **Gemini 3.8 Flash launched via OpenRouter** (tmux `gemini`; provider `google-ai-studio/flex`, half price; reasoning cannot be disabled on this endpoint, so `reasoning.effort = minimal` with a 256-token output budget, thinking tokens logged per request). Watcher on the Mac polls the four LLM logs and the OpenRouter spend every 10 minutes. |
| 2026-10-17 09:20 UTC | **Manuscript revised.** `TEX_paper/elsarticle/manuscript_revised.tex` rewritten against the response letter; tables from `make_tables.py` (`paper_numbers.json`), figures from `make_figures.py`; clean PDF (33 pp.) and latexdiff tracked-changes PDF built with LuaLaTeX (babel `bidi=basic` + Amiri for the Arabic excerpts); deliverables + Overleaf zip in `TEX_paper/revised/`. |
| 2026-10-17 11:30 UTC | **Manuscript and letter corrected after review of the draft**: few-shot comparison now made on the same 700 articles (512-token head at 500/class 96.7 vs best few-shot 95.3; 150-token head needs the full split); subset is 1.4–2.6 points easier for the LLMs, stated plainly; original dense-head recipe (lr 5e-5) added to Table 5 (96.8, above the 96.4 headline; recipe fixed a priori); en_v1 = code wording, manuscript had paraphrased (added to Corrections); QLoRA share corrected to 0.53 % of 7.62 B; residual leakage at cosine ≥ 0.5 computed (`leakage_check.py`: 517 test articles, all systems ≈ 99–100 % on them, gap unchanged); letter cross-references aligned with the compiled numbering; GitHub repository renamed to `Bert-rbf-arabic-text-classification`. |

## 9. Open items

See `STATUS_AND_TODO.md` §3 (author side: OpenAI key, Gemini credits, GPU box reachable, manuscript source) and §4 (my side).
