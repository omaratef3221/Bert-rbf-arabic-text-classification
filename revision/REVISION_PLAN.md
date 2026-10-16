# Revision plan for NLP-D-26-00114

*Arabic News Classification: Lightweight Supervised Models vs. Zero-Shot Large Language Models*
Natural Language Processing Journal, major revision, resubmission due **22 Nov 2026**.

This document records (1) what an audit of the submitted experiments found, (2) how each reviewer
point maps to an action, (3) the new experiment set, (4) decisions the author team must make, and
(5) the run plan for the vast.ai A100 machine. All new code lives in `revision/`.

---

## 1. Audit of the submitted experiments

The supervised pipeline is `Fake_news_RBF/train.py`; the LLM evaluation lives outside the repo in
`../Final Experiments Notebooks/` (four per-provider notebooks, May 2025, and
`llm_classification_inference.py`, Dec 2025). Checking the code and the saved prediction files
against the manuscript gives the following. Items marked **[disclose]** change reported numbers
and should be mentioned in the response letter; items marked **[fix]** only need a corrected text.

| # | Finding | Evidence | Consequence |
|---|---------|----------|-------------|
| A1 | **Claude 3.5 Haiku's 75 % / Religion recall 0 is a pipeline artifact.** The May 2025 notebook called the API with `max_tokens=1`; Claude often answered in Arabic, so outputs were cut to one token (`ال`, `الس`, `الت`). The label-mapping cell had **no `religion` branch** and assigned a **random label among the other six** to anything unmapped. | `Claude LLMs_Classification_for_BERT_RBF.ipynb`, cells 8 and 15 | Religion could never be predicted. In the Dec 2025 re-run with `max_tokens=10`, Claude scores **92.5–93.1 %**, Religion recall 0.76–0.77, in line with the other LLMs. The whole "cultural limitation / safety filter" narrative (Sections 3.3, 4.4.3, 5) must be removed. **[disclose]** |
| A2 | **Every system was scored on a different test sample.** All notebooks and the dense-head branch of `train.py` call `train_test_split` without `random_state`; `os.listdir` order is also unsorted. | dense vs SVM test sets overlap on 3,444 / 6,825 docs; the two 1,000-doc LLM samples overlap on 72 docs | Head-to-head comparisons were not paired. Everything must be re-run on one canonical split. **[disclose]** |
| A3 | **GPT model mismatch.** The Dec 2025 script used `gpt-4o-mini` (temperature 0); the paper reports GPT-4.1. The May 2025 GPT-4.1 run used the Responses API with default temperature on a 1,000-doc sample, while Table 4 lists support 975 per class. | `llm_classification_inference.py`, GPT notebook cell 7–8, `FINAL RESULTS.xlsx` | Re-run with pinned snapshots. **[disclose]** |
| A4 | Non-deterministic decoding: Gemini and Claude at default temperature (1.0), DeepSeek at `temperature=1`. | notebooks, Dec script | Re-run with temperature 0 (Reviewer 1 2(ii)). |
| A5 | Table 4 (Claude 0.75, Religion 0) and Section 4.3.4 ("1,000-sample test set", "32 of 135 Religion samples") come from **different runs** (May full-test vs Dec 1,000-doc sample). | paper p. 17 vs p. 10 | Internal inconsistency; both replaced by the new runs. |
| A6 | Table 6 examples ("Finance mislabeled as Sports") were sampled from `Aug 2025 analysis/final_df_original_model.csv`, a dense run at **92.9 %** with 83 Finance→Sports errors, not the 96.8 % model of Table 4/5. The saved SVM file of that run predicts "Finance" for every row. | `result_sample_basic_model.csv`, `final_df_*.csv` | Qualitative examples must come from the final runs. |
| A7 | Paper says "early stopping based on validation loss"; code trains a fixed 10 epochs. Abstract and contributions say "linear SVM"; the model is an RBF SVM. `degree=6` is listed for an RBF kernel, where it is ignored. | `train.py`, paper pp. 6, 8 | **[fix]** text; new code implements early stopping. |
| A8 | BERT input is truncated to **150 wordpieces** (`batching.py`), never stated; LLMs received the full article (mean 369 words). | `batching.py` | Input-length confound; state it and add a controlled comparison (E6). |
| A9 | Reported training times (4.38 s SVM, 10.54 s dense) exclude the encoder forward pass, which both heads share; SVM "inference time" also includes pickling the model. | `train.py` | Report encoder time separately and head time; fix the timer. |
| A10 | Terminology: "multi-label" in title keywords and text; task is 7-way single-label. | paper | **[fix]** (Reviewer 1 minor). |
| A11 | Mechanistic claims ("fine-tuning adjusts the entire model", "learns Arabic-news-specific embeddings", "knowledge graph", "surface-level heuristics") are unsupported; the encoder is frozen. | Sections 4.1, 4.4.5, 5 | Rewrite as "frozen pretrained representation + supervised decision boundary" (Reviewer 2 points 1 and 7). |

### Local analyses already done (CPU, no GPU needed)

*Leakage / near-duplicates (Reviewer 2, point 8).* Corpus: 45,500 articles; 23 exact duplicates;
183 near-duplicate clusters at TF-IDF cosine ≥ 0.8 covering 1,403 articles. The large clusters
are templated daily market reports in Finance (series of 376, 259 and 188 near-identical articles).
For the December test splits:

| Test docs with a neighbour outside the test set at cosine ≥ t | share | Dense acc on them | SVM acc on them |
|---|---|---|---|
| t = 0.5 | 10.6 % | 0.995–1.000 | 1.000 |
| t = 0.8 | 3.3 % | 1.000 | 1.000 |
| Accuracy after dropping all docs with t ≥ 0.5 | | 96.82 → **96.48** | 97.23 → **96.90** |

So leakage exists, inflates the headline numbers by roughly 0.3 points, and is now removed by
construction in the new split. Accuracy on the 57 % of test articles with no similar article
anywhere (cosine < 0.2) is 95.4 % (dense) and 96.4 % (SVM).

*December 2025 re-scoring of the four LLMs (1,000-doc samples, `max_tokens=10`).*

| Model (as actually called) | acc, dense sample | acc, SVM sample | Religion recall |
|---|---|---|---|
| gpt-4o-mini, T=0 | 0.914 | 0.899 | 0.74 / 0.69 |
| gemini-2.0-flash, default T | 0.913 | 0.940 | 0.74 / 0.78 |
| claude-3-5-haiku-20241022, default T | 0.925 | 0.931 | 0.76 / 0.77 |
| DeepSeek-V3 (Together), T=1 | 0.925 | 0.931 | 0.79 / 0.80 |
| BERT head on the same sample | 0.962 | 0.972 | 0.96 / 0.97 |

Unmapped outputs: 1 per 1,000 for DeepSeek (`الإخبارية`, `Environment`), 0 for the others.
Every LLM has the same weak spot (Religion recall 0.69–0.80); the gap to the supervised heads
is 3–7 points, not 22.

*Dateline shortcut.* 12 % of articles start with a city dateline (`دبي: «الخليج»`). It is strongly
class-skewed (Medical 40 %, Finance 15 %, Tech 12 %, Sports 0.06 %, Religion 1 %), which makes it a
cheap lexical cue for the controlled intervention Reviewer 2 asks for (E7).

*API cost.* Articles average 674 GPT tokens (o200k). A zero-shot pass over the full 6,825-doc test
set is ≈ 5.0 M input tokens: about $10 for GPT-4.1, $4 Claude 3.5 Haiku, $1 Gemini 2.0 Flash,
$1–6 DeepSeek V3. **All LLM experiments can therefore use the full test set**, which also makes
paired significance tests against the supervised models possible.

---

## 2. Reviewer point → action

| Reviewer point | Action | Where |
|---|---|---|
| R1 minor: "multi-label" | Replace with "multi-class (single-label)" everywhere incl. keywords. | text |
| R1-1 / R2-1: frozen vs fine-tuned | State per system what is trained; rewrite Discussion. Add a genuinely fine-tuned CAMeLBERT so the claim can be tested. | text + **E3** |
| R1-2(i) / R2-2: task-adapted LLM | QLoRA on an open-weight LLM on the same train split, with budget (GPU-h, params, tokens). | **E4** |
| R2-2: few-shot LLMs; learning curve | k = 1 and 2 exemplars per class; supervised learning curve 10/50/100/500/1000/all per class, 3 seeds. | **E5**, **E2** |
| R1-2(ii) / R2-4: decoding + snapshots | temperature 0, top-p 1, max_tokens 16, seed, pinned snapshot ids, served model id, date, retry policy logged per request. | `llm_eval.py`, **E1** |
| R1-2(iii, iv) / R2-5: mapping + non-compliance | Deterministic audited mapper (strict and lenient tiers), rates of exact / normalized / Arabic / contains / ambiguous / unmapped per run; unmapped counted as errors; examples in the paper. | `llm_eval.py`, table `mapping_audit.csv` |
| R1-2(v) / R2-3: prompt robustness | 5 prompts (3 English, 2 Arabic), mean ± std and min–max per model; 95 % bootstrap CIs; McNemar vs supervised. | **E1**, `analysis.py` |
| R2-6: verify Claude Religion | Explained by A1; re-run shows no anomaly. Remove the causal discussion; disclose the bug. | text + **E1** |
| R2-7: mechanistic claims | Remove; replace by the two controlled interventions we can support (input length, dateline). | text + **E6**, **E7** |
| R2-8: leakage | Deduplicated, cluster-aware split for all systems; report legacy-split numbers and accuracy-by-similarity bins. | `data_split.py`, **E0**, **E8** |
| R1-1(iii): protocol | Full tables of trainable layers, optimizer, LR schedule, epochs, early stopping, params updated, embedding extraction. | `train_supervised.py` writes all of it to `runs.jsonl` |

---

## 3. Experiments

All experiments use `revision/splits/sanad_alkhaleej_split_v1.csv` (built, sha1 in the
summary json). Train / val / test ≈ 31.8k / 6.8k / 6.8k, 23 exact duplicates removed, no
near-duplicate cluster (cosine ≥ 0.8) straddles a split. A `split_legacy` column reproduces the
paper's random stratified split for comparison.

| ID | Experiment | Spec | Compute / cost |
|---|---|---|---|
| **E0** | Canonical split and leakage report | Done locally. Report cluster statistics and the accuracy-by-similarity table. | 2 min CPU |
| **E1** | Zero-shot LLMs, deterministic, 5 prompts, full test set | GPT-4.1, Claude Haiku, Gemini Flash, DeepSeek V3 (pinned snapshots, see decision D1) + the open-weight model from E4 before adaptation (vLLM). T=0, top-p=1, max_tokens=16. | 4 APIs × 5 prompts ≈ **$80**; open-weight 20 min A100 |
| **E2** | Supervised learning curve | dense and SVM heads, frozen CAMeLBERT, n ∈ {10, 50, 100, 500, 1000, all} per class × 3 seeds, max_len 150 (paper) and 512. Early stopping on val loss. | < 1 h A100 (one encoder pass per length, cached) |
| **E3** | Full fine-tuning of CAMeLBERT | all 109 M params, lr 2e-5, 3 epochs, early stopping, max_len 512, at n ∈ {100, 1000, all}; reports params, GPU-h. | ≈ 1.5 h A100 |
| **E4** | QLoRA LLM baseline | Qwen2.5-7B-Instruct (D2), 4-bit, LoRA r=16 on all linear layers, 1 epoch on the full train split (≈ 19 M tokens) and at n = 100 and 1000 per class; same prompt as zero-shot; budget json. Evaluate via vLLM with `llm_eval.py`. | ≈ 2–3 h A100 for full data; 10 min per small run |
| **E5** | Few-shot LLMs | k = 1 and k = 2 exemplars per class from the train split (fixed seed, 120-word exemplars), primary prompt; full test set for DeepSeek, fixed stratified subset (`splits/llm_subset_*.csv`) for the OpenRouter models, all API models. | k=1 ≈ 4×, k=3 ≈ 8× the zero-shot tokens: ≈ **$150** for the four APIs (D3 offers a cheaper option) |
| **E6** | Input-length control | (a) dense head at 150 vs 512 wordpieces (from E2); (b) best API LLM on the first 100 words only vs full article (one prompt). | ≈ $5 |
| **E7** | Lexical intervention (dateline) | dense head and best LLM evaluated on test articles with the dateline removed; dense head also retrained without datelines. Reports accuracy drop per class, especially Medical. | minutes + ≈ $10 |
| **E8** | Legacy-split replication | dense + SVM on `split_legacy` (paper setting) to show the effect of deduplication. | minutes |
| **E9** | Statistics | 95 % bootstrap CIs for every number; McNemar + paired bootstrap for the key pairs (best LLM vs dense, LoRA vs dense, few-shot vs zero-shot); prompt mean ± std. | `analysis.py` |

Expected headline after the revision (based on the December numbers): supervised heads ≈ 96.5–97 %,
zero-shot LLMs ≈ 90–94 % with a 1–3 point prompt spread, few-shot likely +1–2, LoRA-adapted
7B model likely ≥ 96 %. The paper's conclusion then becomes the defensible one Reviewer 2 states:
*with in-domain labels, a frozen Arabic encoder plus a small head matches or beats much larger
models at a fraction of the cost; without labels, zero-shot LLMs trail by a few points, mostly on
Religion/Culture; the learning curve shows where the two regimes cross.*

---

## 4. Decisions needed from the author team

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Which API snapshots? | **Decided 14 Oct 2026:** originals where still served, otherwise the closest same-tier successor, with served ids and dates logged. Verified against the live APIs: `gpt-4.1-2025-04-14` (original; key pending), `claude-haiku-4-5-20251001` (3.5 Haiku retired; smoke test passed), `gemini-3.5-flash` or `gemini-3.8-flash` (2.0 retired, 2.5 closed to new users; credits pending), DeepSeek via Together serverless `DeepSeek-V4-Flash-0731` or the official DeepSeek API (V3 and V3.1 no longer serverless). Full rationale in `MODEL_SNAPSHOTS.md`. |
| D2 | Open-weight model for E4 | **Decided:** one model, `Qwen/Qwen2.5-7B-Instruct`, used zero-shot and after QLoRA (same weights before/after adaptation). No extra API models; the reviewers asked for an adapted open-weight model, not for more models. |
| D3 | Few-shot setting for E5 | **Decided (16 Oct 2026):** k = 1 and k = 2 exemplars per class. Primary-prompt zero-shot on the full test set for every model; prompt variants, few-shot, controls and the repeat run on a fixed stratified subset (1,400 or 700 articles, `splits/llm_subset_*.csv`) for GPT-4.1, Claude and Gemini via OpenRouter; DeepSeek on the full set for everything (cheap) as the subset-vs-full check. |
| D4 | Keep Claude Haiku in the paper at all? | Yes, re-run properly (Haiku 4.5), and state in the response letter that the original 3.5 Haiku number was a parsing bug. |
| D5 | Hyperparameters of the dense head | Paper: lr 5e-5, 10 epochs. New default: lr 1e-3, early stopping (needed for the small-n learning curve). Report both for the full-data row. |
| D6 | API keys and spend cap | Anthropic and Together keys work. OpenAI keys return 401 (new key needed); Gemini account has no credits. See `STATUS_AND_TODO.md` P1–P2. |

---

## 5. Manuscript changes (independent of the new numbers)

1. Title/keywords/text: "multi-class (single-label)"; "RBF SVM", drop `degree`.
2. Section 3.2: one table saying exactly what is trained for each system (frozen encoder + head; full
   fine-tune; LoRA; none), trainable parameter counts, input truncation.
3. Section 3.3: all five prompts verbatim (English and Arabic), few-shot format, mapping procedure with
   examples, decoding parameters, snapshot ids and dates, retry policy, non-compliance rates.
4. Section 3.4: early stopping as implemented; timing that includes the encoder pass.
5. Section 3.1: deduplication and cluster-aware split; leakage table.
6. Sections 4.4 and 5: delete the Claude-specific cultural/alignment explanation and the
   "knowledge graph", "surface-level heuristics", "fine-tuning adjusts the entire model" claims;
   replace with the E6/E7 interventions and the learning-curve reading.
7. Add: learning curve figure, prompt-robustness table (mean ± std), few-shot and LoRA rows in the
   main table, CIs on every number, McNemar p-values for the headline comparisons.
8. Response letter: candid note on A1–A3 (bugs found during revision, all numbers recomputed).

---

## 6. Run plan on the vast.ai machine

```bash
# 0. setup (A100 40GB image with CUDA 12 / torch 2.x)
git clone <repo> && cd Fake_news_RBF/revision && pip install -r requirements.txt
export SANAD_ROOT=$(python -c "import kagglehub;print(kagglehub.dataset_download('haithemhermessi/sanad-dataset'))")
python data_split.py --root $SANAD_ROOT          # must reproduce sha1 951575cb... (see summary json)
cp "../../Final Experiments Notebooks/.env" .    # API keys (never commit)

# 1. supervised (E2, E3, E6a, E7, E8)
python train_supervised.py --head dense --n-per-class 10 50 100 500 1000 all --seeds 0 1 2 --max-len 150
python train_supervised.py --head dense --n-per-class 10 50 100 500 1000 all --seeds 0 1 2 --max-len 512
python train_supervised.py --head svm   --n-per-class 10 50 100 500 1000 all --seeds 0 1 2 --max-len 150
python train_supervised.py --head dense --n-per-class all --seeds 0 --lr 5e-5 --epochs 10 --patience 100   # paper config
python train_supervised.py --head finetune --n-per-class 100 1000 all --seeds 0 --max-len 512 --epochs 3 --lr 2e-5 --patience 1
python train_supervised.py --head dense --n-per-class all --strip-dateline test_only
python train_supervised.py --head dense --n-per-class all --strip-dateline train_and_test
python train_supervised.py --head dense --n-per-class all --split-col split_legacy
python train_supervised.py --head svm   --n-per-class all --split-col split_legacy

# 2. zero-shot APIs (E1): 4 models x 5 prompts; each run is resumable. Ids per MODEL_SNAPSHOTS.md (14 Oct 2026).
#    These need no GPU: they can run from the laptop; the paper's LLM runs were API calls too.
for p in en_v1 en_v2 en_v3 ar_v1 ar_v2; do
  python llm_eval.py --provider openai    --model gpt-4.1-2025-04-14             --prompt $p
  python llm_eval.py --provider anthropic --model claude-haiku-4-5-20251001      --prompt $p
  python llm_eval.py --provider gemini    --model gemini-3.5-flash               --prompt $p   # or gemini-3.8-flash
  python llm_eval.py --provider together  --model deepseek-ai/DeepSeek-V4-Flash-0731 --prompt $p   # or: --provider deepseek --model deepseek-chat
done
# 3. few-shot (E5) and controls (E6b, E7)
for m in "openai gpt-4.1-2025-04-14" "anthropic claude-haiku-4-5-20251001" "gemini gemini-3.5-flash" "together deepseek-ai/DeepSeek-V4-Flash-0731"; do
  set -- $m; python llm_eval.py --provider $1 --model $2 --prompt en_v2 --shots 1
             python llm_eval.py --provider $1 --model $2 --prompt en_v2 --shots 2
done
python llm_eval.py --provider openai --model gpt-4.1-2025-04-14 --prompt en_v2 --truncate-words 100
# 4. open-weight zero-shot + QLoRA (E1, E4)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 &   # then:
python llm_eval.py --provider vllm --model Qwen/Qwen2.5-7B-Instruct --prompt en_v2
python lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --n-per-class all  --epochs 1 --eval-after-train
python lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --n-per-class 100  --epochs 3 --eval-after-train
python lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --n-per-class 1000 --epochs 2 --eval-after-train
# 5. tables and tests (E9)
python analysis.py --results-dir results --pairs "results/supervised/dense__...pred.csv,results/openai__gpt-4.1-2025-04-14__en_v2__k0.jsonl"
```

Wall-clock estimate: one working day of A100 time plus API runs that can proceed in parallel from
the same machine. Smoke-test every script with `--max-docs 50` / `--smoke` before the full runs.
