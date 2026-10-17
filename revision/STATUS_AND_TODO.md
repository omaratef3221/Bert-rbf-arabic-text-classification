# Status, strategy and open items — NLP-D-26-00114 revision

Last updated: 14 Oct 2026. Deadline for the revised manuscript: **22 Nov 2026**.
Companion documents: `REVISION_PLAN.md` (audit, reviewer-point mapping, experiment specs),
`MODEL_SNAPSHOTS.md` (which LLM snapshots are used and why).

---

## 1. Strategy in one page

**What the reviewers want.** A comparison that is paired, reproducible and not confounded:
the same test set for every system; LLMs run deterministically on pinned snapshots with their
raw outputs and parsing audited; a task-adapted LLM (LoRA) and few-shot LLMs next to the zero-shot
ones; a supervised learning curve showing how many labels the small models need; a leakage-free
split; and a Discussion that only claims what the experiments show.

**What we found when re-reading the code.** The submitted LLM numbers were produced on
different random test samples per model, with non-deterministic decoding, and the Claude result
(75 %, Religion recall 0) was a parsing bug (`max_tokens=1` plus a label mapper without a
"Religion" branch). A proper re-run in Dec 2025 already showed all four LLMs at 90–94 %. The
supervised numbers hold up (the leakage we measured is worth about 0.3 points).

**How we answer.** One canonical split (`revision/splits/sanad_alkhaleej_split_v1.csv`,
deduplicated, near-duplicate clusters never straddle train/test, 973–975 test articles per class)
used by everything; a single LLM harness with temperature 0, pinned ids, raw-output logging and a
deterministic mapper; the experiment set E0–E9 in `REVISION_PLAN.md`; bootstrap CIs and McNemar
tests on every headline comparison; and a candid note in the response letter about the bugs.

**Where things run.**

| Work | Where | Why |
|---|---|---|
| Split building, leakage analysis, all tables and statistics | laptop (done / CPU) | no GPU needed |
| Zero-shot and few-shot API LLMs (GPT, Claude, Gemini, DeepSeek) | laptop or any machine with internet; only API keys needed | the paper's LLM runs were API calls from the laptop too, never on vast.ai |
| Supervised heads, learning curve, full CAMeLBERT fine-tuning | vast.ai A100 | GPU (≈ 2–3 h in total) |
| Open-weight model zero-shot (vLLM) and QLoRA adaptation | vast.ai A100 | GPU (≈ 3–4 h) |

**Expected outcome.** Supervised heads ≈ 96.5–97 %; zero-shot LLMs ≈ 90–94 % with a 1–3 point
prompt spread and a shared weak spot on Religion/Culture; few-shot +1–2 points; a LoRA-adapted
7B model probably at or above the supervised heads. The paper's conclusion becomes: *with in-domain
labels a frozen Arabic encoder plus a small head matches or beats much larger models at a fraction
of the cost; without labels, zero-shot LLMs trail by a few points; the learning curve shows where
the regimes cross.*

---

## 2. Decisions taken (14 Oct 2026)

| Topic | Decision |
|---|---|
| LLM snapshots | Use the original ids where still served; otherwise the closest same-tier successor, with the date of evaluation and the served model id logged per request. Details and rationale in `MODEL_SNAPSHOTS.md`. |
| Open-weight model for the LoRA baseline (Reviewer 1, 2(i)) | One model, `Qwen/Qwen2.5-7B-Instruct`, used twice: zero-shot (same harness and prompts as the API models) and after QLoRA on the train split. Same weights before/after adaptation is what makes the comparison clean. No additional API models are added; the reviewers did not ask for more models, only for an adapted one. |
| Few-shot setting (Reviewer 2, 2) | The reviewers gave no k. We use k = 1 and k = 2 exemplars per class (7 and 14 exemplars; author decision 16 Oct 2026), drawn from the train split with a fixed seed, truncated to 120 words each, full 6,825-article test set, primary English prompt `en_v2`. |
| Prompt robustness (Reviewer 1, 2(v); Reviewer 2, 3) | Five prompts: `en_v1` (paper wording), `en_v2`, `en_v3`, `ar_v1`, `ar_v2`; report mean ± std and min–max per model. |
| Test set for LLMs | Full test set for every run (≈ $10 per zero-shot pass for GPT-4.1, less for the others). |
| Dense-head training | Early stopping on validation loss, lr 1e-3 (needed for the small-n learning curve); the paper's lr 5e-5 × 10 epochs is also run for the full-data row. |

---

## 3. Pending from the author side

| # | Item | Needed for | Status |
|---|---|---|---|
| P1 | OpenAI key / Gemini credits | — | **resolved 16 Oct**: GPT-4.1, Claude Haiku 4.5 and Gemini 3.8 Flash run through OpenRouter (`OPEN_ROUTER_KEY` in `.env`, $55 credit, key limit $50); runs launched 19:41–19:45 UTC |
| P1c | **DeepSeek route.** DeepSeek V3 and V3.1 are no longer serverless on Together. Default is option A, Together serverless `DeepSeek-V4-Flash-0731` (verified working, nothing to do); say so only if you prefer option B (official DeepSeek API, needs a new key in `.env` as `DEEPSEEK_API_KEY`). See `MODEL_SNAPSHOTS.md` §3. | DeepSeek runs | default A unless told otherwise |
| P2 | Spend caps on the OpenAI, Anthropic, Google and Together consoles (≈ $300 total is enough for everything in the plan). | all API runs | open |
| P3 | Rotate the keys that are stored in plaintext in the notebooks on OneDrive (Together and Gemini keys are literally in the `.ipynb` files). Not blocking, but worth doing after the runs. | security | open |
| P4 | vast.ai A100 (40 GB or more), CUDA 12 image, ≈ 1 day of time, with `git`/`ssh` access for me. | E2, E3, E4, E6a, E7, E8 | open |
| P5 | Confirm the manuscript source (LaTeX?) location so the text changes in `REVISION_PLAN.md` §5 can be drafted directly. | writing | open |
| P6 | Decide who writes the response letter; I can draft it from `REVISION_PLAN.md` §1–§2 once numbers are in. | submission | open |

## 3b. Update 17 Oct 2026

All experiments are complete and the GPU instance is destroyed. Author-side items still open: P3 (rotate the
keys that sit in plaintext in the old notebooks), P5 (manuscript source), P6 (who writes the response letter).
My side: M1–M5 done; M6 (tables/figures/CIs/McNemar) partly done (`results/tables/`), figures and the
manuscript/response-letter drafts remain (M6–M7). Final numbers: `EXPERIMENT_LOG.md` §7b.

## 4. Pending from my side

| # | Item | Depends on |
|---|---|---|
| M1 | API smoke tests (14 articles per provider). Anthropic: passing (zero-shot and Arabic 1-shot). DeepSeek V4 Flash on Together: passing (English and Arabic). Gemini: waits for credits. GPT: waits for a key. | P1, P1b |
| M2 | Zero-shot runs: 4 API models × 5 prompts on the full test set (E1). | P1, P2 |
| M3 | Few-shot runs k = 1, 3 (E5), input-length control (E6b), dateline control for the best LLM (E7). | M2 |
| M4 | On the A100: supervised learning curves, full fine-tuning, legacy-split replication, dateline intervention (E2, E3, E6a, E7, E8). | P4 |
| M5 | On the A100: Qwen2.5-7B-Instruct zero-shot via vLLM, QLoRA at n = 100 / 1000 / all, budget report (E4). | P4 |
| M6 | Tables, figures (learning curve, prompt spread), CIs, McNemar (E9); update `REVISION_PLAN.md` with the final numbers. | M2–M5 |
| M7 | Draft the revised Sections 3–5 text and the point-by-point response letter. | M6, P5, P6 |

## 5. Done so far

- Full audit of the submitted pipeline and of all saved prediction files (see `REVISION_PLAN.md` §1).
- Leakage / near-duplicate analysis on the submitted test splits.
- Canonical split built and verified (sha1 `951575cb…`).
- `train_supervised.py`, `llm_eval.py`, `lora_finetune.py`, `analysis.py` written and smoke-tested on CPU / offline.
- Model availability checked on 14 Oct 2026 against the live APIs (see `MODEL_SNAPSHOTS.md`).
