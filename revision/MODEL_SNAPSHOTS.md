# LLM snapshots for the revision: what the paper used, what is still served, what we run now

Checked against the live APIs on **14 Oct 2026** with the project keys. This file is the source of
truth for the "Models and inference settings" paragraph of the revised Section 3.3 and for the
reproducibility table the reviewers asked for (Reviewer 1, 2(ii); Reviewer 2, point 4).

Rule agreed with the authors: **use the original model id where it is still served; where it has
been retired, use the closest same-tier successor that is still served, and state both the id and
the evaluation date in the paper.** The id the API actually served is logged for every request by
`llm_eval.py` (`served_model` field), so the paper can report served ids, not just requested ones.

## 1. Summary table

| Role in the paper | What the paper says | What was actually called (code audit) | Status on 14 Oct 2026 | **Model for the revision** | Verified? |
|---|---|---|---|---|---|
| GPT | GPT-4.1 | May 2025: `gpt-4.1` via the Responses API, default temperature, 1,000-doc sample. Dec 2025: `gpt-4o-mini`, temperature 0 | `gpt-4.1-2025-04-14` is the dated GPT-4.1 snapshot; availability could not be checked because **both project OpenAI keys return HTTP 401** | `gpt-4.1-2025-04-14` (original, pinned) | blocked on a valid key |
| Claude | Claude 3.5 Haiku | `claude-3-5-haiku-20241022`, `max_tokens=1` (May 2025) / `max_tokens=10` (Dec 2025), default temperature | **Retired.** API returns 404 and a deprecation notice (end of life 19 Feb 2026) | `claude-haiku-4-5-20251001` (Claude Haiku 4.5, dated snapshot; the direct successor of 3.5 Haiku in the same Haiku tier) | **yes**: 14/14 smoke test, one-word outputs, temperature 0 accepted |
| Gemini | Gemini 2.0 Flash | `gemini-2.0-flash` (alias), default temperature | **Retired.** Not listed. The next generation `gemini-2.5-flash` is listed but returns 404 "no longer available to new users"; Google's error points to `gemini-3.8-flash`. All 3.x Flash models currently return 402 "prepayment credits are depleted" on the project key | `gemini-3.5-flash` (oldest generally-available Flash the key can reach), falling back to `gemini-3.8-flash` if 3.5 is also closed to this account once credits are added | blocked on Gemini credits |
| DeepSeek | DeepSeek V3 | `deepseek-ai/DeepSeek-V3` on Together AI, `temperature=1` (May 2025 and Dec 2025) | **Retired on Together.** `deepseek-ai/DeepSeek-V3.1` is listed but is a dedicated-endpoint ("non-serverless") model on this account (HTTP 400). Serverless DeepSeek models: `DeepSeek-V4-Flash-0731`, `DeepSeek-V4-Pro-0813`, `DeepSeek-V4.1-Flash`, `DeepSeek-R1-0528` | `deepseek-ai/DeepSeek-V4-Flash-0731` on Together serverless (dated checkpoint, same provider as the paper), reasoning switched off; alternative in §3 | **yes**: 14/14 (English prompt) and 7/7 (Arabic prompt), exact labels, 2 output tokens per answer, no reasoning text |
| Open-weight (new) | not in the paper | — | `Qwen/Qwen2.5-7B-Instruct` weights are public on Hugging Face | `Qwen/Qwen2.5-7B-Instruct` in bf16 on the A100 (zero-shot via vLLM, then QLoRA) | runs on the GPU box |

## 2. Why these replacements

**Claude Haiku 4.5 for Claude 3.5 Haiku.** Same product tier ("Haiku", the small fast model), direct
successor, dated snapshot id, and it still accepts `temperature=0` so the decoding can be made
deterministic. The newer Claude Haiku 5.5 rejects non-default sampling parameters and has
adaptive thinking on by default, which would make the comparison with the other APIs less
controlled. Note for the paper: the 3.5 Haiku number in the submitted manuscript (75 %) was a
parsing artefact (see `REVISION_PLAN.md` §1, A1); it is replaced, not reproduced.

**Gemini 3.5 Flash (or 3.8 Flash) for Gemini 2.0 Flash.** Same "Flash" tier. Gemini 2.5 Flash would
have been the nearest successor but Google has closed it to new users. Among the 3.x Flash models,
3.5 is the oldest generally-available one; 3.8 is the one Google's own error message recommends.
Gemini 3.x models reason ("think") by default; the harness sends `thinking_budget=0` and, if a
model rejects that, `thinking_level="minimal"`, and logs the number of thinking tokens per request
so the paper can state whether any reasoning tokens were used.

**DeepSeek V4 Flash (0731) for DeepSeek V3.** Same provider as the paper (Together AI serverless),
a dated open-weight checkpoint, and the smallest/cheapest member of the current DeepSeek line, which
matches the "Flash"/"mini" tier of the other three APIs better than V4 Pro would. DeepSeek V4 is a
reasoning model: with reasoning on, a 16-token budget is consumed by the hidden reasoning and the
answer comes back empty (observed on 14/14 articles). The harness therefore sends
`chat_template_kwargs: {"thinking": false}` (DeepSeek's documented chat-template switch) and logs
whether any `reasoning_content` was returned; with the switch, every answer is exactly the label
(2 output tokens). The alternative is the official DeepSeek API (§3, option B).

**Qwen2.5-7B-Instruct as the open-weight model.** Reviewer 1 asked for "at least one open-weight LLM"
adapted with LoRA/QLoRA; Reviewer 2 asked for parameter-efficient adaptation. Qwen2.5-7B-Instruct
is a widely used 7B instruction model with strong Arabic coverage, fits an A100-40GB in 4-bit for
QLoRA, and is supported by Unsloth and vLLM. The same weights are used zero-shot (same prompts and
harness as the API models) and after adaptation, which isolates the effect of task adaptation.

## 3. DeepSeek: decision needed

| Option | Model id | Pros | Cons |
|---|---|---|---|
| A (recommended) | Together AI serverless `deepseek-ai/DeepSeek-V4-Flash-0731` (dated checkpoint) | pinned open-weight checkpoint, same provider as the paper, cheap, no new account | two generations after V3 |
| B | Official DeepSeek API `deepseek-chat` | closest to "DeepSeek V3" by name | needs a new DeepSeek API key; `deepseek-chat` is a floating alias, so the served version must be recorded on the day |
| C | Together dedicated endpoint for `DeepSeek-V3.1` | closest checkpoint to V3 | hourly GPU cost, overkill for 7 M tokens |

If the authors have no preference, A is used. Serverless availability is read from Together's
model list at run time; the exact id is logged.

## 4. Decoding and logging settings (identical for every API model)

| Setting | Value | Notes |
|---|---|---|
| temperature | 0 | Anthropic SDK 1.x: sent via `extra_body`; Gemini: `GenerateContentConfig.temperature` |
| top_p | 1 | not sent to Anthropic (temperature and top_p should not both be set) |
| max output tokens | 16 | labels are 1–3 tokens; the paper's `max_tokens=1` is what truncated Arabic answers |
| seed | 42 | honoured by OpenAI and Gemini; ignored elsewhere |
| system prompt | per prompt variant | `en_v2`, `ar_v2` use one; others do not |
| thinking | off | Gemini 3.x: `thinking_budget=0` or `thinking_level="minimal"`, thinking tokens logged |
| retries | up to 6 with exponential backoff (2 s → 60 s) on 429 / 5xx / timeouts only | 400/401/402/403/404/422 and client-side errors fail fast and are recorded |
| timeout | 60 s per request | |
| logged per request | raw output, finish reason, input/output/thinking tokens, served model id, latency, attempts, error, timestamp | JSONL in `results/` |
| scoring | strict mapping (7 labels + Arabic names); unmapped / ambiguous / empty outputs count as errors; lenient mapping reported as a sensitivity | `llm_eval.py --summarize` |

## 5. Smoke-test log

| Date | Provider | Model | Result |
|---|---|---|---|
| 2026-10-14 | Anthropic | `claude-3-5-haiku-20241022` | 404 not_found (retired) |
| 2026-10-14 | Anthropic | `claude-haiku-4-5-20251001` | 14/14 correct, all outputs exact label strings, 5 s wall-clock for 14 requests |
| 2026-10-14 | Gemini | `gemini-2.5-flash`, `gemini-2.5-flash-lite` | 404 "no longer available to new users" |
| 2026-10-14 | Gemini | `gemini-3.5-flash`, `gemini-3.8-flash`, `gemini-3-flash-preview` | 402 "prepayment credits are depleted" (model exists; account needs credit) |
| 2026-10-14 | Together | `deepseek-ai/DeepSeek-V3.1` | 400 "non-serverless model" |
| 2026-10-14 | Together | `deepseek-ai/DeepSeek-V4-Flash-0731`, reasoning on | reachable, but 14/14 answers empty (`finish_reason=length`, all 16 tokens used by reasoning) |
| 2026-10-14 | Together | `deepseek-ai/DeepSeek-V4-Flash-0731`, reasoning off | 14/14 correct (English prompt), 7/7 correct (Arabic prompt), exact labels, 0.6–1.7 s per call |
| 2026-10-14 | Anthropic | `claude-haiku-4-5-20251001`, Arabic system prompt, 1-shot | 7/7 correct, exact labels, ~4–7 k input tokens per request with 7 exemplars |
| 2026-10-14 | OpenAI | any | 401 invalid key (both keys found in the project) |
