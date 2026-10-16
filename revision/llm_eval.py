"""
Reproducible LLM classification harness (zero-shot and few-shot) for SANAD-AlKhaleej.

Fixes every reproducibility gap the reviewers raised:
  * one canonical test split shared with the supervised models (revision/data_split.py)
  * deterministic decoding: temperature=0, top_p=1, fixed max_tokens, seed where supported
  * pinned model snapshot ids; the id the API actually served is logged per request
  * every RAW output is stored (JSONL), together with finish_reason, token usage, latency, retries
  * label mapping is deterministic, audited (method per row) and has NO random fallback;
    unmapped / ambiguous outputs are scored as errors and reported as rates
  * several semantically equivalent prompts (English and Arabic), few-shot with exemplars
    drawn from the TRAIN split only, resumable runs, bounded retry policy

Examples
  python llm_eval.py --provider openai    --model gpt-4.1-2025-04-14           --prompt en_v1
  python llm_eval.py --provider anthropic --model claude-3-5-haiku-20241022     --prompt ar_v1 --shots 1
  python llm_eval.py --provider gemini    --model gemini-2.0-flash-001          --prompt en_v2
  python llm_eval.py --provider deepseek  --model deepseek-chat                 --prompt en_v1
  python llm_eval.py --provider together  --model deepseek-ai/DeepSeek-V3       --prompt en_v1
  python llm_eval.py --provider vllm --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct --prompt en_v1
  python llm_eval.py --provider mock --model mock --prompt en_v1 --max-docs 50      # offline dry run
  python llm_eval.py --summarize results/<run>.jsonl                                 # re-score a finished run
"""
import os, re, sys, json, time, argparse, hashlib, threading, random, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_split import load_split, fewshot_exemplars, CATS  # noqa: E402

LABELS = CATS  # ['Culture','Finance','Medical','Politics','Religion','Sports','Tech']
DATELINE = re.compile(r"^\s*[ء-ي ]{2,20}?\s*[:\-–]?\s*[«\"']?\s*الخليج\s*[»\"']?\s*[:\-–]?\s*")  # "<city>: «الخليج»" prefix (E7)
AR_LABEL = {"Culture": "ثقافة", "Finance": "اقتصاد", "Medical": "طب", "Politics": "سياسة",
            "Religion": "دين", "Sports": "رياضة", "Tech": "تقنية"}

# --------------------------------------------------------------------------------------------
# Prompts. en_v1 reproduces the wording of the submitted paper as closely as possible.
# All variants ask for exactly one label; the few-shot block is inserted only when --shots > 0.
# --------------------------------------------------------------------------------------------
PROMPTS = {
    "en_v1": dict(lang="en", system=None, user=(
        "Act as an Arabic Text Classifier that has to classify between the following:\n{labels}\n\n"
        "Write only the final output class without anything more.\n{examples}"
        "Text:\n{text}\n\nClass:")),
    "en_v2": dict(lang="en", system="You are a precise news-topic classifier for Modern Standard Arabic.", user=(
        "Classify the Arabic news article below into exactly one of these categories:\n{labels}\n\n"
        "Respond with the category name only, copied exactly from the list, with no explanation.\n{examples}"
        "Article:\n{text}\n\nCategory:")),
    "en_v3": dict(lang="en", system=None, user=(
        "Which section of a Gulf Arabic newspaper would the following article appear in? "
        "Choose one of: {labels}.\n{examples}"
        "Article:\n{text}\n\nAnswer with one word from the list:")),
    "ar_v1": dict(lang="ar", system=None, user=(
        "تصرّف كمصنّف نصوص عربية. صنّف المقال التالي إلى واحدة فقط من الفئات التالية:\n{labels}\n\n"
        "اكتب اسم الفئة فقط كما هو مكتوب في القائمة دون أي إضافة.\n{examples}"
        "المقال:\n{text}\n\nالفئة:")),
    "ar_v2": dict(lang="ar", system="أنت مصنّف دقيق لموضوعات الأخبار المكتوبة بالعربية الفصحى.", user=(
        "حدّد القسم الصحفي الذي ينتمي إليه المقال التالي. الأقسام المتاحة هي:\n{labels}\n\n"
        "أجب بكلمة واحدة فقط من القائمة أعلاه.\n{examples}"
        "المقال:\n{text}\n\nالقسم:")),
}


def label_list(lang):
    if lang == "ar":
        return ", ".join(f"{en} ({AR_LABEL[en]})" for en in LABELS)
    return str(LABELS)


def build_prompt_parts(prompt_name, text, exemplars=None):
    """Return (system, prefix, suffix). prefix = instruction + labels + examples (identical for every article of a run,
    hence prompt-cacheable); suffix = the article + answer cue. prefix + suffix is the exact user message."""
    p = PROMPTS[prompt_name]
    ex_block = ""
    if exemplars is not None and len(exemplars):
        hdr = "أمثلة:" if p["lang"] == "ar" else "Examples:"
        lines = [hdr]
        for _, r in exemplars.iterrows():
            lines.append(f"Text: {r.text}\nClass: {r.category}\n" if p["lang"] == "en"
                         else f"المقال: {r.text}\nالفئة: {r.category}\n")
        ex_block = "\n".join(lines) + "\n"
    tpl = p["user"]
    cut = tpl.index("{text}")
    prefix = tpl[:cut].format(labels=label_list(p["lang"]), examples=ex_block)
    suffix = tpl[cut:].format(text=text)
    return p["system"], prefix, suffix


def build_messages(prompt_name, text, exemplars=None):
    system, prefix, suffix = build_prompt_parts(prompt_name, text, exemplars)
    return system, prefix + suffix


# --------------------------------------------------------------------------------------------
# Output -> label mapping. Deterministic, audited, no randomness.
#   method in {exact, normalized, arabic, synonym, contains, AMBIGUOUS, UNMAPPED, EMPTY}
#   strict=True  : only the 7 labels and their Arabic names are accepted (primary metric)
#   strict=False : also accepts common synonyms (Economy->Finance, Health->Medical, ...) (sensitivity)
# --------------------------------------------------------------------------------------------
ARABIC_SYNONYMS = {
    "Sports": ["رياضة", "الرياضة", "رياضية", "الرياضية"],
    "Culture": ["ثقافة", "الثقافة", "ثقافية", "الثقافية", "ثقافه", "الثقافه"],
    "Finance": ["اقتصاد", "الاقتصاد", "مالية", "المالية", "اقتصادية", "الاقتصادية", "مال", "المال", "ماليه", "الماليه"],
    "Medical": ["طب", "الطب", "طبية", "الطبية", "صحة", "الصحة", "طبي", "الطبي", "صحه", "الصحه"],
    "Politics": ["سياسة", "السياسة", "سياسية", "السياسية", "سياسه", "السياسه"],
    "Religion": ["دين", "الدين", "ديني", "الديني", "دينية", "الدينية", "إسلام", "الإسلام", "اسلام", "الاسلام"],
    "Tech": ["تقنية", "التقنية", "تكنولوجيا", "التكنولوجيا", "تقنيه", "التقنيه", "تقني", "التقني"],
}
EN_SYNONYMS = {  # lenient tier only
    "Finance": ["economy", "economic", "economics", "business", "financial", "finances", "markets"],
    "Medical": ["health", "medicine", "healthcare"],
    "Tech": ["technology", "technological", "science and technology"],
    "Sports": ["sport"],
    "Religion": ["religious", "islamic", "islam"],
    "Politics": ["political", "politic"],
    "Culture": ["cultural", "arts", "art", "heritage", "history", "literature"],
}
_PREFIX = re.compile(r"^\s*(class|category|answer|the class is|label|الفئة|القسم|التصنيف|الفئة هي)\s*[:：\-]\s*", re.I)
_STRIP = "\"'`«»“”‘’*_[]()<>.,;!؟?،:\n\t "


def normalize(raw):
    s = (raw or "").strip()
    s = _PREFIX.sub("", s)
    s = s.strip(_STRIP)
    s = re.sub(r"\s+", " ", s)
    return s


def map_output(raw, strict=True):
    """Return (label or UNMAPPED/AMBIGUOUS/EMPTY, method)."""
    if raw is None or not str(raw).strip():
        return "EMPTY", "EMPTY"
    s = normalize(str(raw))
    low = s.lower()
    for L in LABELS:
        if low == L.lower():
            return L, "exact" if str(raw).strip() == L else "normalized"
    for L, words in ARABIC_SYNONYMS.items():
        if s in words:
            return L, "arabic"
    if not strict:
        for L, words in EN_SYNONYMS.items():
            if low in words:
                return L, "synonym"
    # containment: which labels are mentioned anywhere in the (short) output?
    tokens = set(re.findall(r"[a-z]+|[ء-ي]+", low))
    hits = set()
    for L in LABELS:
        if L.lower() in tokens or (L == "Sports" and "sport" in tokens):
            hits.add(L)
    for L, words in ARABIC_SYNONYMS.items():
        if tokens & set(words):
            hits.add(L)
    if not strict:
        for L, words in EN_SYNONYMS.items():
            if tokens & set(words):
                hits.add(L)
    if len(hits) == 1:
        return hits.pop(), "contains"
    if len(hits) > 1:
        return "AMBIGUOUS", "AMBIGUOUS"
    return "UNMAPPED", "UNMAPPED"


# --------------------------------------------------------------------------------------------
# Providers. Each returns dict(raw, finish_reason, in_tokens, out_tokens, served_model, extra)
# --------------------------------------------------------------------------------------------
class Provider:
    name = "base"

    def __init__(self, model, max_tokens, temperature=0.0, top_p=1.0, seed=42, base_url=None, api_key=None):
        self.model, self.max_tokens, self.temperature, self.top_p, self.seed = model, max_tokens, temperature, top_p, seed
        self.base_url, self.api_key = base_url, api_key
        self.cache_key = None  # set by the runner (= run tag); prompt caching is always on where the provider supports it

    def complete(self, system, prefix, suffix):
        raise NotImplementedError


class OpenAICompatible(Provider):
    """OpenAI, DeepSeek (api.deepseek.com), Together, vLLM / any OpenAI-compatible server."""
    name = "openai"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0, timeout=60)

    def complete(self, system, prefix, suffix):
        # The shared prefix comes first in the user turn, so OpenAI's automatic prefix caching (>= 1024 tokens) applies;
        # prompt_cache_key pins all requests of a run to the same cache route. Together/vLLM ignore these fields.
        user_content = prefix + suffix
        if self.name == "openrouter" and self.model.startswith("anthropic/"):  # OpenRouter passes cache_control through to Anthropic
            user_content = [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": suffix}]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_content}]
        kw = dict(model=self.model, messages=msgs, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        if self.name == "openai":
            kw["seed"] = self.seed
            if self.cache_key:
                kw["extra_body"] = {"prompt_cache_key": self.cache_key}
        elif self.name == "openrouter":
            eb = {"seed": self.seed}
            if self.or_providers:  # pin the upstream provider(s) so the served model/quantization is fixed and reportable
                eb["provider"] = {"order": self.or_providers, "allow_fallbacks": False}
            if re.search(r"deepseek|qwen|glm|kimi", self.model, re.I):
                eb["reasoning"] = {"enabled": False}
            kw["extra_body"] = eb
        else:  # open-weight reasoning models (DeepSeek V3.1+/V4, Qwen3): switch "thinking" off via the chat template,
            #    otherwise the whole token budget is spent on reasoning and `content` comes back empty (verified 2026-10-14)
            kw["extra_body"] = {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
        r = self.client.chat.completions.create(**kw)
        c = r.choices[0]
        reasoning = getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None)
        ptd = getattr(r.usage, "prompt_tokens_details", None)
        upstream = (getattr(r, "provider", None) or (getattr(r, "model_extra", None) or {}).get("provider")) if self.name == "openrouter" else None
        return dict(raw=c.message.content, finish_reason=c.finish_reason,
                    in_tokens=getattr(r.usage, "prompt_tokens", None), out_tokens=getattr(r.usage, "completion_tokens", None),
                    served_model=(f"{r.model} via {upstream}" if upstream else r.model),
                    extra={"system_fingerprint": getattr(r, "system_fingerprint", None),
                           "reasoning_chars": (len(reasoning) if reasoning else None),
                           "cached_tokens": getattr(ptd, "cached_tokens", None)})


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key, max_retries=0, timeout=60)

    def complete(self, system, prefix, suffix):
        # anthropic SDK >= 1.0 removed temperature/top_p from the method signature; models up to the 4.5 line still
        # honour them when sent in the request body (Haiku 4.5 does; Haiku 5.5 rejects non-default values).
        # Prompt caching: an explicit breakpoint after the shared prefix block caches system + prefix (5-minute TTL,
        # refreshed by every request); prefixes below the model's minimum cacheable length are silently not cached.
        content = [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": suffix}]
        kw = dict(model=self.model, max_tokens=self.max_tokens, messages=[{"role": "user", "content": content}],
                  extra_body={"temperature": self.temperature})
        if system:
            kw["system"] = system
        r = self.client.messages.create(**kw)
        txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        u = r.usage
        cread, cwrite = getattr(u, "cache_read_input_tokens", 0) or 0, getattr(u, "cache_creation_input_tokens", 0) or 0
        return dict(raw=txt, finish_reason=r.stop_reason, in_tokens=u.input_tokens + cread + cwrite, out_tokens=u.output_tokens,
                    served_model=r.model, extra={"cached_tokens": cread, "cache_write_tokens": cwrite})


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=60_000))  # ms; no default timeout otherwise
        self._cache, self._cache_err, self._lock = None, None, threading.Lock()

    def _explicit_cache(self, system, prefix):
        """Create one explicit context cache per run for the shared prefix (guaranteed cache-read pricing). Prefixes below
        the model's minimum (zero-shot prompts) are rejected by the API; then implicit caching is the fallback."""
        with self._lock:
            if self._cache is None:
                try:
                    c = self.client.caches.create(model=self.model, config=self.types.CreateCachedContentConfig(
                        contents=[prefix], system_instruction=system or None, ttl="10800s", display_name=(self.cache_key or "sanad")[:120]))
                    self._cache = c.name
                except Exception as e:
                    self._cache, self._cache_err = "", f"{type(e).__name__}: {str(e)[:120]}"
            return self._cache

    def complete(self, system, prefix, suffix):
        kw = dict(temperature=self.temperature, top_p=self.top_p, candidate_count=1, max_output_tokens=self.max_tokens, seed=self.seed)
        if not re.search(r"gemini-(1\.|2\.0)", self.model):  # 2.5+ models think by default; a 16-token budget would be eaten by thinking
            kw["thinking_config"] = self.types.ThinkingConfig(thinking_budget=0, include_thoughts=False)
        cache = self._explicit_cache(system, prefix)
        if cache:
            kw["cached_content"] = cache
            contents = suffix
        else:  # implicit caching still applies to identical prefixes >= the model minimum
            kw["system_instruction"] = system or None
            contents = prefix + suffix
        r = self.client.models.generate_content(model=self.model, contents=contents, config=self.types.GenerateContentConfig(**kw))
        cand = r.candidates[0] if r.candidates else None
        fr = str(getattr(cand, "finish_reason", None)) if cand else "NO_CANDIDATE"
        um = getattr(r, "usage_metadata", None)
        try:
            txt = r.text if cand else ""
        except Exception:
            txt = ""
        return dict(raw=txt, finish_reason=fr,
                    in_tokens=getattr(um, "prompt_token_count", None), out_tokens=getattr(um, "candidates_token_count", None),
                    served_model=getattr(r, "model_version", self.model),
                    extra={"thinking_tokens": getattr(um, "thoughts_token_count", None), "cached_tokens": getattr(um, "cached_content_token_count", None),
                           "explicit_cache": bool(cache)})


class MockProvider(Provider):
    """Offline dry run: deterministic pseudo-labels with a few deliberately messy outputs."""
    name = "mock"

    def complete(self, system, prefix, suffix):
        user = prefix + suffix
        h = int(hashlib.sha1(user.encode()).hexdigest(), 16)
        lab = LABELS[h % 7]
        style = h % 23
        raw = {0: f"Class: {lab}", 1: f"['{lab}']", 2: lab.lower(), 3: AR_LABEL[lab], 4: "ال", 5: f"{lab} or Culture",
               6: "Economy"}.get(style, lab)
        return dict(raw=raw, finish_reason="stop", in_tokens=len(user) // 4, out_tokens=3, served_model="mock", extra={})


def make_provider(args):
    key = lambda env: os.environ.get(env) or (_ for _ in ()).throw(SystemExit(f"missing {env} in environment/.env"))
    common = dict(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, seed=args.seed)
    if args.provider == "openai":
        return OpenAICompatible(args.model, api_key=key("OPENAI_API_KEY"), **common)
    if args.provider == "deepseek":
        p = OpenAICompatible(args.model, api_key=key("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", **common); p.name = "deepseek"; return p
    if args.provider == "together":
        p = OpenAICompatible(args.model, api_key=key("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1", **common); p.name = "together"; return p
    if args.provider == "openrouter":  # one key for several vendors; model ids like openai/gpt-4.1, anthropic/claude-haiku-4.5, google/gemini-3.8-flash
        p = OpenAICompatible(args.model, api_key=key("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", **common)
        p.name = "openrouter"; p.or_providers = [s for s in (args.or_provider or "").split(",") if s]; return p
    if args.provider == "vllm":
        p = OpenAICompatible(args.model, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"), base_url=args.base_url or "http://localhost:8000/v1", **common); p.name = "vllm"; return p
    if args.provider == "anthropic":
        return AnthropicProvider(args.model, api_key=key("ANTHROPIC_API_KEY"), **common)
    if args.provider == "gemini":
        return GeminiProvider(args.model, api_key=os.environ.get("GEMINI_API_KEY") or key("GOOGLE_API_KEY"), **common)
    if args.provider == "mock":
        return MockProvider(args.model, **common)
    raise SystemExit(f"unknown provider {args.provider}")


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------
def call_with_retry(provider, system, prefix, suffix, max_attempts=6, log=print):
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            r = provider.complete(system, prefix, suffix)
            r["latency_s"] = round(time.time() - t0, 3)
            r["attempts"] = attempt
            r["error"] = None
            return r
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            transient = not isinstance(e, (TypeError, ValueError, AttributeError, KeyError)) and code not in (400, 401, 402, 403, 404, 422)
            if not transient:  # programming errors and non-retryable API errors: fail fast, record, move on
                log(f"  non-retryable error: {err}")
                return dict(raw=None, finish_reason="ERROR", in_tokens=None, out_tokens=None, served_model=None,
                            extra={}, latency_s=None, attempts=attempt, error=err)
            if attempt == max_attempts:
                return dict(raw=None, finish_reason="ERROR", in_tokens=None, out_tokens=None, served_model=None,
                            extra={}, latency_s=None, attempts=attempt, error=err)
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)


def run(args):
    try:
        from dotenv import load_dotenv
        load_dotenv(args.env_file) if args.env_file else load_dotenv()
    except ImportError:
        pass
    df = load_split(args.root, args.split_csv)
    data = df[df.split == args.split].reset_index(drop=True)
    if args.subset:  # fixed stratified evaluation subset shared by every model (persisted so it is identical everywhere)
        sub_path = os.path.join(HERE, "splits", f"llm_subset_{args.subset}.csv")
        if not os.path.exists(sub_path):
            ids = data.groupby("category").sample(args.subset // 7, random_state=0)["doc_id"]
            pd.DataFrame({"doc_id": sorted(ids)}).to_csv(sub_path, index=False)
        ids = set(pd.read_csv(sub_path)["doc_id"])
        data = data[data.doc_id.isin(ids)].reset_index(drop=True)
    if args.max_docs:
        data = data.groupby("category").head(max(1, args.max_docs // 7)).reset_index(drop=True)
    exemplars = fewshot_exemplars(df, args.shots, args.fewshot_seed, args.shot_words) if args.shots > 0 else None
    if exemplars is not None:
        exemplars = exemplars.sample(frac=1.0, random_state=args.fewshot_seed).reset_index(drop=True)  # fixed interleaved order
    provider = make_provider(args)
    tag = f"{args.provider}__{re.sub(r'[^A-Za-z0-9.-]+', '_', args.model)}__{args.prompt}__k{args.shots}" + (f"__trunc{args.truncate_words}" if args.truncate_words else "") + ("__nodateline" if args.strip_dateline else "") + (f"__sub{args.subset}" if args.subset else "") + (f"__{args.tag_suffix}" if args.tag_suffix else "")
    provider.cache_key = tag  # one prompt-cache namespace per run (OpenAI prompt_cache_key, Gemini cache display name)
    out = args.out or os.path.join(HERE, "results", f"{tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if args.overwrite and os.path.exists(out):
        os.remove(out)
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
    todo = data[~data.doc_id.isin(done)]
    meta = dict(run=tag, provider=args.provider, model=args.model, prompt=args.prompt, shots=args.shots, fewshot_seed=args.fewshot_seed,
                shot_words=args.shot_words, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed,
                split=args.split, split_csv=os.path.basename(args.split_csv), n_docs=int(len(data)), truncate_words=args.truncate_words, subset=args.subset,
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(), exemplar_ids=(exemplars.doc_id.tolist() if exemplars is not None else []),
                prompt_text=PROMPTS[args.prompt])
    with open(out.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] {len(todo)} to do, {len(done)} already done -> {out}")
    lock = threading.Lock()

    def work(row):
        text = DATELINE.sub("", row.text, count=1) if args.strip_dateline else row.text
        text = text if not args.truncate_words else " ".join(text.split()[: args.truncate_words])
        system, prefix, suffix = build_prompt_parts(args.prompt, text, exemplars)
        r = call_with_retry(provider, system, prefix, suffix)
        lab, method = map_output(r["raw"], strict=True)
        lab_l, method_l = map_output(r["raw"], strict=False)
        rec = dict(doc_id=row.doc_id, gold=row.category, raw=r["raw"], pred=lab, map_method=method, pred_lenient=lab_l,
                   map_method_lenient=method_l, finish_reason=r["finish_reason"], in_tokens=r["in_tokens"], out_tokens=r["out_tokens"],
                   served_model=r["served_model"], latency_s=r["latency_s"], attempts=r["attempts"], error=r["error"],
                   ts=datetime.datetime.now(datetime.timezone.utc).isoformat(), **{k: v for k, v in r["extra"].items() if v})
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    t0 = time.time()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, row) for row in todo.itertuples(index=False)]
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            n_ok += rec["pred"] == rec["gold"]
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  running acc={n_ok / i:.3f}  {time.time() - t0:.0f}s")
    res = summarize(out)
    if args.wandb_project:  # opt-in W&B: one run per (provider, model, prompt, shots) with the summary metrics and the mapping audit
        import wandb
        w = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=tag, job_type="llm", tags=["revision", "llm", args.provider],
                       group=f"{args.provider}_{re.sub(r'[^A-Za-z0-9.-]+', '_', args.model)}", config={k: v for k, v in meta.items() if k not in ("exemplar_ids", "prompt_text")}, reinit=True)
        w.log({"accuracy": res["strict"]["accuracy"], "macro_f1": res["strict"]["macro_f1"], "accuracy_lenient": res["lenient"]["accuracy"],
               "unmapped_rate": res["unmapped_rate"], "non_exact_rate": res["non_exact_rate"], "api_error_rate": res["error_rate_api"],
               "input_tokens": res["tokens"]["input"], "output_tokens": res["tokens"]["output"], "cached_share": res["tokens"]["cached_share"],
               "cache_write_tokens": res["tokens"]["cache_write"], "latency_s_median": res["latency_s_median"] or 0,
               "n": res["n"], **{f"{c}/recall": res["strict"]["per_class"][c]["recall"] for c in LABELS},
               **{f"{c}/precision": res["strict"]["per_class"][c]["precision"] for c in LABELS},
               **{f"mapping/{k}": v / res["n"] for k, v in res["mapping_methods"].items()}})
        w.finish()


def summarize(jsonl):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    rows = [json.loads(l) for l in open(jsonl)]
    d = pd.DataFrame(rows).drop_duplicates("doc_id", keep="last")
    res = {}
    for tier, col in [("strict", "pred"), ("lenient", "pred_lenient")]:
        y, p = d["gold"], d[col]
        pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
        res[tier] = dict(accuracy=float((y == p).mean()), macro_f1=float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
                         per_class={L: dict(precision=float(a), recall=float(b), f1=float(c), support=int(s)) for L, a, b, c, s in zip(LABELS, pr, rc, f1, sup)})
    mm = d["map_method"].value_counts().to_dict()
    res["n"] = int(len(d))
    res["mapping_methods"] = mm
    res["unmapped_rate"] = float(d["pred"].isin(["UNMAPPED", "AMBIGUOUS", "EMPTY"]).mean())
    res["non_exact_rate"] = float((d["map_method"] != "exact").mean())
    res["error_rate_api"] = float(d["error"].notna().mean())
    res["finish_reasons"] = d["finish_reason"].astype(str).value_counts().to_dict()
    res["served_models"] = d["served_model"].astype(str).value_counts().to_dict()
    cached = d["cached_tokens"].fillna(0) if "cached_tokens" in d else pd.Series(0, index=d.index)
    cwrite = d["cache_write_tokens"].fillna(0) if "cache_write_tokens" in d else pd.Series(0, index=d.index)
    res["tokens"] = dict(input=int(d["in_tokens"].fillna(0).sum()), output=int(d["out_tokens"].fillna(0).sum()),
                         cached_read=int(cached.sum()), cache_write=int(cwrite.sum()),
                         cached_share=float(cached.sum() / max(1, d["in_tokens"].fillna(0).sum())))
    res["latency_s_median"] = float(d["latency_s"].median()) if d["latency_s"].notna().any() else None
    res["unmapped_examples"] = d.loc[d["pred"].isin(["UNMAPPED", "AMBIGUOUS", "EMPTY"]), "raw"].astype(str).head(30).tolist()
    res["confusion_strict"] = pd.crosstab(d["gold"], d["pred"]).to_dict()
    with open(jsonl.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[{os.path.basename(jsonl)}] n={res['n']} strict acc={res['strict']['accuracy']:.4f} macroF1={res['strict']['macro_f1']:.4f} | "
          f"lenient acc={res['lenient']['accuracy']:.4f} | unmapped={res['unmapped_rate']:.2%} non-exact={res['non_exact_rate']:.2%} | "
          f"Religion recall={res['strict']['per_class']['Religion']['recall']:.3f} | tokens in={res['tokens']['input']} cached={res['tokens']['cached_share']:.0%} | served={list(res['served_models'])}")
    return res


def _selftest():
    cases = {"Religion": ("Religion", "exact"), " religion\n": ("Religion", "normalized"), "Class: Culture": ("Culture", "normalized"),
             "['Religion']": ("Religion", "normalized"), "الرياضة": ("Sports", "arabic"), "ال": ("UNMAPPED", "UNMAPPED"),
             "الس": ("UNMAPPED", "UNMAPPED"), "The class is Finance.": ("Finance", "contains"), "Religion or Culture": ("AMBIGUOUS", "AMBIGUOUS"),
             "History": ("UNMAPPED", "UNMAPPED"), "Economy": ("UNMAPPED", "UNMAPPED"), "الإخبارية": ("UNMAPPED", "UNMAPPED"), "": ("EMPTY", "EMPTY"),
             "Sport": ("Sports", "contains"), "Tech.": ("Tech", "normalized"), "الفئة: دين": ("Religion", "arabic")}
    bad = 0
    for raw, exp in cases.items():
        got = map_output(raw, strict=True)
        if got != exp:
            bad += 1
            print("MISMATCH", repr(raw), "->", got, "expected", exp)
    assert map_output("Economy", strict=False)[0] == "Finance" and map_output("History", strict=False)[0] == "Culture"
    print("mapper self-test:", "OK" if bad == 0 else f"{bad} mismatches")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic", "gemini", "deepseek", "together", "openrouter", "vllm", "mock"])
    ap.add_argument("--or-provider", default="", help="openrouter only: comma-separated upstream provider slugs to pin, e.g. 'openai' or 'anthropic'; fallbacks disabled")
    ap.add_argument("--model", help="pinned snapshot id, e.g. gpt-4.1-2025-04-14, claude-3-5-haiku-20241022, gemini-2.0-flash-001")
    ap.add_argument("--prompt", default="en_v1", choices=list(PROMPTS))
    ap.add_argument("--shots", type=int, default=0, help="exemplars PER CLASS from the train split (0 = zero-shot)")
    ap.add_argument("--fewshot-seed", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--shot-words", type=int, default=120)
    ap.add_argument("--truncate-words", type=int, default=0, help="feed only the first N words (input-length control)")
    ap.add_argument("--strip-dateline", action="store_true", help="remove the '<city>: «الخليج»' dateline prefix before prompting (lexical-shortcut control)")
    ap.add_argument("--tag-suffix", default="", help="suffix for the run name/output file, e.g. rep2 for a repeated determinism run")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--split-csv", default=os.path.join(HERE, "splits", "sanad_alkhaleej_split_v1.csv"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--max-docs", type=int, default=0, help="smoke test: stratified head of N docs")
    ap.add_argument("--subset", type=int, default=0, help="evaluate on the fixed stratified subset splits/llm_subset_<N>.csv (N/7 per class, seed 0; created on first use)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summarize", default=None, help="only re-score this results .jsonl")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"), help="log the run summary to W&B (off when unset)")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.summarize:
        summarize(a.summarize)
    else:
        if not (a.provider and a.model):
            ap.error("--provider and --model are required")
        run(a)
