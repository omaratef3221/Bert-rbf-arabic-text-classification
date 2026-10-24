"""Mark, in blue, the parts of TEX_paper/elsarticle/manuscript_revised.tex that respond to the reviewers.

Two modes (idempotent: existing \\rev{...} / \\revblock markup is stripped first):

  --mode curated (default)
      Marks only what is listed in CURATED below: passages, sentences, headings and floats that were added
      or changed in response to the reviewers (new experiments, protocol details, corrections, the
      terminology fix).  Rewording of material that was already in the submitted version stays black.
      Edit the lists to move the line between "changed in substance" and "reworded".

  --mode diff
      Marks every sentence that differs from the submitted manuscript.tex (word-level inside sentences that
      are otherwise unchanged), every heading not present before and every float whose caption is new.
      Because the paper was rewritten almost completely, this marks most of the text.

The macros (\\rev, \\revblock, the \\ifrevmarkup switch and the title-page legend) are regenerated in the
preamble.  A clean copy compiles from the same file with  \\def\\revclean{1}  defined first.

    python mark_revisions.py                  # curated, rewrite manuscript_revised.tex in place
    python mark_revisions.py --mode diff --report
"""
import argparse
import difflib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_DIR = os.path.join(os.path.dirname(HERE), "TEX_paper", "elsarticle")
ORIG = os.path.join(TEX_DIR, "manuscript.tex")
REV = os.path.join(TEX_DIR, "manuscript_revised.tex")

# ----------------------------------------------------------------------------- what the curated mode marks
# Anchors are plain substrings of the current text.  "paragraphs": the whole paragraph / list item / \paragraph
# line containing the anchor.  "sentences": only the sentence containing the anchor.  "headings": section or
# paragraph titles.  "floats": labels of tables/figures (an \input{tables/tab_x} is matched by its label tab:x).
CURATED = {
    "paragraphs": [
        # abstract and keywords (the first abstract sentence is left black; see "sentences")
        "Arabic NLP\\sep multi-class text classification",
        # introduction: contributions (R1 M2, R2 C2, C3, C7)
        "\\item A paired benchmark of lightweight supervised classifiers",
        "\\item A supervised learning curve from 10 to about 4,500",
        "\\item A reproducible LLM evaluation protocol for Arabic classification",
        "\\item Two controlled interventions (input length and removal of the newspaper dateline)",
        # related work: parameter-efficient adaptation (R1 M2 i)
        "Between zero-shot inference and full fine-tuning lies parameter-efficient adaptation",
        # 3.1 deduplication and the split (R2 C8)
        "News corpora contain repeated and closely related stories, and a random split can place near-identical",
        # 3.2 what is trained (R1 M1, R2 C1, C2)
        "We evaluate five families of systems.",
        "The two frozen-encoder systems are the lightweight classifiers of interest",
        # 3.3 LLM evaluation protocol (R1 M2 ii-iv, R2 C3, C4, C5)
        "\\paragraph{Models and snapshots.}",
        "\\paragraph{Prompts.}",
        "\\textit{System: You are a precise news-topic classifier",
        "\\textit{User: Classify the Arabic news article below",
        "In the few-shot conditions, $k\\in\\{1,2\\}$ exemplars per class",
        "\\paragraph{Label mapping.}",
        "\\paragraph{Evaluation sets.}",
        # 3.4 training configuration incl. the fine-tuning and QLoRA protocols (R1 M1 iii, M2 i)
        "Table~\\ref{tab:hyperparameters} lists the training configuration of every trained system.",
        # 3.5 statistics (R1 M2 v), 3.6 reproducibility (R2 C4), 3.7 corrections (R2 C6)
        "We report accuracy and macro-averaged precision, recall and F1 on the test split",
        "Supervised training, full fine-tuning, QLoRA and the open-weight LLM inference ran on one NVIDIA",
        "While re-implementing the evaluation pipeline for this version we found three defects",
        # 4.1 main results and significance tests (R1 M2 v, R2 C8)
        "Table~\\ref{tab:main} reports all systems on the shared test split.",
        "The headline differences are all significant on paired predictions",
        # 4.3 learning curve (R2 C2), 4.4 robustness and mapping (R1 M2 iii, v; R2 C3, C5), 4.5 controls (R2 C7)
        "Figure~\\ref{fig:lc} and Table~\\ref{tab:lc} report the learning curve.",
        "Table~\\ref{tab:prompts} and Figure~\\ref{fig:prompts} report the sweeps on the fixed 700-article",
        "Table~\\ref{tab:mapping} audits the label-mapping step over every run of each model.",
        "Two interventions test whether either family relies on superficial cues.",
        # discussion (R2 C1, C7) and limitations
        "Third, the controlled analyses argue against two explanations that we had entertained",
        "\\paragraph{Limitations.}",
        # conclusion (first paragraph), data availability, appendix with the verbatim prompts (R2 C3, C4)
        "The split file with per-article checksums, the deduplication clusters, the evaluation subset",
        "All prompts present the seven labels and request exactly one.",
        "(original wording)]", "(primary)]", "\\item[\\texttt{en\\_v3}]", "\\item[\\texttt{ar\\_v1}]", "\\item[\\texttt{ar\\_v2}]",
    ],
    "sentences": [
        # abstract: everything after the opening sentence
        "The supervised systems use a frozen CAMeLBERT encoder with either a dense classification head",
        "are evaluated zero-shot and few-shot with deterministic decoding",
        "All systems are scored on the same deduplicated test split of 6,822 articles",
        "Zero-shot LLMs reach 91.5--92.8\\%",
        "A learning curve shows that about 100 labelled articles per class",
        "Two controlled interventions show that neither family relies on the newspaper dateline",
        # introduction framing and the systems compared (R2 C2)
        "paired comparisons between the two paradigms within the Arabic domain remain scarce",
        "On the supervised side we evaluate two lightweight systems built on a frozen CAMeLBERT encoder",
        "On the LLM side we evaluate GPT-4.1, Claude Haiku 4.5, Gemini 3.8 Flash, DeepSeek V4 Flash",
        "All systems are scored on one deduplicated test split, with deterministic decoding",
        # related work additions (R2 C8; R1 M2 v, R2 C3, C5) and the restated gap
        "broder1997resemblance",
        "Recent work cautions that zero-shot evaluation of LLMs on Arabic is highly sensitive",
        "Two consequences follow for benchmark design",
        "To our knowledge no prior study compares lightweight Arabic transformer classifiers",
        # 3.1 terminology (R1 minor 1, R2 C1)
        "Each article carries exactly one label, so the task is multi-class, single-label classification",
        # 4.2 class-level analysis: seed variation, the Religion result (R2 C6) and the replacement of speculation (R2 C7)
        "full fine-tuning removes most of them",
        "shows seed 0; the three seeds of the dense head trade",
        "The zero-shot LLMs are at or above the supervised systems on",
        "The pattern is the same for all four API models",
        "The articles involved are typically historical or devotional pieces",
        # discussion: what the gain is attributed to (R2 C1), supervision crossover and few-shot (R2 C2), Religion (R2 C6)
        "The question is not whether lightweight models are better than large language models",
        "The gain does not come from adapting the encoder to the domain",
        "When the encoder is adapted, by full fine-tuning or by QLoRA",
        "The learning curve places the crossover with zero-shot LLMs",
        "Their accuracy is nevertheless bounded by two things that in-context examples cannot fix",
        "Few-shot examples help the weaker models by one to five points",
        "The 7B open-weight model is the clearest illustration",
        # conclusion: the protocol and the supervision statement
        "under a protocol in which every system is scored on the same deduplicated test split",
        "About 100 labelled articles per class suffice for a 0.3",
        # 4.6 adaptation budget (R1 M2 i)
        "and QLoRA of the 7B model 3.3 GPU-hours on the full split",
        "Full fine-tuning takes 11.6 minutes on the full split",
        # qualitative analysis: examples now from the final runs
        "The examples below are drawn from the final runs on the shared test split",
    ],
    "headings": [
        "Zero-Shot, Few-Shot and Parameter-Efficient Use of LLMs", "Deduplication and near-duplicate-aware split.",
        "LLM Evaluation Protocol",
        "Evaluation and Statistical Analysis", "Infrastructure and Reproducibility",
        "Corrections to the Originally Reported Experiments", "How Much Supervision Does the Lightweight System Need?",
        "Prompt Robustness, Few-Shot Conditions and Output Mapping", "Controlled Analyses: Input Length and the Dateline",
        "Data and code availability", "Prompt variants",
    ],
    "floats": [
        "tab:split", "tab:models_summary", "tab:llm_snapshots", "tab:hyperparameters", "tab:main", "tab:perclass",
        "fig:lc", "tab:lc", "tab:prompts", "fig:prompts", "tab:mapping", "tab:lora",
    ],
}
LEGEND_CURATED = (r"\ifrevmarkup\vspace{1cm}{\color{blue}\small Text in blue marks additions and changes made in response to "
                  r"the reviewers' comments; tables and figures printed in blue are new. Passages that were only reworded "
                  r"are not marked.}\fi")
LEGEND_DIFF = (r"\ifrevmarkup\vspace{1cm}{\color{blue}\small Text in blue is new or changed relative to the submitted version "
               r"(revisions made in response to the reviewers); tables and figures printed in blue are new or regenerated.}\fi")

WHOLE_SENTENCE_BELOW = 0.8     # diff mode: similarity below which the whole sentence is marked
ABBR = re.compile(r"(?:\b(?:e\.g|i\.e|vs|et al|Fig|Figs|cf|approx|Eq|No|Prof|Sect|Tab|ca|resp|Dr|Mr|Ms|Eqs|al)|\b[A-Z])\.$")
BOUNDARY = re.compile(r"[.!?][)'\"\u2019\u201d]*(\s+)(?=[A-Z\\(\[`\"\u201c]|\d)")
SKIP_ENVS = {"table", "table*", "figure", "figure*", "tabular", "tabularx", "tabular*", "titlepage", "equation", "align"}
PREAMBLE = r"""% ---- revision markup: blue text = additions/changes for the reviewers (generated by revision/mark_revisions.py).
% Clean copy from the same file:  lualatex -jobname=manuscript_revised "\def\revclean{1}\input{manuscript_revised.tex}"
\newif\ifrevmarkup
\ifdefined\revclean\revmarkupfalse\else\revmarkuptrue\fi
\DeclareRobustCommand{\rev}[1]{\ifrevmarkup{\color{blue}#1}\else#1\fi}
\newcommand{\revblock}{\ifrevmarkup\color{blue}\fi}
"""
CMD_ONLY = re.compile(r"\\(label|centering|includegraphics|caption|small|footnotesize|toprule|midrule|bottomrule|addlinespace|cmidrule|multicolumn|setlength|resizebox|bibliography|bibliographystyle|appendix|newpage|clearpage|linenumbers|vspace|hspace|title|noindent|def|ifrevmarkup|sloppy)")


# ----------------------------------------------------------------------------- helpers
def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2]); i += 2; continue
        if c == "%":
            break
        out.append(c); i += 1
    return "".join(out)


def norm_tokens(s):
    s = re.sub(r"\\(?:cite[pt]?|ref|label|eqref|url|input|includegraphics|graphicspath)\*?(?:\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$~&\\]", " ", s)
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", s.lower())


def norm_word(w):
    return " ".join(norm_tokens(w))


def split_sentences(par):
    """Split a paragraph into (sentence, trailing_space) pieces; merges abbreviations and unbalanced groups."""
    pieces, start = [], 0
    for m in BOUNDARY.finditer(par):
        pieces.append((par[start:m.start(1)], m.group(1)))
        start = m.end()
    pieces.append((par[start:], ""))
    merged = []
    for s, ws in pieces:
        if merged:
            prev, pws = merged[-1]
            if ABBR.search(prev.rstrip()) or prev.count("{") != prev.count("}") or prev.count("$") % 2 == 1:
                merged[-1] = (prev + pws + s, ws)
                continue
        merged.append((s, ws))
    return [(s, ws) for s, ws in merged if s.strip()]


def balanced(s):
    return s.count("{") == s.count("}") and s.count("$") % 2 == 0 and "\\begin{" not in s and "\\end{" not in s


def strip_markup(text):
    """Remove \\rev{...} wrappers, \\revblock lines and {\\revblock\\input{...}} groups from a body."""
    text = re.sub(r"\{\\revblock(\\input\{tables/[^}]*\})\}", r"\1", text)
    text = re.sub(r"^[ \t]*\\revblock[ \t]*\n", "", text, flags=re.M)
    out, i = [], 0
    while True:
        j = text.find("\\rev{", i)
        if j < 0:
            out.append(text[i:]); break
        out.append(text[i:j])
        k, depth = j + 5, 1
        while k < len(text) and depth:
            if text[k] == "\\": k += 2; continue
            if text[k] == "{": depth += 1
            elif text[k] == "}": depth -= 1
            k += 1
        out.append(text[j + 5:k - 1]); i = k
    text = "".join(out)
    text = re.sub(r"\\ifrevmarkup\\vspace\{1cm\}\{\\color\{blue\}.*?\}\\fi\n?", "", text, flags=re.S)
    return text


def wrap(s):
    return "\\rev{" + s + "}"


# ----------------------------------------------------------------------------- diff mode
def original_sentences(path):
    body = open(path, encoding="utf-8").read().split("\\begin{document}", 1)[1]
    sents, caps, heads, par = [], [], [], []
    def flush():
        if par:
            sents.extend(s for s, _ in split_sentences(" ".join(par))); par.clear()
    for raw in body.split("\n"):
        line = strip_comment(raw).strip()
        if not line:
            flush(); continue
        m = re.match(r"\\(?:sub)*section\*?\{(.*)\}\s*$", line) or re.match(r"\\paragraph\{(.*?)\}", line)
        if m:
            flush(); heads.append(m.group(1)); continue
        m = re.match(r"\\caption\{(.*)\}\s*$", line)
        if m:
            flush(); caps.append(m.group(1)); sents.append(m.group(1)); continue
        if re.match(r"\\(begin|end|label|centering|includegraphics|toprule|midrule|bottomrule|input|vspace|hline|cmidrule|addlinespace|multicolumn|small|footnotesize|bibliography|appendix|newpage|clearpage|linenumbers|title)", line):
            flush(); continue
        line = re.sub(r"^\\item(\[[^\]]*\])?\s*", "", line)
        if "&" in line and line.endswith("\\\\"):
            sents.extend(c.strip() for c in line[:-2].split("&") if c.strip()); continue
        par.append(line.rstrip("\\").strip())
    flush()
    toks = [(s, norm_tokens(s)) for s in sents + caps]
    return [(s, t) for s, t in toks if t], [norm_tokens(h) for h in heads], [norm_tokens(c) for c in caps]


class Matcher:
    def __init__(self, orig):
        self.orig = orig
        self.sets = [set(t) for _, t in orig]

    def best(self, tokens):
        if not tokens:
            return 1.0, None
        ts = set(tokens)
        best, best_s = 0.0, None
        cands = sorted(range(len(self.orig)), key=lambda i: -len(ts & self.sets[i]) / max(1, len(ts | self.sets[i])))[:40]
        for i in cands:
            s, t = self.orig[i]
            r = difflib.SequenceMatcher(None, tokens, t, autojunk=False).ratio()
            if r > best:
                best, best_s = r, s
                if r == 1.0:
                    break
        return best, best_s


def mark_words(sentence, orig_sentence):
    words, owords = sentence.split(" "), orig_sentence.split(" ")
    sm = difflib.SequenceMatcher(None, [norm_word(w) for w in words], [norm_word(w) for w in owords], autojunk=False)
    out, changed = [], False
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        chunk = words[i1:i2]
        if tag == "equal" or not chunk:
            out.extend(chunk); continue
        run = " ".join(chunk)
        if not balanced(run):
            return None
        out.append(wrap(run)); changed = True
    return " ".join(out) if changed else sentence


class DiffMarker:
    def __init__(self, orig_path):
        orig, heads, caps = original_sentences(orig_path)
        self.m, self.hm, self.cm = Matcher(orig), Matcher([(" ".join(h), h) for h in heads if h]), Matcher([(" ".join(c), c) for c in caps if c])
        self.report = {k: [] for k in ("same", "partial", "whole", "unsafe", "floats_new", "floats_same", "heads_new", "heads_same")}
        self.legend = LEGEND_DIFF

    def paragraph(self, par):
        out = []
        for s, ws in split_sentences(par):
            toks = norm_tokens(s); ratio, best = self.m.best(toks)
            if ratio >= 0.999 or not toks:
                out.append(s + ws); self.report["same"].append(s); continue
            if ratio >= WHOLE_SENTENCE_BELOW and best is not None:
                mw = mark_words(s, best)
                if mw is not None:
                    out.append(mw + ws); self.report["partial"].append(s); continue
            if balanced(s):
                out.append(wrap(s) + ws); self.report["whole"].append(s)
            else:
                out.append(s + ws); self.report["unsafe"].append(s)
        return "".join(out)

    def heading(self, title):
        new = self.hm.best(norm_tokens(title))[0] < 0.999
        self.report["heads_new" if new else "heads_same"].append(title); return new

    def float_(self, caption, label, input_name):
        if input_name:
            self.report["floats_new"].append(input_name); return True
        new = (self.cm.best(norm_tokens(caption))[0] < 0.999) if caption else False
        self.report["floats_new" if new else "floats_same"].append(caption or label); return new


class CuratedMarker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.report = {k: [] for k in ("paragraphs", "sentences", "headings", "floats", "unused_anchors")}
        self.used = set()
        self.legend = LEGEND_CURATED

    def paragraph(self, par):
        for a in self.cfg["paragraphs"]:
            if a in par:
                self.used.add(a)
                if balanced(par):
                    self.report["paragraphs"].append(par[:80]); return wrap(par)
        out, hit = [], False
        for s, ws in split_sentences(par):
            if any(a in s for a in self.cfg["sentences"]) and balanced(s):
                self.used.update(a for a in self.cfg["sentences"] if a in s)
                out.append(wrap(s) + ws); hit = True; self.report["sentences"].append(s[:80])
            else:
                out.append(s + ws)
        return "".join(out)

    def heading(self, title):
        hit = any(a in title for a in self.cfg["headings"])
        if hit: self.used.update(a for a in self.cfg["headings"] if a in title); self.report["headings"].append(title)
        return hit

    def float_(self, caption, label, input_name):
        key = label or (input_name and input_name.replace("tab_", "tab:"))
        hit = key in self.cfg["floats"]
        if hit: self.used.add(key); self.report["floats"].append(key)
        return hit

    def finish(self):
        for k in ("paragraphs", "sentences", "headings"):
            self.report["unused_anchors"].extend(a for a in self.cfg[k] if a not in self.used)
        self.report["unused_anchors"].extend(f for f in self.cfg["floats"] if f not in self.used)


# ----------------------------------------------------------------------------- rewrite the file
def mark_file(rev_path, marker):
    pre, body = open(rev_path, encoding="utf-8").read().split("\\begin{document}", 1)
    body = strip_markup(body)
    pre = re.sub(r"% ---- revision markup.*?\\newcommand\{[^}]*\}\{\\ifrevmarkup\\color\{blue\}\\fi\}\n", "", pre, flags=re.S)
    pre = pre.replace("\\usepackage{lineno}", PREAMBLE + "\n\\usepackage{lineno}", 1)
    lines = body.split("\n")
    out, env, i = [], [], 0
    while i < len(lines):
        raw = lines[i]
        line = strip_comment(raw)
        stripped = line.strip()
        mb = re.match(r"\s*\\begin\{([a-zA-Z*]+)\}", line)
        me = re.match(r"\s*\\end\{([a-zA-Z*]+)\}", line)
        if mb:
            env.append(mb.group(1)); out.append(raw)
            if mb.group(1) in ("table", "table*", "figure", "figure*"):
                j, cap, lab = i + 1, None, None
                while j < len(lines) and not re.match(r"\s*\\end\{(table|figure)\*?\}", lines[j]):
                    mc = re.match(r"\s*\\caption\{(.*)\}\s*$", strip_comment(lines[j]))
                    ml = re.search(r"\\label\{([^}]*)\}", lines[j])
                    if mc: cap = mc.group(1)
                    if ml: lab = ml.group(1)
                    j += 1
                if marker.float_(cap, lab, None):
                    out.append("\\revblock")
            i += 1; continue
        if me:
            if env and env[-1] == me.group(1): env.pop()
            out.append(raw); i += 1; continue
        if stripped == "Email: ashraf@sharjah.ac.ae":
            out.append(raw); out.append(""); out.append(marker.legend); i += 1; continue
        if any(e in SKIP_ENVS for e in env):
            out.append(raw); i += 1; continue
        mi = re.match(r"\s*\\input\{tables/([^}]*)\}", stripped)
        if mi:
            if marker.float_(None, None, mi.group(1)):
                out.append(re.sub(r"(\\input\{tables/[^}]*\})", r"{\\revblock\1}", raw))
            else:
                out.append(raw)
            i += 1; continue
        mh = re.match(r"(\s*\\(?:sub)*section\*?\{)(.*)(\}\s*)$", line)
        if mh:
            title = mh.group(2)
            out.append(mh.group(1) + (wrap(title) if marker.heading(title) else title) + mh.group(3)); i += 1; continue
        mp = re.match(r"(\s*\\paragraph\{)(.*?)(\}\s*)(.*)$", line)
        if mp:
            title, rest = mp.group(2), mp.group(4)
            whole = isinstance(marker, CuratedMarker) and any(a in line for a in marker.cfg["paragraphs"])
            if whole:
                marker.used.update(a for a in marker.cfg["paragraphs"] if a in line); marker.report["paragraphs"].append(line[:80])
            t = wrap(title) if (whole or marker.heading(title)) else title
            body_txt = (wrap(rest) if whole and balanced(rest) else marker.paragraph(rest)) if rest.strip() else ""
            out.append(mp.group(1) + t + mp.group(3) + body_txt); i += 1; continue
        if not stripped or CMD_ONLY.match(stripped) or stripped.startswith("%"):
            out.append(raw); i += 1; continue
        mit = re.match(r"(\s*\\item(?:\[[^\]]*\])?\s*)(.*)$", line)
        prefix, content = (mit.group(1), mit.group(2)) if mit else (re.match(r"\s*", line).group(0), stripped)
        trail = ""
        if content.endswith("\\\\"):
            content, trail = content[:-2].rstrip(), " \\\\"
        if mit and isinstance(marker, CuratedMarker) and any(a in line for a in marker.cfg["paragraphs"]):
            marker.used.update(a for a in marker.cfg["paragraphs"] if a in line); marker.report["paragraphs"].append(line[:80])
            out.append(prefix + (wrap(content) if balanced(content) else content) + trail)
        else:
            out.append(prefix + marker.paragraph(content) + trail)
        i += 1
    open(rev_path, "w", encoding="utf-8").write(pre + "\\begin{document}" + "\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["curated", "diff"], default="curated")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--orig", default=ORIG)
    ap.add_argument("--rev", default=REV)
    a = ap.parse_args()
    marker = CuratedMarker(CURATED) if a.mode == "curated" else DiffMarker(a.orig)
    mark_file(a.rev, marker)
    if isinstance(marker, CuratedMarker):
        marker.finish()
        r = marker.report
        print(f"curated: {len(r['paragraphs'])} paragraphs/items, {len(r['sentences'])} sentences, {len(r['headings'])} headings, {len(r['floats'])} floats marked")
        if r["unused_anchors"]:
            print("ANCHORS NOT FOUND:"); [print("  -", x) for x in r["unused_anchors"]]
    else:
        n = {k: len(v) for k, v in marker.report.items()}
        print("diff: sentences unchanged %(same)d, partly changed %(partial)d, new/rewritten %(whole)d, unsafe %(unsafe)d | floats new %(floats_new)d same %(floats_same)d | headings new %(heads_new)d same %(heads_same)d" % n)
    if a.report:
        for k, v in marker.report.items():
            print(f"\n== {k} =="); [print("  -", s[:140]) for s in v]


if __name__ == "__main__":
    main()
