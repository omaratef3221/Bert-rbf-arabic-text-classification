"""Mark, in blue, what is new or changed in TEX_paper/elsarticle/manuscript_revised.tex relative to the
submitted manuscript.tex.  Idempotent: existing \\rev{...} / \\revblock markup is stripped first.

Granularity
  * prose sentences (paragraphs, list items, abstract, keywords, quoted prompt text):
      - identical to a sentence of the submitted version (after normalisation)  -> untouched
      - close to one (word-level similarity >= 0.8)                              -> only the changed words in \\rev{...}
      - otherwise                                                               -> the whole sentence in \\rev{...}
  * section / paragraph headings not present in the submitted version          -> title in \\rev{...}
  * floats whose caption is not in the submitted version, and every \\input table -> whole float in blue (\\revblock)

The macros (\\rev, \\revblock, the \\ifrevmarkup switch and the title-page legend) are inserted into the
preamble if missing.  A clean copy compiles from the same file with  \\def\\revclean{1}  defined first.

    python mark_revisions.py            # rewrite manuscript_revised.tex in place
    python mark_revisions.py --report   # also print what was and was not marked
"""
import argparse
import difflib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_DIR = os.path.join(os.path.dirname(HERE), "TEX_paper", "elsarticle")
ORIG = os.path.join(TEX_DIR, "manuscript.tex")
REV = os.path.join(TEX_DIR, "manuscript_revised.tex")

WHOLE_SENTENCE_BELOW = 0.8     # similarity below which the whole sentence is marked
ABBR = re.compile(r"(?:\b(?:e\.g|i\.e|vs|et al|Fig|Figs|cf|approx|Eq|No|Prof|Sect|Tab|ca|resp|Dr|Mr|Ms|Eqs|al)|\b[A-Z])\.$")
BOUNDARY = re.compile(r"[.!?][)'\"\u2019\u201d]*(\s+)(?=[A-Z\\(\[`\"\u201c]|\d)")
SKIP_ENVS = {"table", "table*", "figure", "figure*", "tabular", "tabularx", "tabular*", "titlepage", "equation", "align"}
PREAMBLE_MARK = "% ---- revision markup"
PREAMBLE = r"""% ---- revision markup: blue text = new or changed relative to the submitted version (see mark_revisions.py).
% Clean copy from the same file:  lualatex -jobname=manuscript_revised "\def\revclean{1}\input{manuscript_revised.tex}"
\newif\ifrevmarkup
\ifdefined\revclean\revmarkupfalse\else\revmarkuptrue\fi
\DeclareRobustCommand{\rev}[1]{\ifrevmarkup{\color{blue}#1}\else#1\fi}
\newcommand{\revblock}{\ifrevmarkup\color{blue}\fi}
"""
LEGEND = (r"\ifrevmarkup\vspace{1cm}{\color{blue}\small Text in blue is new or changed relative to the submitted version "
          r"(revisions made in response to the reviewers); tables and figures printed in blue are new or regenerated.}\fi")


# ----------------------------------------------------------------------------- normalisation and sentences
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
    t = norm_tokens(w)
    return " ".join(t)


def split_sentences(par):
    """Split a paragraph into (sentence, trailing_space) pieces; merges abbreviations and unbalanced groups."""
    pieces, start = [], 0
    for m in BOUNDARY.finditer(par):
        end = m.start(1)
        pieces.append((par[start:end], m.group(1)))
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


# ----------------------------------------------------------------------------- strip existing markup
def strip_markup(text):
    text = text.replace("{\\revblock", "{").replace("\\revblock\n", "")
    text = re.sub(r"\n\\revblock\n", "\n", text)
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
    text = re.sub(r"\{\\input\{(tables/[^}]*)\}\}", r"\\input{\1}", text)
    return text


# ----------------------------------------------------------------------------- corpus of the submitted version
def original_sentences(path):
    body = open(path, encoding="utf-8").read().split("\\begin{document}", 1)[1]
    sents, caps, heads = [], [], []
    par = []
    def flush():
        if par:
            text = " ".join(par)
            sents.extend(s for s, _ in split_sentences(text))
            par.clear()
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
        if re.match(r"\\(begin|end|label|centering|includegraphics|toprule|midrule|bottomrule|input|vspace|hline|cmidrule|addlinespace|multicolumn|small|footnotesize|bibliography|appendix|newpage|clearpage|linenumbers|title|end)", line):
            flush(); continue
        line = re.sub(r"^\\item(\[[^\]]*\])?\s*", "", line)
        if "&" in line and line.endswith("\\\\"):
            sents.extend(c.strip() for c in line[:-2].split("&") if c.strip()); continue
        par.append(line.rstrip("\\").strip())
    flush()
    toks = [(s, norm_tokens(s)) for s in sents + caps]
    toks = [(s, t) for s, t in toks if t]
    return toks, [norm_tokens(h) for h in heads], [norm_tokens(c) for c in caps]


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


# ----------------------------------------------------------------------------- marking
def mark_words(sentence, orig_sentence):
    """Wrap only the changed word runs of `sentence` (vs orig_sentence) in \\rev{}; None if not safely possible."""
    words = sentence.split(" ")
    owords = orig_sentence.split(" ")
    keys = [norm_word(w) for w in words]
    okeys = [norm_word(w) for w in owords]
    sm = difflib.SequenceMatcher(None, keys, okeys, autojunk=False)
    out, changed_any = [], False
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        chunk = words[i1:i2]
        if tag == "equal" or not chunk:
            out.extend(chunk); continue
        run = " ".join(chunk)
        if not balanced(run) or run.count("\\rev{"):
            return None
        out.append("\\rev{" + run + "}"); changed_any = True
    return " ".join(out) if changed_any else sentence


def mark_paragraph(par, matcher, report):
    pieces = split_sentences(par)
    out = []
    for s, ws in pieces:
        toks = norm_tokens(s)
        ratio, best = matcher.best(toks)
        if ratio >= 0.999 or not toks:
            out.append(s + ws); report["same"].append(s); continue
        if ratio >= WHOLE_SENTENCE_BELOW and best is not None:
            mw = mark_words(s, best)
            if mw is not None:
                out.append(mw + ws); report["partial"].append(s); continue
        if balanced(s):
            out.append("\\rev{" + s + "}" + ws); report["whole"].append(s)
        else:
            out.append(s + ws); report["unsafe"].append(s)
    text = "".join(out)
    # merge adjacent whole-sentence wrappers: "}\s+\rev{" -> keep as is (separate groups are harmless)
    return text


def mark_file(orig_path, rev_path, report):
    orig, heads, caps = original_sentences(orig_path)
    matcher = Matcher(orig)
    head_matcher = Matcher([(" ".join(h), h) for h in heads if h])
    cap_matcher = Matcher([(" ".join(c), c) for c in caps if c])
    pre, body = open(rev_path, encoding="utf-8").read().split("\\begin{document}", 1)
    body = strip_markup(body)                      # only the body carries markup; the preamble block is regenerated
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
            env.append(mb.group(1))
            out.append(raw)
            if mb.group(1) in ("table", "table*", "figure", "figure*"):
                # look ahead for the caption of this float
                j, cap = i + 1, None
                while j < len(lines) and not re.match(r"\s*\\end\{(table|figure)\*?\}", lines[j]):
                    mc = re.match(r"\s*\\caption\{(.*)\}\s*$", strip_comment(lines[j]))
                    if mc: cap = mc.group(1)
                    j += 1
                ratio = cap_matcher.best(norm_tokens(cap))[0] if cap else 0.0
                if ratio < 0.999:
                    out.append("\\revblock"); report["floats_new"].append(cap or "(no caption)")
                else:
                    report["floats_same"].append(cap)
            i += 1; continue
        if me:
            if env and env[-1] == me.group(1): env.pop()
            out.append(raw); i += 1; continue
        if stripped == "Email: ashraf@sharjah.ac.ae" and "\\ifrevmarkup" not in body:
            out.append(raw); out.append(""); out.append(LEGEND); i += 1; continue
        if any(e in SKIP_ENVS for e in env):
            out.append(raw); i += 1; continue
        if stripped.startswith("\\input{tables/"):
            out.append(re.sub(r"(\\input\{tables/[^}]*\})", r"{\\revblock\1}", raw)); report["floats_new"].append(stripped); i += 1; continue
        mh = re.match(r"(\s*\\(?:sub)*section\*?\{)(.*)(\}\s*)$", line)
        if mh:
            title = mh.group(2)
            if head_matcher.best(norm_tokens(title))[0] < 0.999:
                out.append(mh.group(1) + "\\rev{" + title + "}" + mh.group(3)); report["heads_new"].append(title)
            else:
                out.append(raw); report["heads_same"].append(title)
            i += 1; continue
        mp = re.match(r"(\s*\\paragraph\{)(.*?)(\}\s*)(.*)$", line)
        if mp:
            title, rest = mp.group(2), mp.group(4)
            t = "\\rev{" + title + "}" if head_matcher.best(norm_tokens(title))[0] < 0.999 else title
            out.append(mp.group(1) + t + mp.group(3) + (mark_paragraph(rest, matcher, report) if rest.strip() else "")); i += 1; continue
        if not stripped or re.match(r"\\(label|centering|includegraphics|caption|small|footnotesize|toprule|midrule|bottomrule|addlinespace|cmidrule|multicolumn|setlength|resizebox|bibliography|bibliographystyle|appendix|newpage|clearpage|linenumbers|vspace|hspace|title|noindent|def|ifrevmarkup)", stripped) or stripped.startswith("%"):
            out.append(raw); i += 1; continue
        mi = re.match(r"(\s*\\item(?:\[[^\]]*\])?\s*)(.*)$", line)
        prefix, content = (mi.group(1), mi.group(2)) if mi else (re.match(r"\s*", line).group(0), stripped)
        trail = ""
        if content.endswith("\\\\"):
            content, trail = content[:-2].rstrip(), " \\\\"
        out.append(prefix + mark_paragraph(content, matcher, report) + trail)
        i += 1
    open(rev_path, "w", encoding="utf-8").write(pre + "\\begin{document}" + "\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--orig", default=ORIG)
    ap.add_argument("--rev", default=REV)
    a = ap.parse_args()
    report = {k: [] for k in ("same", "partial", "whole", "unsafe", "floats_new", "floats_same", "heads_new", "heads_same")}
    mark_file(a.orig, a.rev, report)
    n = {k: len(v) for k, v in report.items()}
    print("sentences: unchanged %(same)d, partly changed %(partial)d, new/rewritten %(whole)d, unsafe-unmarked %(unsafe)d | floats new %(floats_new)d same %(floats_same)d | headings new %(heads_new)d same %(heads_same)d" % n)
    if a.report:
        for k in ("same", "partial", "unsafe", "floats_same", "heads_same"):
            print(f"\n== {k} ==")
            for s in report[k]:
                print("  -", s[:140])


if __name__ == "__main__":
    main()
