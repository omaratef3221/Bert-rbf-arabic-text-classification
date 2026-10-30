"""Page and line-number index of a compiled manuscript PDF (lineno margin numbers), for the
"Change location" entries of the response letter.

    python letter_locations.py ../TEX_paper/revised/manuscript_revised_tracked_changes.pdf
    python letter_locations.py <pdf> --probe 3        # dump page 3 as extracted, to check the parsing

Headings, table and figure captions are located by their printed text; each heading is reported
with the page(s) and the line-number range up to the next heading.
"""
import argparse
import re

import pypdfium2 as pdfium

HEAD = re.compile(r"^(?:(\d+(?:\.\d+)*)\.\s+|Appendix\s+([A-Z])\.\s+)([A-Z][^\n]{2,90})$")   # "3.2. Systems", "4. Results", "Appendix A. ..."
CAPTION = re.compile(r"^(Table|Figure)\s+([A-Z]?\.?\d+):")
LINE_PREFIX = re.compile(r"^(\d{1,4})\s+(.*)$")
PARA_HEADS = ["Deduplication and near-duplicate-aware split.", "Models and snapshots.", "Prompts.", "Label mapping.",
              "Evaluation sets.", "Limitations."]
OTHER = ["Abstract", "Keywords:", "Data and code availability", "References"]


def page_lines(page):
    """Return [(line_number or None, text)] in reading order; margin line numbers are separate tokens."""
    txt = page.get_textpage().get_text_range()
    raw = [l.strip() for l in txt.split("\n")]
    out, pending = [], None
    for l in raw:
        if not l:
            continue
        l = re.sub(r"[✿︎]+", "", l).strip()          # latexdiff underline glyphs in the tracked-changes copy
        if not l:
            continue
        if re.fullmatch(r"\d{1,4}", l):
            pending = int(l); continue
        m = LINE_PREFIX.match(l)
        if m:
            out.append((int(m.group(1)), m.group(2))); pending = None; continue
        out.append((pending, l)); pending = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--probe", type=int, default=0)
    a = ap.parse_args()
    pdf = pdfium.PdfDocument(a.pdf)
    if a.probe:
        for ln, t in page_lines(pdf[a.probe - 1]):
            print(f"{ln if ln is not None else '':>5}  {t[:110]}")
        return
    items = []            # (kind, title, page, line)
    for pi in range(len(pdf)):
        printed = pi + 1  # elsarticle: title page unnumbered; printed page = pdf page - 1 for pages > 1
        for ln, t in page_lines(pdf[pi]):
            m = HEAD.match(t)
            if m:
                items.append(("heading", t, printed, ln)); continue
            m = CAPTION.match(t)
            if m:
                items.append((m.group(1).lower(), f"{m.group(1)} {m.group(2)}", printed, ln)); continue
            for ph in PARA_HEADS:
                if t.startswith(ph):
                    items.append(("paragraph", ph, printed, ln)); break
            else:
                for o in OTHER:
                    if t.startswith(o):
                        items.append(("other", o, printed, ln)); break
    # line range of each heading = its first line .. line before the next heading/paragraph-head
    heads = [x for x in items if x[0] in ("heading", "paragraph", "other")]
    # last line number per page for ranges that end at a page break
    last_line = {}
    for pi in range(len(pdf)):
        nums = [ln for ln, _ in page_lines(pdf[pi]) if ln is not None]
        last_line[pi + 1] = max(nums) if nums else None
    print(f"{'kind':10s} {'item':62s} {'pdf page':9s} lines")
    for i, (kind, title, page, ln) in enumerate(heads):
        if i + 1 < len(heads):
            nk, nt, npg, nln = heads[i + 1]
            end = (nln - 1) if nln else last_line.get(npg)
            pages = f"p{page}" if npg == page else f"pp{page}-{npg}"
        else:
            end, pages = last_line.get(page), f"p{page}"
        rng = f"{ln}-{end}" if ln and end else (str(ln) if ln else "?")
        print(f"{kind:10s} {title[:62]:62s} {pages:9s} {rng}")
    print()
    for kind, title, page, ln in items:
        if kind in ("table", "figure"):
            print(f"{kind:10s} {title:62s} p{page}")


if __name__ == "__main__":
    main()
