#!/bin/zsh
# Build the revised manuscript: marked copy (blue), clean copy, latexdiff tracked-changes copy; render pages.
# Run from anywhere:  zsh TEX_paper/build_revised.sh      (needs LuaLaTeX, latexmk, latexdiff, bibtex; python3 + pypdfium2 for the page images)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p build_rev/tables build_rev/Figures_revised build_rev/Figures
cp elsarticle/manuscript_revised.tex elsarticle/manuscript.tex elsarticle/elsarticle.cls elsarticle/elsarticle-harv.bst build_rev/
cp elsarticle/tables/*.tex build_rev/tables/
cp references.bib references_revision.bib build_rev/
cp Figures_revised/*.pdf build_rev/Figures_revised/
cp "Figures/data categories dist perc.png" build_rev/Figures/ 2>/dev/null || true
python3 - <<'PY'
import pathlib, re, sys
sys.path.insert(0, "../revision")
from mark_revisions import strip_markup
root = pathlib.Path(".")
pre, body = (root/"elsarticle/manuscript_revised.tex").read_text(encoding="utf-8").split("\\begin{document}", 1)
pre = re.sub(r"% ---- revision markup.*?\\newcommand\{[^}]*\}\{\\ifrevmarkup\\color\{blue\}\\fi\}\n", "", pre, flags=re.S)
s = pre + "\\begin{document}" + strip_markup(body)
new = re.sub(r"\\input\{tables/([A-Za-z_]+)\}", lambda m: (root/f"elsarticle/tables/{m.group(1)}.tex").read_text(encoding="utf-8"), s)
(root/"build_rev/new_flat.tex").write_text(new, encoding="utf-8")
old = (root/"elsarticle/manuscript.tex").read_text(encoding="utf-8").replace(r"\bibliography{references}", r"\bibliography{references,references_revision}")
(root/"build_rev/old_flat.tex").write_text(old, encoding="utf-8")
PY
cd build_rev
latexmk -lualatex -interaction=nonstopmode -bibtex -jobname=manuscript_revised_marked manuscript_revised.tex > compile_marked.log 2>&1 && echo "marked: ok" || { echo "marked: FAILED"; grep -E "^!" manuscript_revised_marked.log | head -5; }
latexmk -lualatex -interaction=nonstopmode -bibtex -jobname=manuscript_revised -usepretex -pretex='\def\revclean{1}' manuscript_revised.tex > compile.log 2>&1 && echo "clean: ok" || { echo "clean: FAILED"; grep -E "^!" manuscript_revised.log | head -5; }
grep "Overfull" manuscript_revised.log | sed -E 's/Overfull \\hbox \(([0-9.]+)pt too wide\) in (paragraph|alignment) at lines ([0-9-]+).*/  overfull \1pt \2 lines \3/' | awk '$2+0 > 5' | sort -rn | head -6
latexdiff --encoding=utf8 --math-markup=0 --graphics-markup=0 --config "PICTUREENV=(?:picture|DIFnomarkup|table|tabular|tabularx|figure)[\w\d*@]*" old_flat.tex new_flat.tex > manuscript_diff.tex 2> latexdiff.err
rm -f manuscript_diff.aux manuscript_diff.bbl manuscript_diff.fdb_latexmk manuscript_diff.fls
{ lualatex -interaction=nonstopmode manuscript_diff.tex && bibtex manuscript_diff && lualatex -interaction=nonstopmode manuscript_diff.tex && lualatex -interaction=nonstopmode manuscript_diff.tex; } > compile_diff.log 2>&1 && echo "diff: ok" || { echo "diff: FAILED"; grep -E "^!" manuscript_diff.log | sort | uniq -c | head -5; }
echo "  undefined refs: clean $(grep -c 'undefined' manuscript_revised.log), marked $(grep -c 'undefined' manuscript_revised_marked.log), diff $(grep -c 'undefined' manuscript_diff.log) (diff: a few refs to deleted figures are expected)"
rm -f *_p??.png
python3 -c "import pypdfium2" 2>/dev/null || python3 -m pip install -q --user pypdfium2 2>/dev/null || true
python3 -c "
import pypdfium2 as pdfium
for n in ['manuscript_revised', 'manuscript_revised_marked', 'manuscript_diff']:
    pdf = pdfium.PdfDocument(n + '.pdf'); print(n, 'pages', len(pdf))
    for i in range(len(pdf)): pdf[i].render(scale=1.3).to_pil().save(f'{n}_p{i+1:02d}.png')
" || echo "(page images skipped: pypdfium2 not available)"
