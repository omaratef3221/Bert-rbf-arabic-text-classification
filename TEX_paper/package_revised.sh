#!/bin/zsh
# Copy the built PDFs into TEX_paper/revised/ and rebuild overleaf_upload.zip from the sources.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
cp build_rev/manuscript_revised.pdf revised/manuscript_revised.pdf
cp build_rev/manuscript_revised_marked.pdf revised/manuscript_revised_marked.pdf
cp build_rev/manuscript_diff.pdf revised/manuscript_revised_tracked_changes.pdf
cp build_rev/manuscript_diff.tex elsarticle/manuscript_tracked_changes.tex
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/tables" "$STAGE/Figures_revised" "$STAGE/Figures"
cp elsarticle/manuscript_revised.tex elsarticle/elsarticle.cls elsarticle/elsarticle-harv.bst references.bib references_revision.bib "$STAGE/"
cp elsarticle/manuscript.tex "$STAGE/manuscript_submitted_v1.tex"
cp build_rev/manuscript_diff.tex "$STAGE/manuscript_tracked_changes.tex"
cp elsarticle/tables/*.tex "$STAGE/tables/"
cp Figures_revised/*.pdf "$STAGE/Figures_revised/"
cp "Figures/data categories dist perc.png" "$STAGE/Figures/"
(cd "$STAGE" && rm -f overleaf_upload.zip && zip -qr "$ROOT/revised/overleaf_upload.zip" . -x '.*')
rm -rf "$STAGE"
ls -la revised | awk 'NR>3 {print $5, $9}'
