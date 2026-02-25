#!/usr/bin/env bash
# Sync backend/app/static → docs/ for GitHub Pages deployment.
# Run from the project root: ./sync_docs.sh

set -euo pipefail

SRC="backend/app/static"
DST="docs"

echo "Syncing CSS..."
cp "$SRC/blog.css" "$DST/blog.css"

echo "Syncing HTML (adjusting paths for static hosting)..."
sed \
  -e 's|href="/static/blog.css"|href="blog.css"|g' \
  -e 's|src="/static/blog.js"|src="blog.js"|g' \
  -e 's|<a href="/legacy"[^<]*</a>[^<]*<a href="/legacy/swadesh"[^<]*</a>[^<]*<a href="/legacy/validation"[^<]*</a>|<span style="color:var(--fg-dim)">Interactive Explorer and legacy pages require the full backend.</span>|g' \
  "$SRC/blog.html" > "$DST/index.html"

echo "Syncing JS (patching API paths → static data/ paths)..."
sed \
  -e 's|fetchJSON("/api/results/sample-concept")|fetchJSON("data/sample_concept.json")|g' \
  -e 's|fetchJSON("/api/results/swadesh-convergence")|fetchJSON("data/swadesh_convergence.json")|g' \
  -e 's|fetchJSON("/api/results/phylogenetic")|fetchJSON("data/phylogenetic.json")|g' \
  -e 's|fetchJSON("/api/results/swadesh-comparison")|fetchJSON("data/swadesh_comparison.json")|g' \
  -e 's|fetchJSON("/api/results/colexification")|fetchJSON("data/colexification.json")|g' \
  -e 's|fetchJSON("/api/results/conceptual-store")|fetchJSON("data/conceptual_store.json")|g' \
  -e 's|fetchJSON("/api/results/offset-invariance")|fetchJSON("data/offset_invariance.json")|g' \
  -e 's|fetchJSON("/api/results/color-circle")|fetchJSON("data/color_circle.json")|g' \
  -e 's|fetchJSON("/api/data/swadesh")|fetchJSON("data/swadesh_corpus.json")|g' \
  -e '/^async function runExplorer() {$/a\
  alert("The Interactive Explorer requires the full backend with the NLLB model running. See the project README for local setup instructions.");\
  return;\
  \/\/ --- original code below (unreachable in static build) ---' \
  "$SRC/blog.js" > "$DST/blog.js"

echo "Done. docs/ is now in sync with backend/app/static/."
