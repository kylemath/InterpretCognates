#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== InterpretCognates Paper Build ==="
echo ""

# --- Step 1: Run analysis scripts (if Python available) ---
if command -v python3 &> /dev/null; then
    echo "=== Step 1: Running analysis scripts ==="

    if [ ! -d "scripts/.venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv scripts/.venv
    fi
    source scripts/.venv/bin/activate
    pip install -q -r scripts/requirements.txt 2>/dev/null

    python3 scripts/run_all.py

    deactivate
    echo ""
else
    echo "=== Step 1: SKIPPED (python3 not found) ==="
    echo "Figures and stats must already exist in figures/ and output/"
    echo ""
fi

# --- Step 2: Check prerequisites ---
echo "=== Step 2: Checking prerequisites ==="

if [ ! -f "output/stats.tex" ]; then
    echo "WARNING: output/stats.tex not found. LaTeX compilation may fail."
fi

MISSING_FIGS=0
for fig in fig_swadesh_ranking fig_phylogenetic fig_swadesh_comparison \
           fig_colexification fig_conceptual_store fig_color_circle \
           fig_offset_combined fig_water_manifold \
           fig_category_summary fig_isotropy_validation fig_mantel_scatter \
           fig_concept_map fig_offset_vector_demo; do
    if [ ! -f "figures/${fig}.pdf" ]; then
        echo "WARNING: figures/${fig}.pdf not found"
        MISSING_FIGS=$((MISSING_FIGS + 1))
    fi
done

if [ $MISSING_FIGS -gt 0 ]; then
    echo "WARNING: $MISSING_FIGS figure(s) missing. Compilation may fail."
fi
echo ""

# --- Step 3: Compile LaTeX ---
echo "=== Step 3: Compiling LaTeX ==="
mkdir -p build

pdflatex -interaction=nonstopmode -output-directory=build main.tex || true
bibtex build/main || true
pdflatex -interaction=nonstopmode -output-directory=build main.tex || true
pdflatex -interaction=nonstopmode -output-directory=build main.tex || true

if [ -f "build/main.pdf" ]; then
    cp build/main.pdf main.pdf
    echo ""
    echo "=== SUCCESS: main.pdf created ==="
else
    echo ""
    echo "=== FAILED: main.pdf not created ==="
    echo "Check build/main.log for errors"
    exit 1
fi
