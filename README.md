# InterpretCognates

[![arXiv](https://img.shields.io/badge/arXiv-2603.02258v2-b31b1b.svg)](https://arxiv.org/abs/2603.02258)
[![Interactive Companion](https://img.shields.io/badge/Interactive_Companion-GitHub_Pages-2ea44f.svg)](https://kylemath.github.io/InterpretCognates/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v2.0.0-purple.svg)](https://github.com/kylemath/InterpretCognates/releases/tag/v2.0.0)

**Interactive web platform, 135-language embedding norm suite, and deterministic statistical verification pipeline for probing how multilingual neural translation networks represent universal conceptual structure across languages.**

- **Live Interactive Companion:** [https://kylemath.github.io/InterpretCognates/](https://kylemath.github.io/InterpretCognates/) (22 synchronized 2D/3D Plotly charts, phylogenetic trees, colexification graphs, and layerwise emergence explorers).
- **Preprint (arXiv v2):** [Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry](https://arxiv.org/abs/2603.02258) ([`paper/main.pdf`](paper/main.pdf); see [`paper/REPRODUCING.md`](paper/REPRODUCING.md) to build from source).
- **Journal Submission:** Submitted to *Behavior Research Methods* (Manuscript ID: `BR-Org-26-1447`).
- **Companion Software Paper (*JOSS*):** [`joss/paper.md`](joss/paper.md).
- **[Screenshot](screenshot.png)** — App overview.

---

## Key Empirical & Methodological Highlights

1. **Additive Language-Shift + Conceptual-Manifold Geometry**: Two-way Concept $\times$ Language variance partitioning and eigenvalue Participation Ratio (PR) dimensionality analysis show that NLLB-200 separates low-dimensional per-language shifts ($\text{PR} = 13.3$, capturing $45.7\%$ of native bare-word variance vs. $13.7\%$ in carrier frames) from an orthogonal, nearly $3\times$ higher-dimensional conceptual manifold ($\text{PR} = 38.7$). Subtracting per-language centroids increases the Conceptual Store Metric (CSM) by **$2.14\times$ on native bare words** ($1.15 \to 2.46$, vs. a $0.99$ permutation null; **$2.27\times$ across disjoint scripts**) and **$1.19\times$ in carrier frames** ($2.25 \to 2.69$).
2. **Layerwise Emergence & POS-Selective Terminal Unpacking**: Tracing representations across Layers $0\dots 11$ reveals a sigmoid phase transition (inflection at Layer 6, peaking at Layer 10 with $4.35\times$ mutual $k$-NN alignment gain), followed by POS-selective cone expansion at Layer 11 driven by pronouns and verbs preparing for decoder cross-attention.
3. **Semantic Dominance & Zipfian Subword Mediation**: Semantic category explains $\eta^2 = 25.4\%$ of cross-lingual convergence variance ($p = 2.21 \times 10^{-4}$)—over $20\times$ orthographic or phonological form ($R^2 \le 0.012$)—while Swadesh noun convergence over non-Swadesh nouns ($d = 0.54\text{--}0.69$) is mechanistically mediated by Zipfian subword compactness ($p = 7.43 \times 10^{-7}$).
4. **Phylogenetic (ASJP) & Lexical Typology ($\text{CLICS}^3$) Benchmarks**: Embedding distances track ASJP phonetic distances within subfamilies ($\rho_s = 0.65$, Jäger's LDND $\rho_s = 0.67$) and recurrent cross-family $\text{CLICS}^3$ colexifications ($d = 1.43, p = 1.55 \times 10^{-33}$).
5. **Psycholinguistic Parallels**: Basic color terms recover the Berlin & Kay evolutionary hierarchy ($\rho_s = 0.91\text{--}0.93$), and high- vs. lower-resource training tiers exhibit asymmetric conceptual convergence ($p \le 3.63 \times 10^{-9}$) analogous to Kroll & Stewart's (1994) Revised Hierarchical Model.
6. **Zero-Hardcoded-Numbers Reproducibility**: All 321 reported statistics in [`paper/main.pdf`](paper/main.pdf) are programmatically generated from `docs/data/*.json` into [`paper/output/stats.tex`](paper/output/stats.tex) via [`paper/scripts/generate_stats.py`](paper/scripts/generate_stats.py).

---

## Citation

If you use `InterpretCognates`, its interactive web explorer, or its 135-language geometric norms, please cite:

```bibtex
@article{mathewson2026universal,
  title={Universal Conceptual Structure in Neural Translation: Probing {NLLB-200}'s Multilingual Geometry},
  author={Mathewson, Kyle E.},
  journal={arXiv preprint arXiv:2603.02258},
  year={2026},
  url={https://arxiv.org/abs/2603.02258},
  note={Submitted to Behavior Research Methods (BR-Org-26-1447)}
}
```

---

## Model Choice

This toolkit uses **Meta NLLB-200** (open weights) through Hugging Face Transformers:

- Practical default in code: `facebook/nllb-200-distilled-600M` (fast local iteration).
- Midpoint quality option: `facebook/nllb-200-distilled-1.3B`.
- Highest quality option: `facebook/nllb-200-3.3B` (requires larger GPU/CPU RAM).

Set the model with:

```bash
export NLLB_MODEL=facebook/nllb-200-distilled-1.3B
```

## Run the Live Probing Backend Locally

From repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Rebuild the Paper, Figures, and 321 Statistical Macros

```bash
cd paper
python3 scripts/generate_stats.py
python3 scripts/generate_figures.py
bash build.sh
bash build_arxiv.sh
```

## Build & Deploy the Interactive Static Companion (`docs/`)

The interactive research companion runs entirely client-side from pre-computed JSON data in `docs/data/`:

```bash
python3 build_static.py          # outputs to docs/
python3 build_static.py dist     # or specify a custom output directory
```

## License

Released under the [MIT License](LICENSE).
