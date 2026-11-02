---
title: 'InterpretCognates: An Interactive Toolkit and 135-Language Probing Suite for Multilingual Representation Geometry'
tags:
  - Python
  - JavaScript
  - psycholinguistics
  - multilingual NLP
  - neural machine translation
  - lexical typology
  - representation geometry
authors:
  - name: Kyle E. Mathewson
    orcid: 0000-0003-4856-3980
    affiliation: "1, 2"
affiliations:
 - name: Department of Psychology, Faculty of Science, University of Alberta, Edmonton, AB, Canada
   index: 1
 - name: Neuroscience and Mental Health Institute, Faculty of Medicine and Dentistry, University of Alberta, Edmonton, AB, Canada
   index: 2
date: 02 November 2026
bibliography: paper.bib
---

# Summary

Massively multilingual neural machine translation (NMT) and language models map hundreds of human languages into shared high-dimensional vector spaces [@costa2022nllb; @chang2022geometry]. For researchers in psycholinguistics, bilingual cognition, and quantitative lexical typology, these internal representations offer a powerful computational testbed for investigating how lexical concepts are organized across languages, scripts, and language families [@kroll1994category; @list2018clics; @jaeger2018global]. However, extracting, de-confounding, and visualizing cross-linguistic representation geometry across over a hundred languages typically requires substantial custom engineering to separate surface-form and language-identity artifacts from underlying semantic structure.

`InterpretCognates` is an open-source Python and browser-based software toolkit designed to make multilingual representation probing accessible, interactive, and deterministically reproducible. Built around Meta's open-weights `NLLB-200` translation model [@costa2022nllb], `InterpretCognates` provides:

1. **A FastAPI & PyTorch Probing Backend (`backend/`)**: Extracts token- and sentence-level hidden states and cross-attention weights across any subset of NLLB-200's 200+ supported languages, supporting both contextualized carrier-frame inputs and decontextualized native bare-word tokens.
2. **A Synchronized Interactive Web Explorer (`docs/`)**: A static, zero-server browser application (hosted at <https://kylemath.github.io/InterpretCognates/>) featuring 22 interactive 2D and 3D Plotly visualizations synchronized one-to-one with our empirical study [@mathewson2026universal]. Users can inspect 135-language concept manifolds, toggle mean-centering and All-but-the-Top (ABTT) anisotropy removal [@mu2018all], explore phylogenetic trees against Automated Similarity Judgment Program (ASJP) phonetic distances [@wichmann2022asjp; @jaeger2018global], and query cross-linguistic colexification networks against $\text{CLICS}^3$ [@rzymski2020database].
3. **A Deterministic Statistical-to-LaTeX Pipeline (`paper/scripts/generate_stats.py`)**: An automated verification workflow that computes 321 statistical tests, effect sizes, null permutations, and confound controls directly from the pre-computed JSON data artifacts and compiles them into LaTeX macros (`paper/output/stats.tex`), guaranteeing zero copy-paste discrepancies between code, interactive web charts, and published prose.

# Statement of Need

Researchers studying the bilingual mental lexicon [@kroll1994category; @kroll2010bilingual], cortical language representations [@malikmoraleda2022universal; @deniz2025tuning], and cross-linguistic lexical semantics [@vulic2020probing] increasingly use neural language model embeddings to generate quantitative semantic norms and test cognitive hypotheses. Yet raw cosine similarities in transformer encoders are heavily confounded by representation anisotropy [@ethayarajh2019contextual; @mu2018all], subword tokenization fertility [@rust2021good], orthographic script sharing, and low-dimensional language-identity shifts [@chang2022geometry; @pires2019multilingual].

Existing interpretability packages generally focus on monolingual token attribution or task-level benchmarking rather than typologically controlled psycholinguistic probing. `InterpretCognates` fills this gap by providing:

- **Pre-computed 135-Language Geometric Norms**: Ready-to-query JSON datasets covering 101 Swadesh core concepts [@swadesh1952lexico], 60 non-Swadesh concrete nouns, basic color terms [@berlin1969basic], and semantic offset pairs across 135 languages (45 language families, 24 scripts) in both contextualized carrier frames and native bare-word forms.
- **Built-in Confound Controls**: Turnkey implementations of two-way Concept $\times$ Language ANOVA variance partitioning, eigenvalue Participation Ratio (PR) dimensionality estimation, ABTT principal component removal ($D = 1\dots 10$), orthographic/phonological Levenshtein regression, and Jäger's [@jaeger2018global] off-diagonal normalized Levenshtein distance (LDND).
- **Interactive Pedagogical and Exploratory Workflows**: Psycholinguists and students can explore 3D conceptual manifolds, compare carrier-frame versus bare-word convergence, and inspect individual language families in real time in any modern web browser without requiring local GPU hardware.

# Software Architecture and Reproducible Pipeline

`InterpretCognates` is organized into three modular layers (\autoref{fig:arch}):

1. **Extraction & Analysis Layer (`backend/app/main.py`, `research/`)**: Wraps Hugging Face `transformers` (`AutoModelForSeq2SeqLM`, `AutoTokenizer`) to run multi-target translation, extract layerwise encoder hidden states (Layers $0\dots 11$), compute attention-weighted pooled representations, and project embeddings via Principal Component Analysis (PCA) and Multidimensional Scaling (MDS).
2. **Interactive Visualization Layer (`docs/index.html`, `docs/blog.js`, `build_static.py`)**: Renders pre-computed JSON artifacts (`docs/data/*.json`) into interactive Plotly charts, phylogenetic dendrograms, layerwise emergence curves, and live translation-similarity explorers. Running `python3 build_static.py` packages the entire interactive suite into a standalone static site (`docs/`).
3. **Programmatic Manuscript Synchronization (`paper/scripts/generate_stats.py`, `paper/scripts/generate_figures.py`)**: Reads the exact JSON files served to the web application to generate all 16 vector PDF figures (`paper/figures/*.pdf`) and all 321 LaTeX `\newcommand` statistical macros (`paper/output/stats.tex`).

![System architecture of the `InterpretCognates` probing toolkit, interactive web companion, and deterministic LaTeX macro pipeline.\label{fig:arch}](../paper/figures/fig_system_architecture.pdf)

# Availability

`InterpretCognates` is released under the MIT License at <https://github.com/kylemath/InterpretCognates> and archived on Zenodo at [`10.5281/zenodo.23898624`](https://doi.org/10.5281/zenodo.23898624), with the live interactive companion hosted at <https://kylemath.github.io/InterpretCognates/> and the companion empirical manuscript available at `arXiv:2603.02258` [@mathewson2026universal].

# Acknowledgements

This research was supported by the Department of Psychology and the Neuroscience and Mental Health Institute at the University of Alberta.

# References
