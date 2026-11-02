# Peer Review: *Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*

**Reviewers:** Prof. Benjamin K. Bergen & Dr. Tyler A. Chang (Department of Cognitive Science, UC San Diego; authors of Chang, Tu, & Bergen, 2022, *"The Geometry of Multilingual Language Model Representations"*, EMNLP)  
**Overall Recommendation:** **Accept with Minor, High-Leverage Revisions** (Strong Accept; exemplary methodological rigor and cross-disciplinary synthesis)

---

## 1. Executive Summary & Relation to Chang, Tu, & Bergen (2022)

This manuscript presents a remarkably thorough, geometrically principled investigation of how a massively multilingual neural machine translation encoder (`facebook/nllb-200-distilled-600M`, $d = 1024$, 12 encoder layers) organizes lexical meaning across 135 typologically diverse languages and 101 core Swadesh concepts. What distinguishes this paper from the first wave of multilingual probing studies is that every substantive claim—from the global two-way variance partition ($\mathbf{e}_{c,\ell} = \bar{\mathbf{e}} + \hat{\boldsymbol{\mu}}_c + \hat{\boldsymbol{\ell}} + \hat{\boldsymbol{\epsilon}}_{c,\ell}$) to the Swadesh category hierarchy, ASJP historical phylogeny, CLICS³ universal colexification, Berlin & Kay (1969) evolutionary color stages, and second-order relational offsets—is paired with targeted geometric, orthographic, tokenization, and permutation null controls across both contextualized (carrier-frame) and decontextualized (native bare-word) regimes.

Mathewson (2026) advances our EMNLP 2022 research program (Chang, Tu, & Bergen, 2022) in four crucial ways:
1. **Self-Supervised Masked LM (`XLM-R`, 88 languages) vs. Many-to-Many Encoder-Decoder NMT (`NLLB-200`, 135 languages):** Demonstrates that even under explicit 200-language translation supervision, NLLB-200 does not collapse native words into a language-amnesic interlingua; instead, it preserves an exact additive decomposition $\mathbf{e}_{c,\ell} = \bar{\mathbf{e}} + \hat{\boldsymbol{\mu}}_c + \hat{\boldsymbol{\ell}} + \hat{\boldsymbol{\epsilon}}_{c,\ell}$.
2. **Quantifying How Shared Context Homogenizes Language Shifts (`25.7%` $\to$ `3.8%` SS):** Resolves why prior studies reported conflicting degrees of language neutrality: bare words have $25.7\%$ language variance and $44.7\%$ concept variance ($\text{CSM}_{\text{raw}} = 1.15$), whereas shared context shrinks language variance nearly sevenfold to $3.8\%$ and expands concept variance to $64.4\%$ ($\text{CSM}_{\text{raw}} = 2.25$), while per-language mean-centering aligns both onto the same conceptual store ($2.46$ and $2.69$).
3. **ABTT $k=0 \to 1$ Knee & Zipfian Subword Fertility Mediation:** Shows that removing a single principal component ($k=1$) drops mean convergence from $0.72$ ($k=0$) to $0.59$ ($k=1$) and plateaus through $k=3$ ($0.58$), $k=5$ ($0.58$), and $k=10$ ($0.56$) while preserving rank order ($\rho_s = 0.990$), and proves via ANCOVA that Zipfian subword compactness ($1.73$ vs. $2.21$ tokens/word, $t = -5.26, p = 7.43 \times 10^{-7}$) mediates the between-list Swadesh concrete noun advantage.

---

## 2. High-Leverage Refinements (Conceptual & Fast Empirical Additions)

### Refinement 1: Quantify the Effective Dimensionality (Participation Ratio) of the Language Subspace $\{\hat{\boldsymbol{\ell}}\}$ vs. Concept Subspace $\{\hat{\boldsymbol{\mu}}_c\}$
From classical doubly-centered Gram matrix eigendecomposition ($B = -\frac{1}{2} H D^{\circ 2} H$) and the Participation Ratio $\text{PR} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$ on `docs/data/phylogenetic.json` and `docs/data/colexification.json`:
- **Bare-Word Language Subspace ($N = 88$ languages):** Top 10 PCs explain $66.6\%$ of language-pair variance; Effective Dimensionality $\text{PR}_{\text{lang, bare}} = \mathbf{13.3}$ ($90\%$ variance in $d_{90} = 30$ dimensions).
- **Carrier-Frame Language Subspace ($N = 135$ languages):** Top 10 PCs explain $51.3\%$ of variance; Effective Dimensionality $\text{PR}_{\text{lang, ctx}} = \mathbf{18.8}$.
- **Concept Subspace ($M = 101$ concepts):** Top 10 PCs explain only $40.1\%$ of inter-concept variance; Effective Dimensionality $\text{PR}_{\text{concept}} = \mathbf{38.7}$ ($90\%$ variance requires $d_{90} = 57$ out of $100$ dimensions)—**nearly $3\times$ the effective dimensionality of the bare-word language space!**
Reporting this directly generalizes Chang, Tu, & Bergen's (2022) subspace-dimensionality asymmetry from masked language models to 200-language NMT.

### Refinement 2: Layer-Wise Emergence of the Berlin & Kay Color Hierarchy (`docs/data/color_circle_layers.json`)
Even at Layer 0, PC1 correlates with Berlin & Kay stage at $|\rho_s| = 0.81$ ($p = 0.0046$), and self-attention across Layers 5–11 sharpens this to $|\rho_s| = 0.91\text{--}0.96$ (peaking at Layer 7 at $|\rho_s| = 0.96, p = 7.7 \times 10^{-6}$ and finishing at Layer 11 at $0.93$).

### Refinement 3: Sharpen the Cognitive Framing of "Distributional Sedimentation" vs. "Perceptual Grounding"
Explicitly distinguish (1) *cross-linguistic extensional equivalence* in parallel translation pairs for concrete nouns versus $1\text{-to-many}$ conditional mappings for verbs and grammatical deictics, and (2) *typological asymmetric differentiation* across the Berlin & Kay hierarchy.
