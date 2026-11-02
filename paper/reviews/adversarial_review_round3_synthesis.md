# Integrative Synthesis & Experimental Control Plan (Round 3 Journal Reviews)

**Sources Synthesized:**
1. **Reviewer 1 (Friendly, Constructive Senior Reviewer — *Nature Human Behaviour* / *TACL*)**: [`journal_reviewer_1_friendly_constructive.md`](file:///Users/kylemathewson/InterpretCognates/paper/reviews/journal_reviewer_1_friendly_constructive.md)
2. **Reviewer 2 (Adversarial, Skeptical Methodological Reviewer — *TACL* / *NeurIPS* / *Cognition*)**: [`journal_reviewer_2_adversarial_skeptical.md`](file:///Users/kylemathewson/InterpretCognates/paper/reviews/journal_reviewer_2_adversarial_skeptical.md)

---

## Overview of Core Methodological & Scientific Upgrades

To ensure that every claim in the manuscript is bulletproof against simpler, non-semantic alternative explanations (surface form, script sharing, subword tokenization fertility, Zipfian word length, carrier-frame syntax, global anisotropy, and stimulus string identity), we implement and integrate **six comprehensive sets of control experiments and mathematical/code fixes**:

### 1. Analysis 1 (Conceptual Store Metric & Additive Decomposition $\mathbf{e}_{c,\ell} = \bar{\mathbf{e}} + \boldsymbol{\mu}_c + \boldsymbol{\ell} + \boldsymbol{\epsilon}_{c,\ell}$)
- **Alternative Explanations Tested**:
  1. *Global-mean / cosine-normalization artifact*: Subtracting any mean vector removes the narrow anisotropic cone so random centroids have expected cosine distance $\approx 1.01$ ($\text{CSM}_{\text{null}} \approx 1.01$).
  2. *Shared-script / cognate inflation*: $92/135$ languages use the Latin script, where cognates share SentencePiece subwords.
- **Controls & Fixes Integrated**:
  - **Zero-sum identifiability**: Explicitly define $\mathbf{e}_{c,\ell} = \bar{\mathbf{e}} + \boldsymbol{\mu}_c + \boldsymbol{\ell} + \boldsymbol{\epsilon}_{c,\ell}$ with $\sum_c \boldsymbol{\mu}_c = \mathbf{0}$ and $\sum_\ell \boldsymbol{\ell} = \mathbf{0}$, and report the **two-way sum-of-squares variance partition** ($\eta^2_{\text{lang}}$, $\eta^2_{\text{concept}}$, $\eta^2_{\text{resid}}$) in both Bare-Word and Carrier-Frame regimes.
  - **Concept-label permutation null** ($1,000$ permutations of concept labels within each language, $\text{CSM}_{\text{null}} = 1.01$) and **global mean-centering baseline** ($\mathbf{e}_{c,\ell} - \bar{\mathbf{e}}$), proving that per-language mean-centering specifically collapses *within-concept* cross-lingual dispersion far beyond global centering or the $1.01$ random null ($2.46$ on bare words, $2.69$ in carrier frame; $p < 0.0001$).
  - **Strict Cross-Script CSM** ($\text{script}(\ell) \neq \text{script}(\ell')$, zero orthographic or subword overlap possible) on both bare words and carrier frames.
  - **Unbiased bootstrap CIs**: Exclude `idx[a] == idx[b]` self-pairs in `_bootstrap_ratio`.
  - **Layer indexing & subword extraction precision**: Clarify that Layers $0\text{--}11$ index Transformer Encoder Blocks $1\text{--}12$ (`hidden_states[1:13]`) and report the exact target-word subword span match rate across the $13,635$ concept–language pairs.

### 2. Analysis 2 & 3 (Swadesh Convergence Hierarchy, Semantic Category $\eta^2$, Polysemy, and Word-Length / Subword-Fertility Controls)
- **Alternative Explanations Tested**:
  1. *Surface form vs. semantic category*: How much variance in Swadesh convergence is explained by semantic category vs. orthography, phonology, word length, subword fertility, and polysemy?
  2. *Zipfian word length / subword fertility in Swadesh vs. non-Swadesh nouns*: Core Swadesh nouns are shorter ($4.19$ vs. $4.85$ chars) and have lower SentencePiece subword fertility than non-Swadesh nouns, and `non_swadesh_controlled.json` included 8 abstract/homograph items (*anger, fear, hunger, thirst, dream, harvest, shadow, well*).
- **Controls & Fixes Integrated**:
  - **8-Category ANOVA & Kruskal-Wallis** ($F(7, 93) = 4.53, p = 2.21 \times 10^{-4}, \eta^2 = 25.4\%$; Kruskal-Wallis $H = 28.00, p = 2.20 \times 10^{-4}$), showing that semantic category explains $>20\times$ the variance of orthographic ($1.2\%$) or phonological ($0.0\%$) similarity—and remains highly significant in an ANCOVA controlling simultaneously for word length, SentencePiece subword fertility, orthographic similarity, and CLICS³ polysemy degree.
  - **Strictly Concrete Non-Swadesh Noun Control ($n=52$ vs. $n=52$) & Zipfian Length/Fertility Mediation Analysis**: Report (a) the POS-matched concrete noun comparison ($p = 0.004$ raw, $p = 8.4 \times 10^{-4}$ ABTT), (b) the strictly concrete $52$-vs-$52$ noun comparison (removing *anger, fear, hunger, thirst, dream, harvest, shadow, well*), and (c) the multiple regression / ANCOVA controlling for cross-lingual character length and SentencePiece subword fertility on both raw and ABTT-corrected embeddings (as well as bare words), transparently showing how Zipf's Law of Abbreviation (shorter, lower-fertility roots for core concepts) partially mediates raw cosine convergence while ABTT-corrected conceptual convergence remains robust.

### 3. Analysis 4 (Phylogenetic ASJP Mantel Analysis — Block-Permutation $p$-Values & Cross-Script Controls)
- **Alternative Explanations Tested**:
  1. *Non-independent dyadic $p$-value*: Treating 321 pairwise distances as i.i.d. inflates naive `spearmanr` significance.
  2. *Shared script confound*: $83.5\%$ ($268/321$) of within-subfamily language pairs share the same writing system.
- **Controls & Fixes Integrated**:
  - **Within-subfamily block-permutation Mantel test** ($9,999$ permutations of languages within each subfamily, $p = 1.0 \times 10^{-4}$) replacing naive i.i.d. dyadic $p$-values.
  - **Cross-script within-subfamily pairs ($n = 53$)** (e.g., Cyrillic vs. Latin Slavic, Devanagari vs. Arabic/Cyrillic Indo-Iranian, Latin vs. Cyrillic Turkic), showing $\rho_s = 0.57$ ($p < 0.0001$) even with zero shared script, alongside same-script within-subfamily pairs ($n = 268, \rho_s = 0.69$).
  - **Partial Mantel test controlling for script identity ($S_{\ell,\ell'} \in \{0,1\}$)** and **stratified bare-word phylogenetic correlations**.

### 4. Analysis 5 (CLICS³ Colexification — String-Identity Filtering, Within-Category & Same-POS Controls, and Bare-Word Verification)
- **Alternative Explanations Tested**:
  1. *Literal string identity in the stimulus table*: When a language in `swadesh_corpus.json` translates two concepts with the exact same word ($w_{a,\ell} = w_{b,\ell}$), within-language cosine similarity is trivially $1.0$.
  2. *Coarse semantic category / POS imbalance*: Colexified pairs are enriched for same-category ($45.8\%$ vs. $11.8\%$) and same-POS ($68.2\%$ vs. $30.4\%$) pairs.
- **Controls & Fixes Integrated**:
  - Harmonize the formula description in Sections 3.2 & 4.5 (mean within-language cosine similarity $\frac{1}{N}\sum_\ell \cos(\mathbf{e}_{a,\ell}, \mathbf{e}_{b,\ell})$ AND cross-lingual centroid similarity $\cos(\hat{\boldsymbol{\mu}}_a, \hat{\boldsymbol{\mu}}_b)$).
  - **String-identity & cognate-filtered control**: Exclude all `(concept_a, concept_b, lang)` triples where $w_{a,\ell} = w_{b,\ell}$ (or within-language normalized edit distance $< 0.2$).
  - **Within-category stratified control** ($n = 783$ same-category pairs: $U = 27{,}919, p = 2.50 \times 10^{-18}$, continuous $\rho_s = 0.32, p = 1.57 \times 10^{-20}$—nearly tripling the unstratified correlation!), **Same-POS control**, and **Decontextualized Bare-Word Colexification test**.

### 5. Analysis 6a (Berlin & Kay Color Hierarchy — Bare-Word Verification, `grey = 5.0` Fix, Unified 135 Languages, and Word-Length Control)
- **Alternative Explanations Tested**:
  1. *English carrier-frame ungrammaticality* (`"I saw a {color} near the river."`).
  2. *Code-vs.-text bug on `grey`* (`7.0` in code vs. `5.0` in text; circle vs. open diamond marker) and `136` vs. `135` languages (`hye_Armn`).
  3. *Zipfian word-length / subword-fertility covariance* across color stages.
- **Controls & Fixes Integrated**:
  - Fix `load_color_terms()` to filter `data.get("colors", {})` and unify the language set to the exact **135 Swadesh languages**.
  - Fix `bk_stage['grey'] = 5.0` (and report both midpoint $5.0$ [$r = 0.89, p = 2.8 \times 10^{-4}$] and upper-bound $7.0$ [$r = 0.82, p = 0.002$]) and plot `grey` as an open diamond (`marker='D'`) in Figure 10b (`fig_color_circle.pdf`).
  - Evaluate the Berlin & Kay hierarchy on **decontextualized bare words** (and language-centered bare words), report the **exact permutation $p$-value**, and report the **partial correlation controlling for word length / subword fertility**.

### 6. Analysis 6b (Semantic Offset Invariance — Doubly Centered Perturbation Coupling $\cos(\hat{\boldsymbol{\epsilon}}_{a,\ell}, \hat{\boldsymbol{\epsilon}}_{b,\ell})$, Same-POS Distance-Matched Controls, and Concave Saturation Baseline)
- **Alternative Explanations Tested**:
  1. *Unsubtracted language shift $\boldsymbol{\ell}$ in `perturbation_coupling`*: `eps_a = mat_a - mu_a` on uncentered embeddings left $\boldsymbol{\ell}$ inside $\boldsymbol{\epsilon}_{a,\ell}$.
  2. *Distance & POS mismatch in the 7 ad-hoc negative controls*: The 7 controls had larger centroid separation ($13.80$ vs. $10.18$) and mixed POS, and `("smoke", "leg")` was silently skipped.
  3. *Concave saturation of expected cosine similarity* $\mathbb{E}[S(a,b)] \approx D / \sqrt{D^2 + c}$ and in-sample centroid inclusion.
- **Controls & Fixes Integrated**:
  - Subtract per-language mean vectors $\hat{\boldsymbol{\ell}}$ before computing perturbation residuals ($\hat{\boldsymbol{\epsilon}}_{c,\ell} = \mathbf{e}_{c,\ell} - \hat{\boldsymbol{\mu}}_c - \hat{\boldsymbol{\ell}} + \bar{\mathbf{e}}$) so `perturbation_coupling` measures true concept-pair residual coupling.
  - Evaluate against both (a) **same-POS, distance-matched never-colexified negative control pairs** and (b) **all never-colexified same-POS pairs across the Swadesh matrix**, testing both linear and theoretical concave ($D/\sqrt{D^2 + c}$) distance-controlled residuals, leave-one-language-out offset consistency, and bare-word offset consistency.
