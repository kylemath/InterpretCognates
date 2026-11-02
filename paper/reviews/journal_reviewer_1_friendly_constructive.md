# Peer Review: *Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*

**Reviewer Role**: Reviewer 1 (Senior Expert in Multilingual NLP Interpretability, Lexical Typology, and Cognitive Science — Friendly & Constructive)  
**Target Venue**: *Nature Human Behaviour* / *Transactions of the Association for Computational Linguistics (TACL)*  
**Recommendation**: **Accept with Minor-to-Moderate Revisions** (Outstanding interdisciplinary contribution; straightforward empirical controls and code-to-text harmonizations will elevate this into a landmark paper).

---

## 1. Executive Summary & Positive Assessment

This manuscript presents an ambitious, methodologically rigorous, and genuinely interdisciplinary investigation of the encoder representation space of Meta's 600M-parameter distilled NLLB-200 translation model (`facebook/nllb-200-distilled-600M`) across 135 typologically diverse languages and 101 Swadesh core concepts. To my knowledge, this is the first study to systematically bridge **mechanistic interpretability of a many-to-many neural translation encoder** with **four independent ground-truth benchmarks from historical linguistics, lexical typology, and cognitive science**:
1. Psycholinguistic/neurobiological models of the bilingual lexicon (BIA+, RHM, and the cortical language-tuning modulation architecture of Chen et al. / Deniz 2025), formalized via the additive decomposition $\mathbf{e}_{c,\ell} = \boldsymbol{\mu}_c + \boldsymbol{\ell} + \boldsymbol{\epsilon}_{c,\ell}$;
2. Historical-comparative lexicostatistics via ASJP v20 phonetic distances (Wichmann et al., 2022; Jäger, 2018);
3. Cross-linguistic polysemy and conceptual adjacency via the CLICS³ colexification database across all $\binom{101}{2} = 5{,}050$ concept pairs (Rzymski et al., 2020); and
4. Perceptual and evolutionary domain universals via the Berlin & Kay (1969) basic color hierarchy and distance-controlled relational offset vectors.

### What This Manuscript Currently Does Exceptionally Well
- **Methodological self-awareness and strong internal controls**: Rather than taking high cosine similarities at face value, the manuscript proactively confronts anisotropy (via ABTT $k$-sensitivity sweeps), contextual template homogenization (by pairing the shared English carrier frame against a decontextualized native bare-word regime), international loanword inflation (by contrasting loanword-heavy non-Swadesh items against non-loanword, POS-matched concrete nouns), ASJP lexical distance saturation at deep time depths (by stratifying pairs into within-subfamily, cross-branch IE, and cross-family tiers), and the mechanical norm-dependence of vector offsets (by regressing $S(a,b)$ on $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|_2$ against negative controls).
- **Reproducibility and transparency**: The automated pipeline (`precompute_paper_data.py` $\to$ `generate_stats.py` $\to$ `output/stats.tex` + `generate_figures.py` + `docs/index.html`) is exemplary. Every statistic in the manuscript is linked programmatically to real activations without hardcoding.
- **Compelling central narrative**: The finding that bare native words and contextualized words converge onto nearly the same conceptual store ratio once per-language mean shifts $\hat{\boldsymbol{\ell}}$ are removed ($\text{CSM}: 1.15 \to 2.46$ [$2.14\times$] on bare words vs. $2.25 \to 2.69$ [$1.19\times$] in the carrier frame) provides a crisp, memorable answer to the opening question of the paper.

Below, I offer **six high-leverage scientific/empirical enhancements** (several of which can be computed in seconds directly from the existing JSON artifacts in `docs/data/`, and the rest with minor additions to `precompute_paper_data.py` / `generate_stats.py`), followed by **Whitesides-style structural and rhetorical improvements** and **line-by-line technical polish** (including a few subtle discrepancies between the text and the Python scripts that are easy to fix).

---

## 2. Constructive Major Suggestions (High-Leverage Scientific & Empirical Enhancements)

### Major Suggestion 1: Control for Coarse Semantic Category in the CLICS³ Colexification Analysis — and Reconcile Within-Language vs. Centroid Similarity (Section 4.5)
- **The Scientific Issue**: In Section 4.5 (`04-results.tex`, lines 191–204), the manuscript compares frequently colexified Swadesh pairs ($\ge 3$ families, $n = 201$, $\mu = 0.37$) against all never-colexified pairs across the entire $101 \times 101$ matrix ($0$ families, $n = 3{,}258$, $\mu = 0.29$; $U = 492{,}477$, $p = 1.55 \times 10^{-33}$, $d = 1.43$; continuous $\rho_s = 0.13$). A sharp lexical typologist will immediately ask: *"Colexified pairs almost always belong to the same or adjacent semantic domains (e.g., two body parts, two actions, two natural phenomena), whereas the 3,258 never-colexified pairs are dominated by cross-category pairs (e.g., a pronoun and a body part). Is the CLICS³ test merely rediscovering coarse semantic categories rather than fine-grained colexification?"*
- **What the Existing Data Already Show (Fast to Add!)**: Inspecting `docs/data/colexification.json` stratified by the 8 Swadesh semantic categories reveals an even stronger result that directly answers this critique:
  1. **Within the exact same semantic category** ($n = 783$ within-category pairs, e.g., body-part vs. body-part, action vs. action): frequently colexified pairs ($\ge 3$ families, $n = 92$) exhibit a mean similarity of **$0.425$** vs. **$0.331$** for never-colexified pairs in the *same* category ($n = 384$; Mann-Whitney $U = 27{,}919$, $p = 2.50 \times 10^{-18}$), and the continuous Spearman rank correlation **nearly triples from $\rho_s = 0.13$ to $\rho_s = 0.32$ ($p = 1.57 \times 10^{-20}$)**!
  2. **Across different semantic categories** ($n = 4{,}267$ cross-category pairs, e.g., *bark–skin* or *earth–mountain*): frequently colexified pairs ($\ge 3$ families, $n = 109$) *still* show significantly higher similarity than never-colexified cross-category pairs ($0.321$ vs. $0.283$; $U = 201{,}303$, $p = 2.08 \times 10^{-7}$).
- **Code-to-Text Precision Fix**: In `backend/app/benchmarks.py` (lines 689–701, `_pair_similarity`), the code actually computes the **mean within-language cosine similarity** $\frac{1}{N}\sum_{\ell=1}^N \cos(\mathbf{e}_{a,\ell}, \mathbf{e}_{b,\ell})$, whereas `04-results.tex` (lines 191, 197, 201) describes it as *"cross-lingual centroid cosine similarity"* between $\boldsymbol{\mu}_c$ and $\boldsymbol{\mu}_{c'}$. Furthermore, `precompute_paper_data.py` (line 810) passes `sw_emb_ctx` (carrier-frame embeddings).
  - **Recommendation**: Clarify the exact definition in Section 3.2/4.5, report the **within-category stratified test** ($\rho_s = 0.32$, $p = 1.57 \times 10^{-20}$), and add a one-line cross-check showing that the CLICS³ colexification separation also holds on **decontextualized bare-word embeddings** (`dectx["_dectx_raw_embeddings"]` or language-centered bare words), ruling out any possibility that English carrier-frame syntax drives the colexification clustering.

### Major Suggestion 2: Quantify Semantic Category Variance ($\eta^2$) and Polysemy Covariates to Contrast with Surface-Form $R^2$ (Sections 4.2 & 4.3)
- **The Scientific Issue**: Section 4.3 (`04-results.tex`, lines 140–143) shows that orthographic similarity ($R^2 = 0.012$, $p = 0.272$) and phonological similarity ($R^2 = 0.000$, $p = 0.929$) explain $<1.3\%$ of the variance in Swadesh convergence, while Section 4.2 (lines 94–103) presents the 8-category semantic hierarchy and notes qualitatively that within-category outliers (*bark, lie, fly, louse*) reflect English lexical polysemy or low frequency. However, the manuscript never reports the formal statistical test or variance explained ($\eta^2$ / $R^2$) by **semantic category** itself!
- **What the Existing Data Already Show**:
  - Running a one-way ANOVA (and non-parametric Kruskal-Wallis test) on `swadesh_convergence.json` across the 8 semantic categories yields **$F(7, 93) = 4.53$, $p = 2.21 \times 10^{-4}$, $\eta^2 = 0.254$ (25.4% of total variance)**, and **Kruskal-Wallis $H = 28.00$, $p = 2.20 \times 10^{-4}$**.
  - Reporting this in Section 4.2/4.3 creates a striking quantitative contrast: **semantic category alone accounts for $25.4\%$ of the variance in cross-lingual convergence—more than $20\times$ the variance explained by orthographic ($1.2\%$) or phonological ($0.0\%$) similarity**.
  - In addition, `docs/index.html` (lines 348–350) mentions controlling for polysemy via WordNet sense counts (`@article{miller1995}` is in `references.bib` but currently uncited in the paper), and CLICS³ already provides the colexification degree of each concept. Adding a Spearman correlation or multiple regression showing how polysemy (e.g., WordNet synset count or CLICS³ concept degree) and subword token length (fertility) relate to $C(c)$ will turn the qualitative discussion of *bark, lie, fly* in Section 4.2 into a rigorous quantitative test.

### Major Suggestion 3: Rule Out Script-Sharing Confounds in the Phylogenetic Analysis and Report Stratified Bare-Word Correlations (Section 4.4)
- **The Scientific Issue**: In Section 4.4 (`04-results.tex`, lines 179–186), embedding distance correlates strongly with ASJP phonetic distance within language subfamilies ($\rho_s = 0.65$, $p = 2.40 \times 10^{-40}$, $n = 321$ pairs), while the global Mantel correlation doubles from $\rho_s = 0.13$ (carrier frame) to $\rho_s = 0.24$ (bare words). A reviewer in computational typology will ask two questions:
  1. *"Within-subfamily pairs (e.g., Romance, Germanic, Bantu) predominantly share the same writing system (Latin script), whereas cross-family pairs often use different scripts. Is the phylogenetic correlation driven by shared script rather than phonetic/genealogical distance?"*
  2. *"What is the within-subfamily correlation on decontextualized bare words (currently only reported for the carrier frame)?"*
- **What the Existing Data Already Show**:
  - Using the existing matrices in `docs/data/phylogenetic.json`, when we restrict strictly to **same-script language pairs** (eliminating any cross-script orthography difference), the global Spearman correlation increases from $\rho_s = 0.13$ to **$\rho_s = 0.21$ ($p = 3.05 \times 10^{-20}$, $n = 1{,}973$ pairs)**, and the **within-subfamily correlation restricted to same-script pairs increases from $\rho_s = 0.65$ to $\rho_s = 0.69$ ($p = 7.87 \times 10^{-39}$, $n = 268$ pairs)**! (Indeed, the 53 within-subfamily pairs that *differ* in script—such as Serbian [Cyrillic] vs. Croatian [Latin], or Hindi [Devanagari] vs. Urdu [Arabic]—actually attenuate the raw subfamily correlation because script divergence inflates embedding distance relative to phonetic distance).
  - **Recommendation**: Report the **same-script restricted** (or partial Mantel controlling for script identity) correlations in Section 4.4, and also save `emb_subset_bare` in `phylogenetic.json` (`precompute_paper_data.py`, line 765) so `generate_stats.py` can report the stratified within-subfamily $\rho_s$ for bare words alongside the global bare-word Mantel $\rho_s = 0.24$.

### Major Suggestion 4: Harmonize the Berlin & Kay `grey` Stage Coding and Verify Bare-Word / Mean-Centered Color Hierarchy (Section 4.6)
- **The Scientific Issue & Code-to-Text Mismatch**:
  - In `04-results.tex` (line 217 and line 223), the manuscript states:
    - Line 217 (Fig. 10 caption): *"with `grey` shown as an open diamond representing its wildcard Stage III–VII status"*.
    - Line 223: *"Across the 10 stage-ordered terms, PC1 correlates with Berlin & Kay evolutionary stage at Spearman $\rho_s = 0.93$ ($p = 8.48 \times 10^{-5}$), and across all 11 terms (assigning `grey` its midpoint stage $5.0$) at Pearson $r = 0.82$ ($p = 0.002$)."*
  - However, in `paper/scripts/generate_stats.py` (line 302) and `paper/scripts/generate_figures.py` (line 764), the dictionary actually sets `'grey': 7.0` (NOT `5.0`!), and `generate_figures.py` (line 793) plots `grey` as a standard circle marker at `7.0 - 0.22 = 6.78` rather than an open diamond at Stage $5.0$!
  - **What Happens When You Fix `'grey': 5.0`**: When `grey` is assigned its true Stage III–VII midpoint of `5.0`, the 11-term correlation is **even stronger**: **Pearson $r = 0.89$ ($p = 2.83 \times 10^{-4}$)** and **Spearman $\rho_s = 0.86$ ($p = 6.08 \times 10^{-4}$)** (compared to $r = 0.82$, $p = 0.002$ when `grey = 7.0`).
  - **Additional Language Count Discrepancy**: `ColorNumLanguages` is `136`, whereas `NumLanguages` for Swadesh is `135`. Why? Because `backend/app/data/color_terms.json` includes Armenian (`hye_Armn`), which is absent from `swadesh_100.json`, AND in `backend/app/benchmarks.py` (line 416), `load_color_terms()` loops over `data.get("concepts", {})` instead of `data.get("colors", {})` when popping `EXCLUDED_LANGUAGES`!
  - **Recommendation**:
    1. Fix `bk_stage['grey'] = 5.0` in `generate_stats.py` and `generate_figures.py` (or report both midpoint $5.0$ [$r = 0.89$] and upper-bound $7.0$ [$r = 0.82$]), and plot `grey` as an open diamond (`marker='D'`, `facecolor='none'`) in `generate_figures.py` to match the Figure 10 caption.
    2. Fix `data.get("colors", {})` in `benchmarks.py:load_color_terms()` and either harmonize the language list to the 135 Swadesh languages or explicitly note in Section 3.1 that the color corpus includes Armenian ($N = 136$).
    3. Briefly verify in the text whether PC1 also recovers the Berlin & Kay hierarchy when color centroids are computed after per-language mean-centering or on bare words.

### Major Suggestion 5: Exact Two-Way Variance Partition for Equation (1) and True Language-Centered Perturbation Residuals $\hat{\boldsymbol{\epsilon}}_{c,\ell}$ (Sections 3.2, 4.1 & 4.6)
- **The Scientific Issue**: Equation (1) ($\mathbf{e}_{c,\ell} = \boldsymbol{\mu}_c + \boldsymbol{\ell} + \boldsymbol{\epsilon}_{c,\ell}$) is the theoretical spine of the paper. Two targeted refinements will make this decomposition mathematically airtight:
  1. **Direct Variance Partition ($R^2$ of $\boldsymbol{\mu}_c$ vs. $\boldsymbol{\ell}$)**: Currently, Section 4.1 evaluates Equation (1) via the Conceptual Store Metric (a ratio of cosine distances). Complementing CSM with the exact two-way additive variance partition (fraction of total sum of squares $\text{SS}_{\text{total}} = \sum_{c,\ell} \|\mathbf{e}_{c,\ell} - \bar{\mathbf{e}}\|_2^2$ explained by language centroids $\text{SS}_{\text{lang}}$, concept centroids $\text{SS}_{\text{concept}}$, and residual interaction $\text{SS}_{\text{resid}}$) in both the Bare-Word and Carrier-Frame regimes will directly quantify how much variance shifts from $\boldsymbol{\ell}$ to $\boldsymbol{\mu}_c$.
  2. **Subtracting $\hat{\boldsymbol{\ell}}$ When Computing Perturbation Coupling in `semantic_offset_invariance`**: Look closely at `backend/app/benchmarks.py` (lines 974–977):
     ```python
     mu_a = np.mean(mat_a, axis=0)
     mu_b = np.mean(mat_b, axis=0)
     eps_a = mat_a - mu_a
     eps_b = mat_b - mu_b
     ```
     Because `mat_a` and `mat_b` are uncentered embeddings $\mathbf{e}_{a,\ell}$ and $\mathbf{e}_{b,\ell}$, subtracting only the concept mean $\boldsymbol{\mu}_a$ leaves the shared language shift $\boldsymbol{\ell}$ inside both `eps_a` and `eps_b` ($\mathbf{e}_{a,\ell} - \boldsymbol{\mu}_a = \boldsymbol{\ell} + \boldsymbol{\epsilon}_{a,\ell}$)! While $\boldsymbol{\ell}$ cancels out when taking the difference $\mathbf{d}_\ell(a,b) = \mathbf{e}_{b,\ell} - \mathbf{e}_{a,\ell}$, it does *not* cancel out in $\cos(\mathbf{e}_{a,\ell} - \boldsymbol{\mu}_a, \mathbf{e}_{b,\ell} - \boldsymbol{\mu}_b)$ unless per-language means $\hat{\boldsymbol{\ell}}$ are subtracted first ($\hat{\boldsymbol{\epsilon}}_{c,\ell} = \mathbf{e}_{c,\ell} - \hat{\boldsymbol{\mu}}_c - \hat{\boldsymbol{\ell}} + \bar{\mathbf{e}}$). Computing perturbation coupling on the doubly centered residuals $\hat{\boldsymbol{\epsilon}}_{a,\ell}, \hat{\boldsymbol{\epsilon}}_{b,\ell}$ will ensure the code matches the exact definition of $\boldsymbol{\epsilon}_{c,\ell}$ in Equation (1) and Section 3.2 (line 64).

### Major Suggestion 6: Subword Tokenization Fertility & Exact-Match Extraction Rate (Section 3.1 & Section 5.3)
- **The Scientific Issue**: In Section 3.1 (line 12), target word representations in the carrier frame are extracted by locating subword token indices $T(w_{c,\ell})$ via `_find_word_token_indices`, *"falling back to full-sentence mean-pooling only when subword boundaries merge across token pieces."* A meticulous NLP reviewer will ask:
  1. What percentage of the $101 \times 135 = 13{,}635$ concept–language pairs succeed via exact subword span isolation vs. falling back to full-sentence pooling? (If fallback were frequent, English carrier tokens would be averaged directly into $\mathbf{e}_{c,\ell}$).
  2. Does cross-lingual subword fertility (mean number of SentencePiece tokens per word across the 135 languages) correlate with concept convergence $C(c)$?
- **Recommendation**: Report the exact subword span match rate in Section 3.1 (e.g., $>99\%$), and report the correlation between mean subword token count and $C(c)$ in Section 4.3 alongside orthographic and phonological similarity.

---

## 3. Constructive Structural, Rhetorical & Presentation Improvements (Whitesides Principles)

Following George Whitesides' principles of scientific communication (*outline-driven flow, self-contained figures/tables, equation clarity, and foregrounding the main claim at the start of each paragraph*), the manuscript is already very well structured. The following targeted refinements will sharpen its impact:

### A. Equations & Mathematical Notation (Section 1 & Section 3.2)
1. **Identifiability Constraint on Equation (1) (`01-introduction.tex`, line 11; `03-methods.tex`, line 43)**:
   - In $\mathbf{e}_{c,\ell} = \boldsymbol{\mu}_c + \boldsymbol{\ell} + \boldsymbol{\epsilon}_{c,\ell}$, adding a constant vector $\mathbf{v}$ to $\boldsymbol{\mu}_c$ and subtracting $\mathbf{v}$ from $\boldsymbol{\ell}$ leaves $\mathbf{e}_{c,\ell}$ unchanged unless a centering convention is stated.
   - **Fix**: Explicitly state the standard zero-sum identifiability constraint ($\sum_\ell \boldsymbol{\ell} = \mathbf{0}$ and $\mathbb{E}[\boldsymbol{\epsilon}_{c,\ell}] = \mathbf{0}$, or define $\hat{\boldsymbol{\ell}} = \frac{1}{M}\sum_{c=1}^M (\mathbf{e}_{c,\ell} - \bar{\mathbf{e}})$ relative to the grand mean $\bar{\mathbf{e}}$) so that $\hat{\boldsymbol{\mu}}_c$ and $\hat{\boldsymbol{\ell}}$ are uniquely defined.
2. **Consistent Notation Between Equations (3)–(4) and Section 4.6 (`03-methods.tex`, lines 43–64)**:
   - In Equation (4) (`03-methods.tex`, line 45), the concept centroid is denoted $\bar{\mathbf{e}}_c = \frac{1}{N}\sum_{\ell=1}^N \mathbf{e}_{c,\ell}$, whereas in Equation (1) and line 57 it is denoted $\boldsymbol{\mu}_c$. Using $\hat{\boldsymbol{\mu}}_c$ (or noting $\bar{\mathbf{e}}_c \equiv \hat{\boldsymbol{\mu}}_c$) keeps the notation unified across all sections.
   - In Equation (4), clarify in the text that for language-centered CSM, $\mathbf{e}_{c,\ell}$ in the denominator is replaced by $\mathbf{e}^{\text{cen}}_{c,\ell} = \mathbf{e}_{c,\ell} - \hat{\boldsymbol{\ell}}$ (note that the numerator concept centroids $\hat{\boldsymbol{\mu}}_c$ are invariant to per-language centering when the concept $\times$ language matrix is complete, up to a constant global shift, so the CSM improvement comes specifically from collapsing the within-concept cross-lingual dispersion in the denominator!). Pointing out this mathematical insight in Section 3.2 or 4.1 is very illuminating for readers.

### B. Figure Ordering, Self-Contained Captions, and Visual Polish
1. **Figure 1 Placement (`03-methods.tex`, lines 16–21)**:
   - Currently, `fig_water_manifold.pdf` is placed inside Section 3.1 (Methods) as Figure 1, before Results even begins, and is only briefly mentioned in Section 3.1 (line 12) and Section 4.2 (line 84).
   - **Suggestion**: Introduce Figure 1 explicitly in Section 1/Section 3.1 as a concrete visual schematic of the extraction pipeline and language-family subclustering.
2. **Figure 2 (`fig_conceptual_store.pdf`, `04-results.tex`, lines 47–52)**:
   - In `generate_figures.py` (line 632), the x-axis labels are `'Contextual (Shared Frame)'` (left) and `'Decontextualized (Bare Word)'` (right), whereas the text in Section 4.1 (lines 54–57) and Table 1 (rows 1a, 1b) discusses **Bare Word first** ($1.15 \to 2.46$, $2.14\times$) and **Carrier Frame second** ($2.25 \to 2.69$, $1.19\times$). Ordering the two bar groups in `fig_conceptual_store.pdf` in the same order as the text (or matching the text order to the figure) will make reading effortless.
3. **Figure 3c (`fig_layerwise_trajectory.pdf`, `04-results.tex`, line 62)**:
   - The caption states: *"Per-concept convergence heatmap across layers (rows sorted from highest to lowest final convergence)"*, and does not mention that Panel (c) plots the **top 17 and bottom 18 concepts ($n = 35$ shown)** rather than all 101 concepts (`generate_figures.py`, line 2090: `n_show = min(35, len(all_concepts))`). Update the caption to say *"for the top-17 and bottom-18 Swadesh concepts"* and consider adding a thin horizontal divider between the top-17 and bottom-18 blocks in Panel (c).
4. **Figure 10 (`fig_color_circle.pdf`, `04-results.tex`, lines 214–219)**:
   - As noted in Major Suggestion 4, update Panel (b) in `generate_figures.py` so that `grey` is plotted at Stage $5.0$ with an open diamond marker (`marker='D'`) as stated in the caption.
5. **Figure 12 (`fig_offset_vector_demo.pdf`, `04-results.tex`, lines 244–249)**:
   - In `precompute_paper_data.py` (line 832), `top_k` selects the top 5 semantic pairs sorted by `centroid_offset_norm` (`fire–water`, `sun–moon`, `eye–ear`, `night–sun`, `dog–fish`), whereas the reader might expect the top 5 semantic pairs sorted by **residual consistency** (`man–woman`, `big–small`, `fire–water`, `sun–moon`, `one–two`) after reading lines 238–252! Explaining in the caption or switching `top_k` to the top-5 residual-consistency pairs will align Figure 12 with the main takeaway of Section 4.6.

### C. Table 1 (`04-results.tex`, lines 7–39)
- Table 1 is an excellent executive summary table. Two small enhancements would make it even stronger:
  1. In Row `3a. Surface-Form Controls`, add the **Semantic Category ANOVA** result ($\eta^2 = 0.254$, $p = 2.21 \times 10^{-4}$) right next to `Ortho R² = 0.012; Phon R² = 0.000` so the reader sees both what *doesn't* explain convergence and what *does* explain it.
  2. In Row `4a. Subfamily Phylogeny`, include the same-script control ($\rho_s = 0.69$) or bare-word subfamily $\rho_s$.

### D. Alignment with Companion Webpage (`docs/index.html`)
- **Three specific places where `docs/index.html` can be synchronized with the manuscript**:
  1. **Section 6.6b of `docs/index.html` (lines 979–1019, Layer-wise Color Circle Evolution)**: Mention in Section 4.6 of the paper that layer-wise color emergence mirrors the Layer 6 CSM inflection.
  2. **Number of offset control pairs**: `DEFAULT_OFFSET_PAIRS` in `benchmarks.py` (lines 924–932) defines 8 negative control pairs, including `("smoke", "leg")`. Because `"leg"` is not one of the 101 concepts in `swadesh_100.json`, `("smoke", "leg")` is silently skipped, leaving 7 control pairs (`OffsetControlCount = 7`, total `22` pairs). Removing `("smoke", "leg")` from `DEFAULT_OFFSET_PAIRS` will make the code intention and output count ($15 + 7 = 22$) 100% intentional.

---

## 4. Minor Comments & Technical Polish (Line-by-Line)

- **`01-introduction.tex`, line 25**: *"across $k \in \{1, 3, 5, 10\}$"* vs. `03-methods.tex` line 29 and Table 1 row 2b: *"$k \in \{0, 1, 3, 5, 10\}$"*. Make the set of $k$ values consistent (`{0, 1, 3, 5, 10}`).
- **`03-methods.tex`, line 19 (Figure 1 caption)**: *"across 29 typologically diverse languages"* vs. `docs/index.html` line 192 (*"each of 30 typologically diverse languages"*). Update `docs/index.html` line 192 from `30` to `29`.
- **Replace the 4 remaining hardcoded numbers in `04-results.tex` with `stats.tex` macros**: `1{,}792` ($\ge 1$ CLICS³ family), `3{,}828` ($\binom{88}{2}$), `81.8\%` ($3{,}133 / 3{,}828$), and `39.1\%` ($22.8\% + 16.3\%$).
- **Uncited Bibliography Entries in `paper/references.bib`**: Cite `vaswani2017`, `mikolov2013`, `voita2021`, `goyal2022`, `hewitt2019`, and `miller1995` in the appropriate sections.
