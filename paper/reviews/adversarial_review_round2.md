# Adversarial Peer Review Report — Round 2 (Post-Revision Audit)

**Manuscript Title:** *Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*  
**Companion Interactive Article:** `docs/index.html` & `docs/blog.js`  
**Audit Scope:** Full post-revision verification across `paper/main.tex`, `paper/coverpage.tex`, `paper/sections/*.tex`, `paper/output/stats.tex`, `paper/references.bib`, `paper/scripts/*.py`, `backend/app/benchmarks.py`, `paper/figures/*.pdf`, `docs/index.html`, `docs/blog.js`, and `docs/data/*.json`.

---

## Executive Summary & Round 1 Resolution Audit

The authors have carried out a remarkably substantive, honest, and mathematically rigorous revision. **Every single Major methodological, statistical, and architectural flaw from Round 1 (A1–A6, B1–B3, C1–C2, W1–W5) has been resolved in the underlying codebase, statistical macros (`paper/output/stats.tex`), and core experimental analyses:**

| Round 1 Issue | Status in Round 2 | Verification Summary |
| :--- | :--- | :--- |
| **A1** (Model size contradiction: 3.3B vs. 600M) | **RESOLVED** | Abstract (`coverpage.tex` L16), Intro (`01-introduction.tex` L8), Methods (`03-methods.tex` L6), and Table 1 (`04-results.tex` L10) now consistently specify `facebook/nllb-200-distilled-600M` (12 encoder layers, $d=1024$). (Note: 1 stale mention of "200 [languages]" remains in `docs/index.html` L85.) |
| **A2 & A3** (Carrier-sentence regime vs. bare-word regime; missing paired $t$-test) | **RESOLVED** | Methods (`03-methods.tex` L18–31) now transparently defines both the **Shared Carrier-Frame Regime** (`"I saw a {word} near the river."` / `"I drink {word}."`) and the **Decontextualized Bare-Word Regime**. Results (`04-results.tex` L118–125) accurately reports that concept rank order is invariant ($\rho_s = 0.867, p = 1.12 \times 10^{-31}$) while the shared English frame elevates absolute similarity by $\Delta\mu = +0.128$ ($\mu_{\text{ctx}} = 0.58$ vs. $\mu_{\text{bare}} = 0.46$, paired $t = 15.70, p = 9.87 \times 10^{-29}$). (Note: 1 stale sentence in `docs/index.html` L1213 still claims translated carrier sentences.) |
| **A4 & C2** (Conceptual Store Metric in carrier frame vs. bare words; fabricated $2\times$ threshold) | **RESOLVED** | Added bare-word CSM to `conceptual_store.json` and `stats.tex`: mean-centering improves CSM by **$2.14\times$** on bare words ($1.15 \to 2.46$) vs. **$1.19\times$** in the shared frame ($2.25 \to 2.69$), proving that the carrier frame itself pre-aligns language centroids in raw space. The fabricated "Correia et al. $2\times$ threshold" apology has been completely excised. |
| **A5** (Layer-wise non-monotonicity; Peak at Layer 10 vs. Layer 11 dip; Fig. 3c bug) | **RESOLVED** | Prose (`04-results.tex` L62, L67–68) and macros (`\LayerwisePeakLayer = 10`, `\LayerwisePeakConv = 0.86`, `\LayerwiseFinalConv = 0.80`) now accurately characterize the Layer 10 peak and Layer 11 decoder-preparation dip. The string/int key bug in `generate_figures.py` L2036–2050 for Figure 3c (`fig_layerwise_trajectory.pdf`) is fixed. |
| **A6** (Offset invariance negative controls & centroid-separation confound) | **RESOLVED** | `benchmarks.py` and `generate_stats.py` now separate the 15 semantic pairs ($\mu = 0.82$) from the 7 negative controls ($\mu = 0.86$), derive why raw cosine consistency scales with $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|_2$ ($r = 0.84, p = 1.03 \times 10^{-6}$), and demonstrate significant semantic alignment via distance-controlled residuals ($U = 90, p = 0.003, d = 1.06$) and perturbation coupling $\cos(\boldsymbol{\epsilon}_{a,\ell}, \boldsymbol{\epsilon}_{b,\ell})$ ($0.29$ vs. $0.14, p = 2.35 \times 10^{-5}$). |
| **B1** (Swadesh vs. non-Swadesh POS/loanword confound & 40-lang regression truncation) | **RESOLVED** | Removed the `[:40]` truncation in `generate_stats.py`: orthographic/phonological regressions now use all **92 Latin-script languages** ($R^2 = 0.012, p = 0.272$; $R^2 = 0.000, p = 0.929$). Added POS-matched Swadesh concrete nouns ($n=52$) vs. controlled non-Swadesh concrete nouns ($n=60$), showing significant Swadesh superiority in both raw ($0.83$ vs. $0.78, U = 2016, p = 0.004, d = 0.45$) and ABTT-corrected ($0.75$ vs. $0.65, U = 2099, p = 8.40 \times 10^{-4}, d = 0.54$) spaces, plus explicit surface-form overlap quantification ($0.12$ vs. $0.13$ vs. $0.26$ for loanwords). |
| **B2** (Stratified phylogenetic Mantel & bare-word Mantel) | **RESOLVED** | Reports within-subfamily ($\rho_s = 0.65, p = 2.40 \times 10^{-40}, n=321$), cross-branch IE ($\rho_s = 0.10, n=374$), and cross-family saturation ($\rho_s = 0.04, n=3133, 81.8\%$), alongside bare-word global Mantel ($\rho_s = 0.24, p = 0.001$) vs. carrier-frame ($\rho_s = 0.13, p = 0.020$). |
| **B3** (CLICS³ 58/101 gloss bug & unweighted Cohen's $d$) | **RESOLVED** | Expanded `_CLICS_GLOSS_TO_SWADESH` in `benchmarks.py` to all **101 Swadesh concepts** ($\binom{101}{2} = 5,050$ pairs; $n_1 = 201$ strongly colexified, $n_2 = 3,258$ non-colexified; $U = 492,477, p = 1.55 \times 10^{-33}$, sample-size-weighted pooled $d = 1.43$, continuous $\rho_s = 0.13, p = 8.85 \times 10^{-21}$). |
| **C1** (False PC3 luminance claim in color space) | **RESOLVED** | Replaced the false PC3 luminance story with the true discovery: **PC1 ($22.8\%$ variance) recovers the Berlin & Kay (1969) evolutionary stage hierarchy** ($\rho_s = 0.93, p = 8.48 \times 10^{-5}$; Pearson $r = 0.82, p = 0.002$), illustrated in the new 2-panel `fig_color_circle.pdf`. |
| **W1–W5** (Whitesides organization, numbered equations, Master Table 1, Conclusion) | **RESOLVED** | Results reorganized into 6 logical subsections; numbered Equations (1)–(5) added; Master Summary Table 1 (`tab:master-summary`) added; Discussion and Conclusion rewritten. |

However, our Round 2 line-by-line and pixel-by-pixel audit uncovered **three categories of remaining issues** that must be cleaned up before final acceptance:
1. **Hallucinated / Non-Corpus Concept Examples in Prose (`04-results.tex`, `05-discussion.tex`, `docs/index.html`)** where the text names concepts or control pairs that do not exist in the corpus or do not match the adjacent figure.
2. **Visual Label Collisions in Compiled PDF Figures (`paper/figures/*.pdf`)**.
3. **Minor Residual Mismatches in Bibliography/Text (`deniz2025` vs. `Chen et al.`), Discussion BIA+ Gating Prose, and Stale Website Text (`docs/index.html`, `docs/blog.js`)**.

---

## Reviewer A (Mechanistic Interpretability & Multilingual NLP)

### Assessment of Revisions
The revisions to the architectural, carrier-frame, CSM, layer-wise, and offset-invariance analyses are exemplary. By explicitly contrasting the shared English carrier frame against the decontextualized bare-word baseline across convergence rankings ($\Delta\mu = +0.128, t = 15.70$), CSM ($1.19\times$ vs. $2.14\times$), and Mantel tests ($\rho_s = 0.13$ vs. $0.24$), what was previously a hidden methodological confound is now a central mechanistic insight into how contextual templates pre-align multilingual representations. Likewise, the mathematical decomposition of raw offset consistency into centroid separation $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|_2$ ($r = 0.84$) plus perturbation coupling $\cos(\boldsymbol{\epsilon}_{a,\ell}, \boldsymbol{\epsilon}_{b,\ell})$ completely rescues Section 4.6.

### Remaining Issues to Fix (Reviewer A)

1. **[CRITICAL — Prose vs. Figure Mismatch] Hallucinated Negative Control Pairs and Semantic Pair in Section 4.6 (`paper/sections/04-results.tex` L234, L242, L252; `paper/sections/05-discussion.tex` L33; `docs/index.html` L830, L839)**
   - In `paper/sections/04-results.tex` and `05-discussion.tex`, the prose states that the 7 negative control pairs include `\emph{tree--water, eye--skin, bone--ashes}` and mentions `\emph{come--go}`.
   - Look at `backend/app/benchmarks.py` L906–932 and Figure 13a (`fig_offset_combined.pdf`):
     - The actual 7 negative control pairs in `DEFAULT_OFFSET_PAIRS` (and plotted on the y-axis of Figure 13a with `[ctrl]`) are: **`tree–red`**, **`stone–sleep`**, **`seed–night`**, **`rain–tooth`**, **`path–tongue`**, **`cloud–bone`**, and **`leaf–come`**. (`tree–water`, `eye–skin`, and `bone–ashes` do not exist in `offset_invariance.json`; in fact, the corpus concept is `ash`, not `ashes`!)
     - The actual motion pair in `DEFAULT_OFFSET_PAIRS` is **`come–give`** ($\mu = 0.704$), NOT `come–go` (`go` is not in the 101-concept Swadesh corpus!).
   - Replace those examples in `04-results.tex`, `05-discussion.tex`, and `docs/index.html` with the actual pairs from Figure 13 (`\emph{tree--red, stone--sleep, seed--night}`).

2. **[MINOR — Prose vs. Figure 3c Mismatch] Non-Swadesh Concept `mother` Cited in Layer-wise Trajectory (`paper/sections/04-results.tex` L70)**
   - `04-results.tex` L70 states that concrete nouns `(\emph{water, night, sun, mother})` achieve high cross-lingual alignment early. `mother` is **not** in the 101-concept Swadesh corpus! Update L70 to cite concepts actually visible at the top of Figure 3c (e.g., `\emph{rain, tree, water, night}`).

3. **[MINOR — Caption vs. Figure 14 Mismatch] `four` vs. `five` in Figure 14 (`paper/sections/04-results.tex` L247)**
   - `fig_offset_vector_demo.pdf` plots 5 offset pairs (`top-5`), whereas the caption says `four representative semantic pairs`. Also, ensure `offset_vector_demo.json` filters out `is_control` pairs before selecting the top-5 pairs so that only genuine semantic pairs are plotted.

4. **[MINOR — Discussion BIA+ Gating Nuance] Source-Language Token Conditioning (`paper/sections/05-discussion.tex` L14)**
   - Reword `05-discussion.tex` L14 to note that the encoder uses a single shared parameter space conditioned on a prepended source-language token rather than separate language-specific encoders.

---

## Reviewer B (Quantitative Typology, Historical Linguistics & Statistical Methodology)

### Assessment of Revisions
All three Major statistical and typological flaws from Round 1 have been thoroughly resolved. Expanding the Latin-script orthographic and phonological regressions to all 92 Latin-script languages ($R^2 = 0.012$ and $R^2 = 0.000$), adding the POS-matched concrete noun comparison ($n=52$ Swadesh vs. $n=60$ controlled non-Swadesh concrete nouns, $p = 0.004$ raw, $p = 8.40 \times 10^{-4}$ ABTT-corrected) alongside explicit orthographic overlap ($0.12$ vs. $0.13$ vs. $0.26$), stratifying the Mantel test by phylogenetic tier, and fixing the CLICS³ mapping to cover all 101 Swadesh concepts ($\binom{101}{2} = 5,050$ pairs) with sample-size-weighted pooled Cohen's $d = 1.43$ make the quantitative typology sections airtight.

### Remaining Issues to Fix (Reviewer B)

1. **[CRITICAL — Prose vs. Corpus Mismatch] Non-Swadesh Concepts Cited as Swadesh Colexification Pairs (`paper/sections/04-results.tex` L201; `docs/index.html` L762)**
   - In `04-results.tex` L201 (and `docs/index.html` L762), the text cites `\emph{hand--arm}, \emph{foot--leg}, \emph{tree--wood}, \emph{skin--bark}, \emph{earth--sand}, and \emph{see--know}`.
   - **`arm`, `leg`, and `wood` are NOT in the 101-concept Swadesh corpus!**
   - Replace them with the actual top colexified pairs in our 101-concept Swadesh dataset (matching Figure 11 `fig_colexification.pdf`): `\emph{bark--skin}` (49 families), `\emph{lie--sleep}` (42 families), `\emph{man--person}` (41 families), `\emph{drink--eat}` (26 families), `\emph{earth--mountain}` (25 families), `\emph{big--many}` (22 families), and `\emph{that--this}` (19 families).

2. **[MINOR — Prose vs. Corpus Mismatch] Non-Existent Controlled Concept `blanket` (`paper/sections/04-results.tex` L153)**
   - Replace `blanket` (not in `non_swadesh_controlled.json`) with `needle, ladder` (`\emph{hammer, basket, axe, spider, needle}`).

3. **[MINOR — Figure 11 Legend Clarification] `Colexified (1,792)` vs. `n = 201` (`paper/scripts/generate_figures.py` L521)**
   - Clarify in `fig_colexification.pdf` legend and caption that the 1,792 red points show all pairs attested in $\ge 1$ CLICS³ family, of which $n = 201$ are attested in $\ge 3$ families.

---

## Reviewer C (Cognitive Science, Psycholinguistics & Neuroscience of Bilingualism)

### Assessment of Revisions
Replacing the unsupported PC3 luminance narrative with the empirical demonstration that PC1 ($22.8\%$ variance) recovers the Berlin & Kay (1969) evolutionary stage hierarchy ($\rho_s = 0.93, p = 8.48 \times 10^{-5}$) turns the color analysis into one of the strongest cognitive findings in the paper.

### Remaining Issues to Fix (Reviewer C)

1. **[MINOR — Bibliography vs. Website Citation Consistency] `deniz2025` Key vs. `Chen, Catherine` First Author (`paper/references.bib` L338–348; `docs/index.html`)**
   - In `references.bib`, the entry `deniz2025` has first author `Chen, Catherine` and last author `Deniz, Fatma` (rendering as `Chen et al. (2025)` in LaTeX), whereas `docs/index.html` writes `Deniz et al. (2025)`. Align `docs/index.html` to `Chen et al. (2025)` and remove the `% TODO` comment in `references.bib`.

---

## The Editor (in the spirit of George M. Whitesides)

### Assessment of Revisions & Remaining Visual Polish
The manuscript's organization, Master Summary Table 1, numbered equations, and narrative arc now follow the Whitesides standard closely. Before final sign-off, clean up the following visual label collisions in `paper/scripts/generate_figures.py`:

1. **`fig_swadesh_ranking.pdf`**: Add top y-axis headroom so `night` is not clipped by the top spine, and use `adjust_text` so `tree` does not touch the legend box and `liver`/`bite` and `bark`/`feather`/`claw` do not overlap.
2. **`fig_color_circle.pdf`**:
   - In Panel (a), separate `white` and `black` labels (currently overlapping as `whitek`) and separate `purple` and `brown`.
   - In Panel (b), stagger `orange` (`y_off = 7`) and `pink` (`y_off = -12`) at Stage VII, and `white` (`y_off = 9`) and `black` (`y_off = -12`) at Stage I.
3. **`fig_conceptual_store.pdf`**: Place numeric value labels above the upper 95% CI error bar caps (`ci_high + 0.08`), increase `ylim`, and format multiplication factors as `1.19×` and `2.14×` (without `+`).
4. **`fig_offset_combined.pdf` & `fig_offset_vector_demo.pdf`**:
   - In `fig_offset_combined.pdf`, use consistent colors for negative controls across Panels (a) and (b), and adjust `xlim`/label placement in Panel (b) so `come–give` does not touch the legend and `tree–red` / `stone–sleep` do not overlap.
   - In `fig_offset_vector_demo.pdf`, filter `offset_vector_demo.json` to semantic pairs (`not is_control`) and stagger the vector tip labels in Panel (b).
5. **`fig_carrier_baseline.pdf`**: Use `adjust_text` in Panel (a) (`path`/`small`) and move the legend in Panel (b) so it does not overlap `moon` and `path`.
6. **Preprint & Website Polish**: Remove `"and the anonymous reviewers"` in `paper/main.tex` L15, and fix the remaining stale strings in `docs/index.html` (L85, L261, L491, L661, L971, L1189–1218).

---

## Editorial Decision (Round 2)

**Decision:** **Minor Revisions (Accept Pending Final Polish)**
