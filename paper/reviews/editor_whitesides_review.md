# Adversarial Peer Review Report (Round 1): *Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*

**Overall Assessment:** **Major Revision / Reject in Current Form.**  
While the manuscript addresses an important interdisciplinary question and builds a genuinely useful open-source toolkit (`InterpretCognates`), a line-by-line audit of the LaTeX source (`paper/sections/*.tex`), statistical macros (`paper/output/stats.tex`), data-generation pipeline (`paper/scripts/precompute_paper_data.py`, `generate_stats.py`, `generate_figures.py`), backend implementation (`backend/app/modeling.py`, `benchmarks.py`), and companion site (`docs/index.html`) reveals **critical discrepancies between what the paper claims and what the code actually executes**. Several flagship claims rest on code-switched English prompts falsely described as translated sentences, uncorrected embeddings falsely described as isotropy-corrected, negative controls that outperform the test pairs, a statistical test ($t = 15.69, p = 9.99 \times 10^{-29}$) interpreted as its exact opposite, and a misread PCA axis that misses the paper's strongest actual finding.

Below are the detailed reports from the three domain reviewers and the Style/Structure Editor.

---


## 4. Editor — George Whitesides Scientific Writing & Structure Review

Auditing the manuscript against George Whitesides' classic guide (*"Whitesides' Group: Writing a Paper"*, *Adv. Mater.* **2004**, *16*, 1375–1377) reveals pervasive structural and stylistic problems characteristic of a manuscript drafted by stitching together chronological lab notes.

### W1. Stop Apologizing for Pre-Registration Guesses; Frame Hypotheses Around What the Data Actually Show
Whitesides emphasizes that a paper is a logical argument explaining what nature (or the system) actually does—not a diary of the author's initial guesses that didn't pan out:
1. **Color Space Geometry (Section 4.11):** Instead of claiming PC3 is a luminance axis (which is false in the data), state the true discovery: **PC1 ($22.8\%$ variance) orders basic color terms along the Berlin & Kay (1969) evolutionary hierarchy ($\rho = 0.94$), while PC2–PC3 organize the chromatic hue circle.**
2. **Phylogenetic Structure (Section 4.8):** Instead of presenting $\rho = 0.13$ globally and apologizing that it explains only $2\%$ of variance (or leaving a failed `Hypothesis 3: rho > 0.3` box on the website), state the true linguistic finding: **NLLB-200 strongly recapitulates genealogical distance *within* language subfamilies ($\rho = 0.65, p = 2.4 \times 10^{-40}$), while cross-family distances saturate ($\rho = 0.04$) for the exact same reason lexicostatistics saturates beyond ~5,000 years BP.**
3. **Conceptual Store Separation (Section 4.10):** Delete the invented apology about falling short of a "$2\times$ threshold." State the actual geometric result: **subtracting per-language centroids increases the between-to-within concept distance ratio from $2.25$ to $2.69$ ($1.19\times$, non-overlapping 95% CIs) across 135 languages (and from $3.44$ to $4.01$ on the 39-language core subset).**
4. **Carrier Frame Effect (Section 4.6):** State what $t = 15.69, p = 9.99 \times 10^{-29}$ and $\rho = 0.867$ actually mean: **a shared contextual frame uniformly elevates baseline cross-lingual cosine similarity ($+0.128$) while leaving the relative semantic convergence hierarchy across concepts invariant ($\rho = 0.867$).**
5. **Offset Consistency vs. Cluster Separation (Section 4.12):** Explicitly report both the 15 semantic pairs and the 7 unrelated negative controls, and explain the geometric insight they reveal: **in uncentered embedding space, cross-lingual vector-offset consistency is driven by between-concept centroid distance ($\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|$), causing distant unrelated pairs (`tree--red`, `stone--sleep`) to match or exceed close antonymous pairs (`hot--cold`, `black--white`) unless offset consistency is normalized by centroid distance or evaluated via analogy retrieval accuracy.**

### W2. Organize by Scientific Logic and Importance, Not Chronological Trial-and-Error
* **Delete the "Initial Failed Experiment" Narrative in Section 4.3:** Section 4.3 (`04-results.tex`, L33–38) narrates: *"An initial comparison against 60 modern and institutional terms... yielded higher convergence for the non-Swadesh set... This comparison is confounded by loanword bias... To obtain a properly controlled comparison, we constructed a second non-Swadesh set..."* In Whitesides' words, **do not write a chronological travelogue of your mistakes**. Present the three vocabulary tiers (Swadesh core vocabulary vs. matched non-loanword concrete nouns vs. international loanwords) as a single, deliberate 3-way design testing *semantic universality vs. orthographic borrowing*.
* **Unify the Ordering Across Abstract, Introduction, Methods, and Results:**
  Currently, the Abstract and Introduction (L18–30) list 4 findings in the order: `(1) Phylogenetic -> (2) Colexification -> (3) Conceptual Store -> (4) Offset Invariance`, followed by Swadesh ranking and Color Circle as afterthoughts. Yet Section 3.3 (Methods) orders them `Swadesh -> Phylogenetic -> Colexification -> Conceptual Store -> Color Circle -> Offset Invariance -> Isotropy`, and Section 4 (Results) splinters into **12 flat subsections** (`4.1` to `4.12`) where methodological sanity checks (`4.5 Isotropy`, `4.6 Carrier`) sit right in the middle of substantive results!
  * **Recommended Logical Architecture for Section 4 (Results):**
    * **4.1 Cross-Lingual Concept Convergence Tracks Concreteness Rather Than Surface Form** (integrating the "water" manifold illustration, the 101-concept Swadesh ranking, semantic category hierarchy, orthographic/phonological variance decomposition $R^2 \le 1.2\%$, and the 3-way loanword vs. matched control comparison).
    * **4.2 Robustness to Anisotropy Correction, Contextual Framing, and Encoder Depth** (consolidating the three methodological/mechanistic validations: ABTT $k$-sensitivity $\rho \ge 0.98$, contextualized vs. bare-word rank invariance $\rho = 0.867$, and layer-wise emergence peaking at Layer 10).
    * **4.3 Subtracting Language Centroids Isolates a Shared Conceptual Subspace** (Conceptual Store Metric across 135 languages and layer-wise CSM trajectory, plus the 2D PCA concept map across 19 families).
    * **4.4 Internal Geometry Captures Cross-Linguistic Colexification and the Berlin & Kay Color Hierarchy** (CLICS³ continuous and binary colexification tests + the corrected Berlin & Kay PC1 evolutionary stage and PC2–PC3 chromatic circle analysis).
    * **4.5 Cross-Lingual Relational Offsets Reflect Centroid Separation Across Language Families** (honest analysis of the 15 semantic pairs vs. 7 negative controls, demonstrating the role of centroid norm $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|$).
    * **4.6 Embedding Distances Recapitulate Linguistic Phylogeny Within Language Families** (hierarchical clustering of 135 languages and the stratified ASJP Mantel analysis showing $\rho = 0.65$ within subfamilies vs. deep-time saturation across macro-families).

### W3. Replace Generic Lab-Notebook Headings with Information-Rich Claims
Every subsection heading in Section 4 is currently a telegraph label (`4.4 Category Summary`, `4.9 Colexification Proximity`, `4.10 Conceptual Store Metric`, `4.11 Color Circle`). Replace them with informative headings (as above) that state the scientific takeaway.

### W4. Compress Methods and Results Using Explicit Equations and a Master Summary Table
* **Zero Equations in Section 3 (`03-methods.tex`):** Adding 4 compact, numbered equations in Section 3 will eliminate pages of repetitive prose and resolve all mathematical ambiguities:
  1. **ABTT Isotropy Correction:** $\tilde{\mathbf{e}}_{c,\ell} = (\mathbf{e}_{c,\ell} - \bar{\mathbf{e}}) - \sum_{j=1}^{k} \langle \mathbf{e}_{c,\ell} - \bar{\mathbf{e}}, \mathbf{u}_j \rangle \mathbf{u}_j$.
  2. **Per-Concept Cross-Lingual Convergence:** $S(c) = \binom{L}{2}^{-1} \sum_{\ell < \ell'} \cos(\tilde{\mathbf{e}}_{c,\ell}, \tilde{\mathbf{e}}_{c,\ell'})$.
  3. **Per-Language Mean-Centering & Conceptual Store Metric (CSM):** $\hat{\mathbf{e}}_{c,\ell} = \mathbf{e}_{c,\ell} - \frac{1}{|C|}\sum_{c'} \mathbf{e}_{c',\ell}$, and $\text{CSM} = \bar{d}_{\text{between}}(\boldsymbol{\mu}_c, \boldsymbol{\mu}_{c'}) / \bar{d}_{\text{within}}(\mathbf{e}_{c,\ell}, \mathbf{e}_{c,\ell'})$.
  4. **Cross-Lingual Offset Consistency:** $\Omega(a,b) = \frac{1}{L}\sum_{\ell=1}^{L} \cos(\mathbf{e}_{b,\ell} - \mathbf{e}_{a,\ell}, \bar{\mathbf{d}}_{a,b})$.
* **Zero Tables in Section 4 (`04-results.tex`):** Add **Table 1 (Master Summary of Experiments and Statistical Tests)** reporting for every experiment: Experiment Name, Hypothesis Tested, $N_{\text{concepts}}$, $N_{\text{langs}}$, Embedding Regime (ABTT $k=3$ vs. Raw vs. Mean-Centered), Primary Metric/Statistic, Effect Size, and $p$-value.

### W5. Rewrite the Conclusion to Synthesize Rather Than Repeat Numbers
* `06-conclusion.tex` (L3–8) currently repeats 11 numerical macros verbatim from `stats.tex`, and L11–16 repeats the exact same three future-work items (`per-head attention decomposition`, `ATL probe`, `RHM asymmetry`) that were just presented in Section 5.4 (`05-discussion.tex`, L84–99)!
* **Required Fix:** Cut the redundant number-dump in the Conclusion and merge or deduplicate the repeated Future Work list. Use the Conclusion to synthesize the three overarching principles revealed by the study: (1) translation training induces a factorized geometry where language identity acts largely as a low-dimensional offset over a shared conceptual manifold; (2) cross-lingual convergence is bounded by perceptual grounding and polysemy; and (3) multilingual encoders preserve both synchronic conceptual associations (colexification, color hierarchy) and shallow diachronic genealogy (within-subfamily phylogeny).

### W6. Line-by-Line Style, Figure-Caption, and Companion-Site Fixes
1. **Naked "This" Violations (must be followed by a specific noun):**
   * `04-results.tex` L89: *"This confirms..."* $\to$ *"This rank stability confirms..."*
   * `04-results.tex` L166: *"This is consistent..."* $\to$ *"This priority of meaning over form is consistent..."*
   * `04-results.tex` L211: *"This mirrors..."* $\to$ *"This separation mirrors..."*
   * `04-results.tex` L228: *"This extends..."* $\to$ *"This cross-lingual regularity extends..."*
   * `05-discussion.tex` L10: *"This is the neural analogue..."* $\to$ *"This tuning modulation is the neural analogue..."*
   * `05-discussion.tex` L42: *"This may reflect..."* $\to$ *"This modest ratio may reflect..."*
   * `05-discussion.tex` L77: *"This echoes..."* $\to$ *"This graded association echoes..."*
   * `05-discussion.tex` L80: *"This is consistent..."* $\to$ *"This genealogical signal is consistent..."*
2. **Figure Placement & Caption Errors:**
   * **Figure 1 (`fig_water_manifold.pdf`):** Placed inside Section 3.2 (`03-methods.tex`, L29–34) two pages before it is referenced in Section 4.1 (`04-results.tex`, L9). Move it to Section 4.1.
   * **Figure 7c (`fig_layerwise_trajectory.pdf`):** Fix the integer/string key bug in `generate_figures.py` L1944 (`concept_data[c].get(str(final_layer), concept_data[c].get(final_layer, 0))`) and regenerate `fig_layerwise_trajectory.pdf` so the rows are actually sorted by final-layer convergence!
   * **Figure 8 (`fig_phylogenetic.pdf`):** Caption L135 says `\MantelNumLangs{}` ($88$), but the plot contains `\PhyloNumLanguages{}` ($135$) languages. Fix macro in caption.
   * **Figure 11 (`fig_color_circle.pdf`):** In `generate_figures.py` (L719), relabel axes/titles and update caption (`04-results.tex`, L203) to remove the false *"square markers"* and *"PC3 (luminance)"* claims and accurately highlight PC1's alignment with the Berlin & Kay hierarchy.
   * **Figure 12 (`fig_offset_combined.pdf`):** Caption (`04-results.tex`, L222) fails to mention that only the top 8 and bottom 8 pairs are shown (with 6 middle pairs omitted), and fails to distinguish the 15 semantic pairs from the 7 negative controls.
   * **Figure 13 (`fig_offset_vector_demo.pdf`):** Text (L237) and caption (L242) say *"top-4 concept pairs"*, whereas the figure plots **5** pairs (`top-5 offset pairs`). Reconcile the count.
   * **Orphaned Figures:** `generate_figures.py` generates `fig_swadesh_comparison.pdf` and `fig_variance_decomposition.pdf`, neither of which is referenced in the LaTeX files.
   * **Preprint Acknowledgments (`paper/main.tex`, L15):** Thanks *"the anonymous reviewers for their constructive feedback"* on an initial arXiv preprint. Remove or revise until peer review has actually occurred.
3. **Companion Website (`docs/index.html`) Inconsistencies to Sync with Paper:**
   * **L84:** Claims *"we operate on 200 [languages]"* (actual corpus is 135 languages after 6 exclusions).
   * **L141:** Falsely claims `"I saw a {word} near the river."` is translated into each target language.
   * **L250 (Figure 2 caption):** Falsely labels the raw water similarity matrix as *"Isotropy-corrected cosine similarity matrix"*.
   * **L346, L374, L400:** Claims orthographic regression uses *"21 Latin-script languages (210 pairs)"* (actual corpus has 84 Latin-script languages; script samples 40 = 780 pairs).
   * **L452 & L796:** Leaves stale pre-registration "Hypothesis boxes" ($\rho > 0.3, p < 0.01$ and $\ge 2\times$ improvement) right above stat badges showing $\rho = 0.13, p = 0.020$ and $1.19\times$.
   * **L596 & L655:** Numbers two different figures as **Figure 7b**.
   * **L755:** Claims CLICS³ tests `"foot"–"leg"` and `"hand"–"arm"`, which are absent from `colexification.json` because `leg` and `arm` are not in `swadesh_100.json`.
   * **L1172:** Uses pre-experiment conditional tense (*"The phylogenetic correlation, if confirmed at statistically significant levels, would be a landmark finding"*).
