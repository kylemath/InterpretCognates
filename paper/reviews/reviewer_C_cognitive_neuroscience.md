# Adversarial Peer Review Report (Round 1): *Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*

**Overall Assessment:** **Major Revision / Reject in Current Form.**  
While the manuscript addresses an important interdisciplinary question and builds a genuinely useful open-source toolkit (`InterpretCognates`), a line-by-line audit of the LaTeX source (`paper/sections/*.tex`), statistical macros (`paper/output/stats.tex`), data-generation pipeline (`paper/scripts/precompute_paper_data.py`, `generate_stats.py`, `generate_figures.py`), backend implementation (`backend/app/modeling.py`, `benchmarks.py`), and companion site (`docs/index.html`) reveals **critical discrepancies between what the paper claims and what the code actually executes**. Several flagship claims rest on code-switched English prompts falsely described as translated sentences, uncorrected embeddings falsely described as isotropy-corrected, negative controls that outperform the test pairs, a statistical test ($t = 15.69, p = 9.99 \times 10^{-29}$) interpreted as its exact opposite, and a misread PCA axis that misses the paper's strongest actual finding.

Below are the detailed reports from the three domain reviewers and the Style/Structure Editor.

---


## 3. Reviewer C — Cognitive Neuroscience & Psycholinguistics Critic

### C1. Berlin & Kay Color Circle (Section 4.11 & Figure 11): The Paper Hallucinates a "PC3 Luminance Axis" While Missing That PC1 Actually Recovers the Berlin & Kay (1969) Evolutionary Hierarchy!
* **The Claim:**
  * Section 4.11 (`04-results.tex`, L210–211): *"The 3D projection (Figure 11b) reveals that the achromatic terms—white, black, and grey—separate cleanly along the third principal component, forming a luminance axis orthogonal to the chromatic plane. This mirrors the perceptual distinction between hue and brightness and is consistent with the achromatic terms' special status in the Berlin and Kay evolutionary hierarchy."*
  * Figure 11 caption (L203): *"(b) 3D view: the third principal component separates achromatic terms (white, black, grey; square markers) from the chromatic plane, revealing a luminance axis orthogonal to the hue circle..."*
  * Discussion Section 5.1 (`05-discussion.tex`, L26–27): *"a third principal component naturally separates achromatic terms (white, black, grey) by luminance (Figure 11b)... with the hue circle in the PC1–PC2 plane and luminance along PC3."*
* **What `docs/data/color_circle.json` Actually Contains:**
  Here are the exact cross-lingual centroid coordinates $(x=\text{PC1}, y=\text{PC2}, z=\text{PC3})$ from `color_circle.json` (sorted by PC1, which explains $22.8\%$ of variance vs. $16.3\%$ for PC2 and $13.5\%$ for PC3):

  | Color Term | B&K Evolutionary Stage | PC1 (`x`, 22.8%) | PC2 (`y`, 16.3%) | PC3 (`z`, 13.5%) |
  | :--- | :---: | :---: | :---: | :---: |
  | **white** | Stage I | **$-5.26$** | $-2.62$ | $+0.22$ |
  | **black** | Stage I | **$-4.77$** | $-1.46$ | $+1.36$ |
  | **grey** | Stage VII (wildcard) | **$-0.56$** | $+1.79$ | $+1.22$ |
  | **red** | Stage II | **$-0.29$** | $-0.91$ | **$-5.84$** |
  | **green** | Stage III/IV | **$-0.16$** | **$+5.74$** | $+0.05$ |
  | **blue** | Stage IV/V | **$+0.02$** | $+2.07$ | $-0.39$ |
  | **brown** | Stage VI | **$+0.32$** | $+0.21$ | $+1.62$ |
  | **yellow** | Stage III/IV | **$+0.69$** | $+0.67$ | $+0.10$ |
  | **purple** | Stage VII | **$+2.31$** | $-1.02$ | $+0.37$ |
  | **pink** | Stage VII | **$+3.80$** | $-1.88$ | $-1.59$ |
  | **orange** | Stage VII | **$+3.90$** | $-2.58$ | **$+2.88$** |

* **Three Devastating Errors in the Paper's Account:**
  1. **PC3 (`z`) is NOT a luminance axis and does NOT separate `white`, `black`, and `grey`:** Along PC3, `white` ($+0.22$) sits right alongside `green` ($+0.05$), `yellow` ($+0.10$), and `purple` ($+0.37$), while `brown` ($+1.62$) and `orange` ($+2.88$) have higher PC3 values than `black` ($+1.36$) and `grey` ($+1.22$). Instead, **PC3 opposes `red`/`pink` ($-5.84, -1.59$) against `orange`/`brown` ($+2.88, +1.62$)**.
  2. **Figure 11b has no "square markers":** The caption claims achromatic terms are plotted with *"square markers"*, but `generate_figures.py` (L697–716) plots all 11 colors with identical circle scatter points.
  3. **PC1 (`x`) actually tracks the Berlin & Kay (1969) Evolutionary Hierarchy!** Look at PC1 (`x`): it places Stage I (`white` $-5.26$, `black` $-4.77$) at the far left; Stage II (`red` $-0.29$) and Stages III–VI (`green` $-0.16$, `blue` $+0.02$, `brown` $+0.32$, `yellow` $+0.69$) in the center; and Stage VII (`purple` $+2.31$, `pink` $+3.80$, `orange` $+3.90$) at the far right! Across the 10 canonical stage-ordered terms (excluding `grey`, which Berlin & Kay 1969 explicitly classify as a wildcard term that can enter at earlier stages), the Spearman correlation between **PC1 and Berlin & Kay evolutionary stage is $\boldsymbol{\rho = 0.94}$ ($p < 0.0001$)** (and $\rho = 0.85, p = 0.0009$ across all 11 terms)! Meanwhile, the **PC2–PC3 plane** (not PC1–PC2!) forms the chromatic hue circle (`green` at $+5.74$ on PC2 vs. `red` at $-5.84$ on PC3 vs. `orange` at $+2.88$ on PC3).
  * **Impact:** By writing the prose from a preconception instead of inspecting `color_circle.json`, the paper made a demonstrably false claim about PC3 while overlooking its most remarkable cognitive science discovery: **the dominant principal component (PC1) of NLLB-200's cross-lingual color space spontaneously recovers the Berlin & Kay (1969) evolutionary hierarchy!**

### C2. Overclaiming and Misstating Cognitive & Neuroimaging Models (BIA+, RHM, Correia et al. 2014, Chen et al. / Deniz 2025)
* **BIA+ (Dijkstra & van Heuven, 2002):**
  * Section 5.1 (`05-discussion.tex`, L12–16) claims that NLLB-200's shared encoder is a *"functional analogue of the nonselective access mechanism that BIA+ posits for human bilinguals: it maps inputs from all 135 languages into a common representational space without language-specific gating."*
  * **Critique:** In BIA+, "language-nonselective access" means that an unlabelled visual word form (e.g., *room*) simultaneously activates orthographic and phonological lexical competitors across all known languages *without the reader knowing or specifying the input language tag in advance*. In NLLB-200 (`modeling.py`, L105, L164), the caller **explicitly sets `tokenizer.src_lang = lang`**, which prepends a **language-specific BCP-47 token** to the encoder input sequence! Thus, NLLB-200's encoder has **explicit language-selective input gating** at token 0. Moreover, BIA+ is a model of bottom-up orthographic/phonological form neighborhood interference (homographs and cognates), whereas NLLB's cross-lingual convergence holds for non-cognate translations across distinct scripts. The analogy must be qualified carefully as a *shared parameter space with language-tagged inputs*, not non-selective lexical form access.
* **Revised Hierarchical Model (Kroll & Stewart, 1994; Kroll et al., 2010):**
  * Section 5.1 (L17) and Section 5.4 (L96–99) invoke RHM, and Section 5.4 proposes testing RHM translation asymmetry (`L1->L2` vs. `L2->L1`) *"by comparing embeddings extracted with L1 vs. L2 carrier sentences."*
  * **Critique:** RHM models developmental asymmetry between a dominant native language (L1) and a weaker second language (L2), positing that L2$\to$L1 translation uses direct lexical links while L1$\to$L2 requires conceptual mediation. NLLB-200 has no L1 vs. L2, no separate lexical vs. conceptual pathway in the encoder, and probing the **encoder alone** does not perform translation in a source$\to$target direction at all! If an analogy to RHM is made in Future Work, it must be framed around **high-resource vs. low-resource translation directions using full encoder–decoder cross-attention or causal interventions**, not encoder-only carrier sentences.
* **Correia et al. (2014) ATL Hub and the Fabricated "$2\times$ Threshold":**
  * Section 5.2 (`05-discussion.tex`, L40–41) states: *"The $1.19\times$ improvement in concept separability after mean-centering, while in the predicted direction, falls short of the $2\times$ threshold informally predicted from cognitive parallels with Correia et al. (2014)."*
  * **Critique:** Correia et al. (2014) performed cross-lingual fMRI MVPA classification on 4 spoken animal/non-animal words in Dutch–English bilinguals (achieving ~54–58% decoding accuracy against a 50% chance baseline). They **never defined a between/within cosine distance ratio and never predicted a $2\times$ threshold**. As `docs/index.html` (L787) reveals, the "$2\times$ threshold" came from *"the executive summary"* of the author's own initial project proposal! Attributing an arbitrary internal project target to Correia et al. (2014) is scholarly misattribution and must be removed.
* **Chen et al. / Deniz (2025) Misattribution and Conceptual Mismatch:**
  * **Author Citation Bug:** In `paper/references.bib` (L340–348), the entry `deniz2025` has author list `Chen, Catherine and Gong, Xue L. and Tseng, Christine and Gallant, Jack L. and Klein, Daniel L. and Deniz, Fatma`. Because `natbib` uses the first author's surname, `\citet{deniz2025}` renders in the compiled PDF as **Chen et al. (2025)**, creating a jarring mismatch with the BibTeX key and notes—and note the unresolved comment on L338 of `references.bib` (`% TODO: Verify DOI — 10.1073/pnas.2503721123 could not be confirmed online. bioRxiv preprint: 10.1101/2024.06.24.600505`).
  * **Theoretical Nuance:** Chen et al. showed that bilingual cortical representations exhibit *concept-category-dependent tuning shifts* between languages (i.e., how semantic dimensions are weighted rotates depending on the language being processed), whereas subtracting a per-language mean vector in NLLB-200 removes a *uniform, concept-invariant translational offset* across all words in a language. The true analogue of Chen et al.'s tuning shift in NLLB-200 is not the global language centroid removed by mean-centering, but rather the **family-specific relational geometry variations** seen in the per-family concept maps (Figure 10) and per-family offset consistency (Figure 12b).

---
