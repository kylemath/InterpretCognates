# Field-Leader Peer Review Panel Synthesis (Round 4: Pre-Submission Polish)

**Manuscript:** *Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*  
**Date:** November 2, 2026  
**Overall Panel Recommendation:** **Unanimous Accept with Minor Revisions** (*TACL*, *Computational Linguistics*, *Nature Human Behaviour*, *Open Mind*, or *Cognition*)

---

## 1. Panel Composition & Individual Verdicts

| Review File | Simulated Field Leaders (Cited in Manuscript) | Domain Focus | Verdict |
| :--- | :--- | :--- | :--- |
| [`field_leader_1_bergen_chang.md`](file:///Users/kylemathewson/InterpretCognates/paper/reviews/field_leader_1_bergen_chang.md) | **Prof. Benjamin K. Bergen & Dr. Tyler A. Chang** (UC San Diego — `chang2022`) | Multilingual subspace geometry, effective dimensionality, anisotropy ($k$-sensitivity), embodied vs. distributional grounding | **Accept with Minor Revisions** |
| [`field_leader_2_list_jaeger.md`](file:///Users/kylemathewson/InterpretCognates/paper/reviews/field_leader_2_list_jaeger.md) | **Prof. Johann-Mattis List & Prof. Gerhard Jäger** (Univ. Passau / MPI-EVA & Univ. Tübingen — `list2018`, `rzymski2020`, `jaeger2018`) | Quantitative historical linguistics, ASJP LDND vs. NLD, $\text{CLICS}^3$ Concepticon polysemy vs. homophony, morphological simplexity | **Accept with Minor Revisions** |
| [`field_leader_3_fedorenko_deniz_kroll.md`](file:///Users/kylemathewson/InterpretCognates/paper/reviews/field_leader_3_fedorenko_deniz_kroll.md) | **Prof. Evelina Fedorenko, Prof. Fatma Deniz & Prof. Judith F. Kroll** (MIT, TU Berlin / UC Berkeley & UC Irvine — `malikmoraleda2022`, `deniz2025`, `kroll1994`, `kroll2010`) | Core fronto-temporal language network vs. ATL conceptual hub, two-level cortical bilingual tuning ($\hat{\boldsymbol{\ell}} + \hat{\boldsymbol{\epsilon}}_{c,\ell}$), RHM resource asymmetry | **Accept with Minor Revisions** |
| [`field_leader_4_voita_vulic.md`](file:///Users/kylemathewson/InterpretCognates/paper/reviews/field_leader_4_voita_vulic.md) | **Dr. Elena Voita & Prof. Ivan Vulić** (FAIR / Edinburgh & Univ. Cambridge — `voita2019`, `voita2021`, `vulic2020`) | Layer-wise NMT encoder evolution (Layer 10 $\to$ 11 cone expansion & POS-selective unpacking), *Multi-SimLex* hierarchy, offset variance identity | **Accept with Minor Revisions** |

---

## 2. Actionable Enhancements Integrated into the Manuscript

All four panels confirmed that the manuscript's statistical controls and data pipeline are rock-solid and identified high-leverage insights that could be verified directly from the existing precomputed JSON artifacts in `docs/data/`:

1. **Subspace Effective Dimensionality ($\text{PR} = (\sum \lambda_i)^2 / \sum \lambda_i^2$) & Chang et al. (2022) Alignment (Bergen & Chang):**
   - Added automated Participation Ratio macros (`\PRLangBare` $= 13.3$, `\PRLangCtx` $= 18.8$, `\PRConcept` $= 38.7$) to `generate_stats.py` and integrated into Section 4.1 (`04-results.tex`) and Section 2.1 (`02-background.tex`). The language-identity subspace $\{\hat{\boldsymbol{\ell}}\}$ is concentrated in a low-dimensional shift manifold ($\text{PR} = 13.3$ on bare words; top 10 PCs $= 66.6\%$ of variance), whereas the conceptual manifold $\{\hat{\boldsymbol{\mu}}_c\}$ spans nearly $3\times$ higher effective dimensionality ($\text{PR} = 38.7$; top 10 PCs $= 40.1\%$).
   - Clarified in `01-introduction.tex` both the concept ($\CSMSSConceptBare\%$ bare, $\CSMSSConceptCtx\%$ carrier) and language ($\CSMSSLangBare\%$ bare, $\CSMSSLangCtx\%$ carrier) variance shares in Equation 1.
   - Reported the ABTT $k=0 \to 1$ knee ($0.72 \to 0.59 \to 0.58$) in Section 4.2 (`04-results.tex`).

2. **Layer 10 $\to$ Layer 11 Cone Expansion, POS-Selective Unpacking, *Multi-SimLex*, and Offset Variance Identity (Voita & Vulić):**
   - Unpacked the exact mechanism of the Layer 10 $\to$ Layer 11 transition in Section 4.1 (`04-results.tex`): while within-concept distance expands by $+45.5\%$ ($0.136 \to 0.198$) as pronouns ($\Delta = -0.15$) and verbs ($\Delta = -0.09$) unpack target-language morphosyntactic conditioning for the decoder (reducing mean $C(c)$ from $0.86$ to $0.80$, while concrete nouns fitting the nominal slot peak at Layer 11 at $0.95\text{--}0.96$), between-concept centroid distance expands even faster by $+47.1\%$ ($0.462 \to 0.680$), raising language-centered CSM from $3.89$ (Layer 10) to $4.01$ (Layer 11).
   - Connected the POS/concreteness gradient (People $0.77$, Nature $0.72$ $>$ Descriptors $0.57$ $>$ Actions $0.54$ $>$ Pronouns $0.45$) directly to *Multi-SimLex* (`\citet{vulic2020}`) in Section 2.1 and Section 4.2.
   - Added the explicit perturbation variance identity $\sigma_{\Delta\epsilon}^2 = \sigma_{\epsilon_a}^2 + \sigma_{\epsilon_b}^2 - 2\sigma_{\epsilon_a}\sigma_{\epsilon_b}\cos(\hat{\boldsymbol{\epsilon}}_{a,\ell}, \hat{\boldsymbol{\epsilon}}_{b,\ell})$ in Section 3.2 (`03-methods.tex`) and cited top perturbation-coupled pairs (*man–woman* $+0.35$, *I–we* $+0.34$, *die–kill* $+0.24$, *eat–drink* $+0.21$) in Section 4.6 (`04-results.tex`).

3. **ASJP LDND vs. NLD, $\text{CLICS}^3$ Accidental Homophony vs. Conceptual Polysemy Step Function, and Morphological Simplexity (List & Jäger):**
   - Verified and reported in Section 3.2 and Section 4.4 that Jäger's (2018) off-diagonal normalized **LDND** metric agrees at $\rho_s = 0.983$ with NLD within subfamilies and slightly *increases* the within-subfamily embedding correlation to $\rho_s = \MantelLDNDSubfamilyRho{}$ ($0.67$) and global Mantel to $\rho_s = \MantelLDNDRho{}$ ($0.15$).
   - Reported the nonlinear $\text{CLICS}^3$ family-tier step function in Section 4.5 (`04-results.tex`): single-family attestations ($1$ family, $n=1{,}241$, $\mu = 0.30$ — which mix accidental homophony like English *bark*) show virtually zero gain over never-colexified pairs ($0$ families, $n=3{,}258$, $\mu = 0.29$), whereas recurrent cross-family colexifications jump sharply at $\ge 3$ families ($\mu = 0.37$), $\ge 5$ families ($\mu = 0.44$), and $\ge 10$ families ($\mu = 0.48$). Emphasized *latent conceptual colexification* on string-distinct pairs ($w_{a,\ell} \neq w_{b,\ell}$) in Section 5.2.
   - Connected the Zipfian subword fertility difference ($1.73$ vs. $2.21$ subwords) in Section 4.3 to cross-linguistic *morphological simplexity* (monomorphemic Swadesh roots vs. polymorphemic derived/compound peripheral nouns).

4. **Two-Level Cortical Bilingual Tuning ($\hat{\boldsymbol{\ell}} + \hat{\boldsymbol{\epsilon}}_{c,\ell}$), Fronto-Temporal Language Network vs. ATL Hub, and RHM Resource Asymmetry (Fedorenko, Deniz & Kroll):**
   - Articulated the two-level parallel with `\citet{deniz2025}` (Chen et al., 2025) in Section 2.2 and Section 5.1: $\hat{\boldsymbol{\ell}}$ corresponds to global cortical language-indicator shifts, while the $\sim 30\%$ interaction residual $\hat{\boldsymbol{\epsilon}}_{c,\ell}$—which exhibits $100\%$ semantic pair co-rotation ($+0.17$ carrier / $+0.23$ bare vs. $0.00$ controls)—directly mirrors semantic-cluster-specific language tuning modulations in bilingual cortex.
   - Distinguished the core fronto-temporal language network (`\citet{malikmoraleda2022}`, selective for lexical-syntactic form across 45 languages, analogous to Layers 0–5) from downstream amodal conceptual hubs like the ATL (`\citet{correia2014}`, analogous to language-centered Layers 6–10) in Section 2.2, Section 5.1, and Section 6.
   - Reported the computational analogue of Kroll's (`\citet{kroll1994,kroll2010}`) Revised Hierarchical Model resource/proficiency asymmetry across high-resource ($n=40$) vs. lower-resource ($n=95$) languages in Section 4.4 and Section 5.1: high-resource languages lie significantly closer to the shared conceptual manifold across Swadesh distances ($\mu = 0.24$ vs. $0.30, p = 2.80 \times 10^{-9}$), 3D color centroids ($\mu = 1.42$ vs. $2.05, p = 1.65 \times 10^{-8}$), and relational offset vectors ($\mu = 0.89$ vs. $0.80, p = 3.63 \times 10^{-9}$).
   - Added layer-wise color emergence statistics from `color_circle_layers.json` ($|\rho_s| = 0.81$ at Layer 0 $\to 0.96$ at Layer 7 $\to 0.93$ at Layer 11) in Section 4.6 and discussed *distributional sedimentation vs. direct sensorimotor grounding* in Section 5.2.
