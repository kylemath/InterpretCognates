# Joint Peer Review: *Universal Conceptual Structure in Neural Translation: Probing NLLB-200’s Multilingual Geometry*

**Reviewers:**  
- **Dr. Elena Voita** (FAIR / University of Edinburgh; *ACL 2019* "The Bottom-Up Evolution of Representations in the Transformer", *ACL 2019* "Analyzing Multi-Head Self-Attention", *ACL 2021* "Analyzing the Source and Target Contributions to Predictions in Neural Machine Translation")  
- **Prof. Ivan Vulić** (Language Technology Lab, University of Cambridge; *Computational Linguistics 2020* "Multi-SimLex: A Large-Scale Evaluation of Multilingual and Cross-Lingual Lexical Semantic Similarity")

**Overall Recommendation:** **Strong Accept (Champion at *TACL* / *Computational Linguistics* / *ACL*) subject to targeted mechanistic and lexical-semantic refinements**

---

## 1. Executive Summary & Joint Assessment

This manuscript presents one of the cleanest, most methodologically self-aware investigations of multilingual representation geometry we have read in recent years. Rather than treating a pretrained multilingual Transformer as a black-box similarity engine and stopping at 2D PCA/t-SNE visualizations or uncalibrated cosine similarities, the author formulates a falsifiable linear-algebraic model of the encoder space in Meta’s `facebook/nllb-200-distilled-600M` ($L=12$, $d=1024$) across $M=101$ Swadesh concepts and $N=135$ languages:
$$\mathbf{e}_{c,\ell} \;=\; \bar{\mathbf{e}} \;+\; \hat{\boldsymbol{\mu}}_c \;+\; \hat{\boldsymbol{\ell}} \;+\; \hat{\boldsymbol{\epsilon}}_{c,\ell}, \qquad \sum_{c=1}^{M}\hat{\boldsymbol{\mu}}_c = \mathbf{0}, \quad \sum_{\ell=1}^{N}\hat{\boldsymbol{\ell}} = \mathbf{0}.$$
Every claim is paired with an adversarial null or confound control: shared English carrier templates (`"I saw a {word} near the river."`) versus decontextualized native bare words; All-But-The-Top (ABTT, $k \in \{0,1,3,5,10\}$) isotropy correction; strictly disjoint writing-system pairs ($\text{script}(\ell) \neq \text{script}(\ell')$); loanword-heavy versus non-loanword concrete noun baselines with ANCOVA mediation by SentencePiece subword fertility; within-subfamily block-permutation Mantel tests ($9{,}999$ permutations) against ASJP v20; string-distinct ($w_{a,\ell} \neq w_{b,\ell}$) and cognate-excluded ($\text{NLD} \ge 0.2$) colexification tests against CLICS$^3$; and distance- and POS-matched negative controls for relational offsets.

---

## 2. High-Leverage Refinements & Actionable Recommendations

### Refinement 1: Explain the Exact Mechanistic Decoupling Between Layer 10 and Layer 11 (`docs/data/layerwise_metrics.json`)
Reconstructing the raw between-concept centroid distance $d_{\text{between}} = \text{CSM}_{\text{raw}} \times (1 - \bar{C})$ across layers in `docs/data/layerwise_metrics.json` reveals **two distinct geometric regimes** across the encoder stack:
- **Layers 6 $\to$ 10 (Within-Concept Compression at Fixed Centroid Separation):** Between Layer 6 (the CSM inflection point) and Layer 10, raw between-concept centroid distance is virtually constant ($0.466$ at `L6` $\to 0.462$ at `L10`), whereas mean within-concept distance is compressed by **$-43.3\%$** (from $0.240$ at `L6` down to $0.136$ at `L10`).
- **Layer 10 $\to$ Layer 11 (Final-Layer Cone Expansion & Morphosyntactic Differentiation):** Between Layer 10 and Layer 11, raw within-concept distance expands by **$+45.5\%$** ($0.136 \to 0.198$), **but raw between-concept centroid distance expands by $+47.1\%$ ($0.462 \to 0.680$)!** Because the inter-concept numerator expands even faster than the within-concept denominator, language-centered CSM rises from **$3.89$ (`L10`) to $4.01$ (`L11`)**.
- Furthermore, concrete nouns that fit the nominal slot (`rain, tree, dog, fish, bird, cloud, star`) actually *increase* slightly to peak at Layer 11 ($0.95\text{--}0.96$), whereas the `L10` $\to$ `L11` dip is driven selectively by Pronouns ($\Delta = -0.15$), Actions/verbs ($\Delta = -0.09$), and polysemous items (*louse, know, bark, what, that, give, say*) as the final encoder layer unpacks morphosyntactic features for decoder cross-attention (*Voita et al., 2019, 2021*).

### Refinement 2: Explicitly Connect the Noun–Adjective–Verb Convergence Gradient to *Multi-SimLex* (*Vulić et al., 2020*)
Add a direct comparison to `\citet{vulic2020}` (*Multi-SimLex*) in Sections 4.2 and 5.2, highlighting that the gap between concrete noun categories (**People** $0.77$, **Nature** $0.72$, **Animals** $0.60$) and **Properties/adjectives** ($0.57$), **Actions/verbs** ($0.54$), and **Pronouns** ($0.45$)—which persists on decontextualized bare words—replicates at 135-language scale the noun $>$ adjective $>$ verb cross-lingual alignment asymmetry established in human and embedding evaluations on *Multi-SimLex*.

### Refinement 3: Connect Centroid-Separation Regression ($r = 0.84$) and Perturbation Coupling $\cos(\hat{\boldsymbol{\epsilon}}_{a,\ell}, \hat{\boldsymbol{\epsilon}}_{b,\ell})$ via the Exact Variance Identity
In `03-methods.tex` (line 68), write out the exact identity $\sigma_{\Delta\epsilon}^2 = \mathbb{E}[\|\hat{\boldsymbol{\epsilon}}_{b,\ell} - \hat{\boldsymbol{\epsilon}}_{a,\ell}\|_2^2] = \sigma_{\epsilon_a}^2 + \sigma_{\epsilon_b}^2 - 2\sigma_{\epsilon_a}\sigma_{\epsilon_b}\cos(\hat{\boldsymbol{\epsilon}}_{a,\ell}, \hat{\boldsymbol{\epsilon}}_{b,\ell})$, and note in `04-results.tex` how tight paradigmatic pairs at small $D$ (*I–we* at $D=6.92$, coupling $+0.34$; *die–kill* at $D=7.98$, coupling $+0.24$; *man–woman* at $D=9.26$, coupling $+0.35$) achieve the strongest perturbation co-rotation.
