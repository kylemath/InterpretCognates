# Peer Review: *Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*

**Reviewer Role:** Reviewer 2 (Adversarial, Methodological & Statistical Audit — *TACL* / *NeurIPS* / *Cognition*)  
**Recommendation:** **Major Revision Required (Borderline Reject without Control Experiments)**

---

## 1. Overall Assessment & Core Skeptical Critique

This manuscript investigates an important, well-motivated question at the intersection of NLP interpretability, linguistic typology, and bilingual psycholinguistics: whether the shared encoder of Meta's 600M-parameter distilled NLLB-200 translation model (`facebook/nllb-200-distilled-600M`) organizes lexical representations around a language-invariant conceptual manifold ($\mathbf{e}_{c,\ell} \approx \boldsymbol{\mu}_c + \boldsymbol{\ell} + \boldsymbol{\epsilon}_{c,\ell}$) or primarily reflects surface form and language identity. The authors should be commended for several strong methodological instincts already present in the work: pairing the shared English carrier frame with a decontextualized bare-word baseline (Analyses 1 & 2), testing ABTT isotropy sensitivity across $k \in \{0, 1, 3, 5, 10\}$, identifying the modern loanword confound (*coffee, radio, internet*), stratifying the ASJP Mantel analysis by genealogical depth, and recognizing that subtracting two vectors from the same language cancels $\boldsymbol{\ell}$ in relational offset evaluations.

**However, a forensic audit of the underlying codebase (`backend/app/benchmarks.py`, `backend/app/modeling.py`, `paper/scripts/precompute_paper_data.py`, `paper/scripts/generate_stats.py`, `paper/scripts/generate_figures.py`) and precomputed JSON artifacts (`docs/data/*.json`) reveals that several of the paper's flagship "universal conceptual structure" findings are vulnerable to simpler, non-semantic null hypotheses—and in several places suffer from direct mathematical or implementation bugs where the code does not compute what the manuscript equations claim.**

Most critically:
1. **The Conceptual Store Metric (CSM) "improvement factor" is confounded by global mean-centering and lacks a permutation null baseline:** On pure i.i.d. random Gaussian noise + per-language shifts (zero cross-lingual conceptual structure), per-language mean-centering mechanically jumps CSM from $0.21$ to $1.01$ ($4.76\times$) simply because subtracting $\hat{\boldsymbol{\ell}}$ removes the global mean vector $\bar{\boldsymbol{\ell}}$ from the concept centroids $\bar{\mathbf{e}}_c$ before cosine normalization cancels the $1/\sqrt{N}$ shrinkage of random centroids.
2. **The POS-matched Swadesh vs. non-Swadesh concrete noun advantage ($p = 0.004, d = 0.45$) collapses to statistical non-significance ($t = +0.84, p = 0.4014$) when controlling for word length (`Mean_Word_Len`, $t = -2.54, p = 0.0123$):** Swadesh concrete nouns are dramatically shorter across the 39 languages than the 60 non-Swadesh controls ($4.19 \pm 0.41$ vs. $4.85 \pm 0.67$ chars, Mann-Whitney $p = 9.02 \times 10^{-9}$), and the "controlled concrete noun" list in `non_swadesh_controlled.json` actually contains abstract emotion/state nouns (*anger, fear, hunger, thirst, dream, harvest*) and English homographs (*well*).
3. **The within-subfamily phylogenetic correlation ($\rho_s = 0.65$) is $83.5\%$ same-script pairs ($268 / 321$) and reports an invalid i.i.d. $p$-value ($p = 2.40 \times 10^{-40}$) on non-independent dyadic distance matrix entries.**
4. **The CLICS³ colexification test (`benchmarks.py:689–701`) does not compute cross-lingual centroid similarity $\cos(\boldsymbol{\mu}_a, \boldsymbol{\mu}_b)$ as claimed in Section 4.5, is confounded by identical translation strings ($10.9\%$ of colexified pairs share literal identical strings in `swadesh_corpus.json`), and is heavily imbalanced on Part of Speech ($68.2\%$ vs. $30.4\%$ same-POS).**
5. **The Berlin & Kay PC1 hierarchy ($\rho_s = 0.93$) is computed solely in the ungrammatical English carrier frame `"I saw a {color} near the river."`, covaries at $\rho_s = 0.891$ ($p = 2.3 \times 10^{-4}$) with cross-linguistic word length (Zipf's Law of Abbreviation), and contains a direct code-vs.-text contradiction on how `grey` is coded (`7.0` in code vs. `5.0` in text).**
6. **In the Semantic Offset Invariance analysis (`benchmarks.py:974–989`), the "perturbation coupling" $\cos(\boldsymbol{\epsilon}_{a,\ell}, \boldsymbol{\epsilon}_{b,\ell})$ forgot to subtract the per-language vector $\boldsymbol{\ell}$ (`eps_a = mat_a - mu_a` on raw uncentered embeddings!), and the negative controls are neither distance-matched ($p = 0.0015$) nor POS-matched ($43\%$ cross-POS vs. $0\%$ in semantic pairs).**

Below, I detail all six methodological challenges and exact code/text discrepancies, followed by concrete control experiments required to make the paper's claims bulletproof.

---

## 2. Six Major Methodological Challenges & Alternative Explanations

### Challenge 1: The Conceptual Store Metric (CSM) & The Global-Mean / Cosine-Normalization Illusion (Section 4.1)

* **The Simpler Alternative Explanation ("Null Hypothesis"):**  
  Look closely at Equation (3) (`03-methods.tex:47`):
  $$\text{CSM} = \frac{\frac{2}{M(M-1)}\sum_{c < c'} d_{\cos}(\bar{\mathbf{e}}_c,\, \bar{\mathbf{e}}_{c'})}{\frac{1}{M}\sum_{c=1}^{M} \frac{2}{N(N-1)}\sum_{\ell < \ell'} d_{\cos}(\mathbf{e}_{c,\ell},\, \mathbf{e}_{c,\ell'})}$$
  Consider a strict null hypothesis where **zero universal conceptual structure exists**: let $\mathbf{e}_{c,\ell} = \boldsymbol{\ell} + \boldsymbol{\epsilon}_{c,\ell}$ where $\boldsymbol{\mu}_c \equiv \mathbf{0}$ and $\boldsymbol{\epsilon}_{c,\ell} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_{1024})$ (or equivalently, where concept labels $c \in \{1, \dots, 101\}$ are randomly permuted independently within each language $\ell$).
  - In **raw space**, every concept centroid $\bar{\mathbf{e}}_c = \bar{\boldsymbol{\ell}} + \frac{1}{N}\sum_\ell \boldsymbol{\epsilon}_{c,\ell}$ is dominated by the global mean vector $\bar{\boldsymbol{\ell}}$ because the noise averages down by $1/\sqrt{N}$ ($N=135$). Consequently, all raw centroids $\bar{\mathbf{e}}_c$ point in nearly the same direction $\bar{\boldsymbol{\ell}}$, making the numerator $d_{\cos}(\bar{\mathbf{e}}_c, \bar{\mathbf{e}}_{c'})$ artificially small (e.g., $0.21$), while the denominator is large because $\boldsymbol{\ell} \neq \boldsymbol{\ell}'$.
  - When we apply **per-language mean-centering** ($\mathbf{e}^{\text{cen}}_{c,\ell} = \mathbf{e}_{c,\ell} - \hat{\boldsymbol{\ell}}$), $\bar{\boldsymbol{\ell}}$ is removed. Even though $\bar{\mathbf{e}}^{\text{cen}}_c = \frac{1}{N}\sum_\ell (\boldsymbol{\epsilon}_{c,\ell} - \bar{\boldsymbol{\epsilon}}_\ell)$ has norm $O(1/\sqrt{N})$, **cosine distance $d_{\cos}(\bar{\mathbf{e}}^{\text{cen}}_c, \bar{\mathbf{e}}^{\text{cen}}_{c'})$ re-normalizes each centroid to unit length**, completely canceling the $1/\sqrt{N}$ shrinkage! As a result, two random centroids in $\mathbb{R}^{1024}$ have expected cosine distance $1 + \frac{1}{M-1} = 1.0100$, and within-concept distance is $1.0000$, yielding $\text{CSM}_{\text{centered}}^{\text{null}} = 1.0100$.
  - Simulating $101 \times 135 \times 1024$ i.i.d. Gaussian noise with `_conceptual_store_ratio` (`precompute_paper_data.py:524`) yields $\text{CSM}_{\text{raw}} = 0.2122 \to \text{CSM}_{\text{centered}} = 1.0100$—a **$4.76\times$ "improvement factor" on pure noise**! And at Layer 0 in `layerwise_metrics.json`, raw CSM is $0.3404$ and centered CSM is $1.1288$ ($3.32\times$ jump), almost entirely reflecting this baseline normalization artifact rather than semantic abstraction.
  - Furthermore, because $92/135$ languages ($68.1\%$, accounting for $46.3\%$ of all language pairs) use the Latin script (`_Latn`), shared SentencePiece subwords on cognates/loanwords reduce within-concept distance even on bare words.

* **Code & Mathematical Discrepancies Found:**
  1. **Two incompatible definitions of CSM in the codebase:** In `backend/app/benchmarks.py:829–833` (`conceptual_store_metric`), concept centroids are computed as the arithmetic mean of **unnormalized** embeddings (`np.mean(vecs, axis=0)`), yielding Layer 11 centered CSM $= 2.69$ (135 languages). In `paper/scripts/precompute_paper_data.py:538–544` (`_conceptual_store_ratio`), vectors are **L2-normalized to unit length before averaging** (`vecs = vecs / norm; centroids.append(vecs.mean(axis=0))`), yielding Layer 11 centered CSM $= 4.01$ (39 languages). Equation (3) in `03-methods.tex:47` writes unnormalized $\bar{\mathbf{e}}_c = \frac{1}{N}\sum_\ell \mathbf{e}_{c,\ell}$, which does not match Figure 3b (`_conceptual_store_ratio`).
  2. **Biased bootstrap confidence intervals (`benchmarks.py:845–856`):** In `_bootstrap_ratio`, concepts are resampled with replacement (`idx = rng.choice(n, size=n, replace=True)`), and `between` distances are computed over all pairs `for a in range(n) for b in range(a + 1, n)` of `centroids = [concept_centroids[i] for i in idx]`. Because sampling with replacement produces duplicate concepts ($\approx 1/M$ of all pairs have `idx[a] == idx[b]`), identical centroids with $d_{\cos} = 0$ are included in the "between-concept" average!

* **Required Controls:**
  1. Report a **concept-permutation null distribution** for CSM (permuting concept labels independently within each language $1,000$ times) and compare centered CSM against **global mean-centering** ($\mathbf{e}_{c,\ell} - \bar{\mathbf{e}}$, which removes only 1 vector instead of 135 language vectors) so readers see how much of the raw-to-centered jump is due to removing global anisotropy vs. language-specific shifts $\boldsymbol{\ell}$.
  2. Compute bare-word and carrier-frame CSM restricted strictly to **cross-script language pairs** ($\text{script}(\ell) \neq \text{script}(\ell')$, where zero orthographic or SentencePiece subword overlap is possible).
  3. Unify the CSM centroid definition between `benchmarks.py` and `precompute_paper_data.py`, and exclude `idx[a] == idx[b]` self-pairs in `_bootstrap_ratio`.

---

### Challenge 2: The POS-Matched Swadesh vs. Non-Swadesh Comparison Collapses Under Word-Length / Fertility Control (Section 4.3)

* **The Simpler Alternative Explanation ("Null Hypothesis"):**  
  Section 4.3 claims that when matched on part of speech and low orthographic similarity ($0.12$ vs. $0.13$), Swadesh concrete nouns ($n=52$, $\mu=0.83$ raw, $0.75$ ABTT) converge significantly more strongly than controlled non-Swadesh concrete nouns ($n=60$, $\mu=0.78$ raw, $0.65$ ABTT; $p = 0.004$ raw, $p = 8.40 \times 10^{-4}$ ABTT).
  However, inspecting the actual word forms across the 39 languages in `swadesh_corpus.json` and `backend/app/data/non_swadesh_controlled.json` reveals two severe confounds:
  1. **Massive Word-Length / Subword-Fertility Imbalance:** Across the 39 languages, the 52 Swadesh concrete nouns have a mean length of **$4.19 \pm 0.41$ characters**, whereas the 60 non-Swadesh nouns have a mean length of **$4.85 \pm 0.67$ characters**—a massive difference (**Mann-Whitney $U = 574.5, p = 9.02 \times 10^{-9}$**). Longer, lower-frequency words are segmented into more SentencePiece subword tokens and suffer higher mean-pooling dilution. Across the combined $112$ nouns, mean word length correlates strongly and negatively with convergence ($\rho_s = -0.336, p = 2.98 \times 10^{-4}$; within the 60 non-Swadesh nouns alone, $\rho_s = -0.343, p = 0.0074$).
  2. **When we run an OLS regression / ANCOVA of convergence on `Is_Swadesh + Mean_Word_Len + Ortho_Sim` across the 112 nouns, the Swadesh effect vanishes completely:**
     - `Is_Swadesh`: $\beta = +0.0204 \pm 0.0242$, **$t = +0.84, p = 0.4014$ (non-significant)**
     - `Mean_Word_Len`: $\beta = -0.0481 \pm 0.0189$, **$t = -2.54, p = 0.0123$ (significant)**
     - `Ortho_Sim`: $\beta = +0.1371 \pm 0.3679$, $t = +0.37, p = 0.7100$
  3. **Stimulus Contamination in `non_swadesh_controlled.json`:** Although the manuscript repeatedly describes the 60-item control set as "concrete nouns (*hammer, basket, axe, spider, needle*)" (`03-methods.tex:8`, `04-results.tex:153`), inspecting `non_swadesh_controlled.json` shows that it includes **abstract emotion, psychological, and event nouns** (`anger`, `fear`, `hunger`, `thirst`, `dream`, `harvest`, `shadow`) as well as the highly polysemous English homograph `well` (which in `"I saw a well near the river"` vs. bare word or translation tables is confounded with the adverb *well*).

* **Required Controls:**
  1. Explicitly test and report a **multiple regression / ANCOVA controlling simultaneously for mean character length, SentencePiece subword token count (fertility), and orthographic similarity** (or construct a length/fertility-matched subset of truly concrete nouns after removing abstract items like *anger, fear, hunger, thirst, dream, harvest*).
  2. Re-evaluate the Swadesh vs. non-Swadesh concrete noun comparison on **decontextualized bare words** and across **cross-script language pairs only**.
  3. If the Swadesh advantage over non-Swadesh concrete nouns is mediated by Zipfian word length / subword fertility rather than "evolutionary lexical stability" per se, state this transparently in Section 4.3 and Section 5.2! That is itself an important, honest mechanistic finding for *TACL*/*NeurIPS*.

---

### Challenge 3: Phylogenetic Signal vs. Script Identity & Non-Independent Dyadic $p$-Values (Section 4.4)

* **The Simpler Alternative Explanation ("Null Hypothesis"):**  
  Section 4.4 reports that within language subfamilies ($n = 321$ pairs), NLLB-200 embedding distance correlates strongly with ASJP phonetic distance ($\rho_s = 0.65, p = 2.40 \times 10^{-40}$).
  1. **Script Sharing Confound:** Auditing the $321$ within-subfamily pairs in `phylogenetic.json` reveals that **$268 / 321$ pairs ($83.5\%$) share the exact same writing system** (e.g., Romance in Latin script, Germanic in Latin script, Bantu in Latin script, Austronesian in Latin script). When two closely related languages share the same script (`spa_Latn`–`por_Latn`), cognates share literal SentencePiece subword IDs at the input layer.
  2. **Invalid i.i.d. $p$-value on dyadic distance matrix entries (`generate_stats.py:120`):** The manuscript states in Section 3.2 (`03-methods.tex:53`) that ASJP comparisons use "a Spearman Mantel test with 999 permutations, both globally and stratified by language subfamily." **This is untrue for the stratified statistics.** In `generate_stats.py:120`, `\MantelWithinSubfamilyP` ($2.40 \times 10^{-40}$) is computed via naive i.i.d. `scipy.stats.spearmanr(w_sub_a, w_sub_e)` treating $321$ pairwise distances among a small set of languages as independent observations! Because $\binom{k}{2}$ pairs from a $k$-language clade have only $O(k)$ degrees of freedom, treating dyads as i.i.d. vastly inflates significance ($p = 10^{-40}$ is statistically invalid for matrix entries).

* **Required Controls:**
  1. Replace the invalid i.i.d. `spearmanr` $p$-value ($2.40 \times 10^{-40}$) with a **within-subfamily block-permutation Mantel test** (permuting language labels within each subfamily across $9,999$ permutations).
  2. Report the correlation separately for **different-script within-subfamily pairs** ($n = 53$, e.g., Serbian/Russian Cyrillic vs. Polish/Slovenian Latin; Hindi Devanagari vs. Persian/Pashto Arabic vs. Tajik Cyrillic; Turkish Latin vs. Kazakh/Tatar Cyrillic—where our audit shows $\rho_s = 0.574$, demonstrating that genealogical signal *does* survive zero script overlap!) and report a **partial Mantel test controlling for script identity ($S_{\ell,\ell'} \in \{0,1\}$)**.
  3. Report the stratified correlations on **decontextualized bare words** and **language-centered embeddings**, not only carrier-frame embeddings.

---

### Challenge 4: CLICS³ Colexification — Formula Mismatch, String-Identity Contamination, and POS Imbalance (Section 4.5)

* **The Simpler Alternative Explanation ("Null Hypothesis"):**  
  1. **Direct Code-vs.-Text Mismatch on the Similarity Metric:** Section 4.5 (`04-results.tex:191, 197, 201`) repeatedly states that the experiment measures **cross-lingual centroid cosine similarity** between concept centroids $\boldsymbol{\mu}_c$ and $\boldsymbol{\mu}_{c'}$. However, inspecting `backend/app/benchmarks.py:689–701` (`colexification_test`) and `precompute_paper_data.py:810` reveals that the code does **not** compute $\cos(\boldsymbol{\mu}_a, \boldsymbol{\mu}_b)$, nor does it use ABTT-corrected or language-centered vectors! Instead, it computes the mean **within-language** cosine similarity on raw contextualized embeddings:
     $$S_{\text{code}}(a, b) = \frac{1}{N} \sum_{\ell=1}^{N} \cos(\mathbf{e}_{a,\ell},\, \mathbf{e}_{b,\ell}).$$
  2. **Literal String-Identity Contamination in the Stimulus Table:** Because `colexification_test` averages within-language similarity $\cos(\mathbf{e}_{a,\ell}, \mathbf{e}_{b,\ell})$, whenever a language $\ell$ in `swadesh_corpus.json` colexifies concepts $a$ and $b$ (i.e., translates both English glosses with the **exact same native string** $w_{a,\ell} = w_{b,\ell}$), the model receives the exact same input string `"I saw a {w} near the river."` and outputs $\mathbf{e}_{a,\ell} \equiv \mathbf{e}_{b,\ell}$, injecting a trivial $\cos = 1.0000$ into the average! Auditing `swadesh_corpus.json` against `colexification.json` shows that **$10.9\%$ of frequently colexified pairs ($\ge 3$ families) have identical translation strings in our 135-language stimulus table** (up to $27/135 = 20\%$ of languages for a single pair!), compared to only $0.6\%$ of never-colexified pairs.
  3. **Severe Part-of-Speech / Semantic-Category Imbalance:** In `benchmarks.py:621–650`, the authors wrote a helper `_semantic_proximity` to select $50$ domain-matched `non_colexified_pairs`—**and then never used `non_colexified_pairs` in `colexification_test` (`benchmarks.py:729`)!** Instead, `colexification_test` dumps all $3,258$ zero-frequency pairs into `non_colex_sims`. Among the $201$ colexified pairs ($\ge 3$ families), **$68.2\%$ share the same Part of Speech** and **$45.8\%$ share the exact same semantic category**, whereas among the $3,258$ never-colexified pairs, only **$30.4\%$ share the same POS** and **$11.8\%$ share the same semantic category**. Because embeddings in `"I saw a {word} near the river."` cluster strongly by POS/category, comparing $68\%$ same-POS pairs against $30\%$ same-POS pairs inflates the Mann-Whitney $U$ and Cohen's $d = 1.43$.

* **Required Controls:**
  1. Recompute the CLICS³ test after **excluding all within-language pairs $(a, b, \ell)$ where $w_{a,\ell} = w_{b,\ell}$** (or where normalized edit distance $< 0.2$, to eliminate morphological cognates/compounds within $\ell$).
  2. Restrict the Mann-Whitney $U$ and Spearman tests to **same-POS pairs** ($n = 137$ colexified vs. $n = 990$ non-colexified) and **same-semantic-category pairs** ($n = 92$ vs. $n = 384$), and test on **decontextualized bare words** as well as true **ABTT-corrected / language-centered concept centroids** $\cos(\boldsymbol{\mu}_a, \boldsymbol{\mu}_b)$.
  3. Reconcile the prose in Section 3.2 and Section 4.5 with the exact formula computed by the code.

---

### Challenge 5: Berlin & Kay Color Hierarchy — Carrier-Frame Ungrammaticality, Word-Length Confound, and the `grey` Coding Bug (Section 4.6)

* **The Simpler Alternative Explanation ("Null Hypothesis"):**  
  Section 4.6 claims that PC1 ($22.8\%$ of centroid variance) of the 11 basic color terms across 136 languages recovers the Berlin & Kay (1969) evolutionary hierarchy ($\rho_s = 0.93, p = 8.48 \times 10^{-5}$; Pearson $r = 0.82, p = 0.002$).
  1. **Word-Length / Zipfian Frequency Confound ($\rho_s = 0.891$):** Auditing `color_terms.json` across all 136 languages reveals that mean character length of the color translations correlates with Berlin & Kay evolutionary stage at **Spearman $\rho_s = 0.873$ ($p = 4.49 \times 10^{-4}$)** and with **PC1 at Spearman $\rho_s = 0.891$ ($p = 2.33 \times 10^{-4}$), Pearson $r = 0.788$ ($p = 0.0040$)**! Stage I–II terms (`white` $4.54$ chars, `black` $4.70$, `red` $4.91$) are short, high-frequency monomorphemic roots across languages, whereas Stage VI–VII terms (`brown` $5.94$, `purple` $5.75$, `pink` $5.55$, `orange` $6.88$) are longer, polymorphemic or derived/loaned words split into multiple SentencePiece tokens.
  2. **Contextual Carrier-Frame Artifact:** In `precompute_paper_data.py:933`, color terms are embedded exclusively inside `"I saw a {word} near the river."` (never on bare words!). In English and many target languages, splicing a bare color adjective after `"I saw a ..."` (`"I saw a red near the river"`, `"I saw a orange near the river"`) is syntactically anomalous unless the color term can double as a concrete noun (*an orange*, *a brown*, *a white*).
  3. **Direct Code-vs.-Text Bug on `grey` (`generate_stats.py:302`, `generate_figures.py:764` vs. `04-results.tex:217, 223`):**
     - `04-results.tex:223` states: *"and across all 11 terms (assigning grey its midpoint stage 5.0) at Pearson $r = 0.82$ ($p = 0.002$)."*
     - `04-results.tex:217` (Figure 11 caption) states: *"with grey shown as an open diamond representing its wildcard Stage III–VII status."*
     - **Both statements are contradicted by the code:** In `generate_stats.py:302` and `generate_figures.py:764`, `'grey': 7.0` is hardcoded as Stage **7.0**, not **5.0**! (Assigning `grey = 7.0` yields $r = 0.824, p = 0.0018$; assigning `grey = 5.0` actually yields $r = 0.886, p = 0.00028$!) And in `generate_figures.py:793`, `grey` is plotted as a standard filled circle at Stage VII (`x = 6.78`), **not** an open diamond!
  4. **Language Count Discrepancy (`136` vs. `135`):** `color_terms.json` evaluates **136** languages (`\ColorNumLanguages = 136`) because `hye_Armn` (Armenian) is present in `color_terms.json` but omitted from `swadesh_100.json` (`135` languages). Furthermore, `load_color_terms()` in `benchmarks.py:416` loops over `data.get("concepts", {})` instead of `data.get("colors", {})`, failing to strip `EXCLUDED_LANGUAGES` from the color translation dictionary.

* **Required Controls:**
  1. Recompute the Berlin & Kay PCA on **decontextualized bare words** (and language-centered bare words) across the exact same **135** languages used in the rest of the paper.
  2. Report the **exact permutation $p$-value** over all $10!$ (or $11!$) stage assignments and report the **partial correlation between PC1 and Berlin & Kay stage controlling for mean word length and SentencePiece subword token count** (our check shows partial $r = 0.687, p = 0.028$ when controlling for character length, which is much more modest than $p = 8.48 \times 10^{-5}$).
  3. Fix the `grey = 5.0` vs. `7.0` bug and the open-diamond marker in `generate_stats.py` and `generate_figures.py`.

---

### Challenge 6: Semantic Offset Invariance — Unsubtracted $\boldsymbol{\ell}$ in Perturbation Coupling, Concave Saturation Bias, and Non-Matched Controls (Section 4.6)

* **The Simpler Alternative Explanation ("Null Hypothesis"):**  
  Section 4.6 argues that after controlling for centroid separation $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|_2$, 15 semantic pairs exhibit higher residual offset consistency ($U = 90, p = 0.003, d = 1.06$) and higher cross-concept perturbation coupling ($\cos(\boldsymbol{\epsilon}_{a,\ell}, \boldsymbol{\epsilon}_{b,\ell}) = 0.29$ vs. $0.14$, $p = 2.35 \times 10^{-5}$) than 7 negative controls.
  A forensic inspection of `benchmarks.py:906–1051` and `offset_invariance.json` uncovers four serious flaws:
  1. **Implementation Bug — Per-Language Shift $\boldsymbol{\ell}$ Was Never Subtracted in `perturbation_coupling` (`benchmarks.py:974–989`):**  
     In `precompute_paper_data.py:818`, `semantic_offset_invariance` is called on **`sw_emb_ctx` (raw, uncentered contextual embeddings)**. Inside `semantic_offset_invariance` (`benchmarks.py:974–977`), the code computes:
     ```python
     mat_a = np.array([vecs_a[l] for l in common_langs], dtype=np.float64)
     mat_b = np.array([vecs_b[l] for l in common_langs], dtype=np.float64)
     mu_a = np.mean(mat_a, axis=0)
     mu_b = np.mean(mat_b, axis=0)
     eps_a = mat_a - mu_a
     eps_b = mat_b - mu_b
     ```
     Because `vecs_a[l]` is the raw embedding $\mathbf{e}_{a,\ell} = \boldsymbol{\mu}_a + \boldsymbol{\ell} + \boldsymbol{\epsilon}_{a,\ell}$, subtracting only `mu_a` leaves **`eps_a` $= \boldsymbol{\ell} + \boldsymbol{\epsilon}_{a,\ell}$**! The code **forgot to subtract $\hat{\boldsymbol{\ell}}$**! Thus, what the paper reports as $\cos(\boldsymbol{\epsilon}_{a,\ell}, \boldsymbol{\epsilon}_{b,\ell})$ (`\OffsetCouplingSemanticMean = 0.29` vs. `0.14`) is actually $\cos(\boldsymbol{\ell} + \boldsymbol{\epsilon}_{a,\ell},\, \boldsymbol{\ell} + \boldsymbol{\epsilon}_{b,\ell})$, which is heavily confounded by the shared language vector $\boldsymbol{\ell}$ and shared POS/slot interaction with the carrier sentence!
  2. **The Negative Controls Are Neither "Distance-Matched" Nor POS-Matched:**  
     - The Abstract (`coverpage.tex:20`) calls the 7 controls *"distance-matched negative controls."* In reality, the centroid separations $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|_2$ of the 15 semantic pairs (mean $10.18$, with $12/15$ pairs $< 10.3$) and 7 negative controls (mean $13.80$, with $7/7$ pairs $> 10.37$ and $6/7$ pairs $> 12.3$) **barely overlap at all** (**Mann-Whitney $U = 10.0, p = 0.0015$**)!
     - Furthermore, **$15/15$ ($100\%$) of the semantic pairs are same-POS** (`Noun–Noun`, `Adj–Adj`, `Verb–Verb`, `Num–Num`, `Pron–Pron`), whereas **$3/7$ ($43\%$) of the negative controls are cross-POS** (`stone–sleep` Noun–Verb, `tree–red` Noun–Adj, `leaf–come` Noun–Verb).
  3. **Nonlinear / Concave Ceiling Distortion in Linear Residual Regression (`benchmarks.py:1043`):**  
     By spherical geometry, for isotropic perturbations of variance $\sigma^2$, expected cosine consistency with the centroid offset of norm $D = \|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|_2$ follows the **strictly concave saturating curve** $\mathbb{E}[S(a,b)] \approx \frac{D}{\sqrt{D^2 + c}} < 1$. Because the semantic pairs sit in the middle of the curve ($D \approx 9$–$11$, where a concave curve arches *above* a straight secant line) while the negative controls sit at the right tail ($D \approx 13$–$15.5$, where a straight line overshoots the $1.0$ asymptote) and the two lowest-$D$ semantic pairs (`I–we` at $D=6.92$, `come–give` at $D=7.80$) also have strongly negative linear residuals ($-0.046, -0.057$), fitting a misspecified **linear** OLS line (`np.polyfit(norms, cons, 1)`) artificially inflates the residual gap between $D \approx 9.5$ and $D \approx 15$! Replacing the linear fit with the theoretical concave curve $D/\sqrt{D^2 + c}$ or a quadratic fit weakens the residual Mann-Whitney $p$-value by $4\times$–$6\times$ (from $p = 0.003$ to $p = 0.013$ / $p = 0.019$).
  4. **In-Sample Centroid Leakage & Silent Dropping of `("smoke", "leg")`:**  
     - In Equation (4), $\cos(\mathbf{d}_\ell(a,b), \bar{\mathbf{d}}(a,b))$ evaluates $\mathbf{d}_\ell$ against an average $\bar{\mathbf{d}}$ that includes $\mathbf{d}_\ell$ itself, rather than the leave-one-language-out centroid $\bar{\mathbf{d}}_{-\ell}(a,b)$ (or mean pairwise $\cos(\mathbf{d}_\ell, \mathbf{d}_{\ell'})$).
     - In `benchmarks.py:931`, `DEFAULT_OFFSET_PAIRS` defines **23** pairs (15 semantic + 8 controls), including `("smoke", "leg")`, which is silently dropped because `"leg"` is absent from `swadesh_100.json`.

* **Required Controls:**
  1. Fix `benchmarks.py:976–977` so that `eps_a` and `eps_b` are computed on **per-language mean-centered embeddings** ($\boldsymbol{\epsilon}_{c,\ell} = \mathbf{e}_{c,\ell} - \boldsymbol{\mu}_c - \hat{\boldsymbol{\ell}} + \bar{\mathbf{e}}$), matching Equation (1).
  2. Replace the 7 ad-hoc controls with a systematic set of **genuinely distance-matched and POS-matched unrelated pairs** (e.g., never-colexified same-POS Swadesh pairs whose centroid separation $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|_2$ falls in the exact same $[7.0, 14.5]$ range as the 15 semantic pairs!), and fit the nonlinear saturating baseline $D/\sqrt{D^2 + c}$ as well as leave-one-out offset consistency on **both carrier-frame and bare-word embeddings**.

---

## 3. Specific Statistical, Mathematical & Textual Audits

1. **Layer Indexing Mismatch ("Input Embedding Layer" vs. Encoder Block 1):**
   - In `04-results.tex:62` (Figure 3 caption), `05-discussion.tex:14`, `06-conclusion.tex:5`, and `docs/index.html:1070`, "Layer 0" (`\LayerwiseInputConv = 0.35`) is repeatedly described as *"the input embedding layer (Layer 0)"* or *"surface-sensitive token embeddings at Layer 0."*
   - However, in `backend/app/modeling.py:195, 221–222` (`embed_word_in_context_all_layers`, `include_embedding_layer: bool = False`) and `precompute_paper_data.py:563`, `hidden_states[0]` (the actual token embedding layer) is **discarded** (`hidden_states = hidden_states[1:]`), and Encoder Blocks $1 \dots 12$ are renumbered $0 \dots 11$! Thus, "Layer 0" in Figure 3 is **the output of Transformer Encoder Block 1** (after 1 round of self-attention with the English carrier frame), NOT the token embedding layer. This must be corrected in the text or `include_embedding_layer=True` should be included as Layer 0.

2. **Target-Word Subword Token Extraction Fallback Rate Not Reported (`modeling.py:124–153, 180–185`):**
   - `03-methods.tex:12` states that in the carrier frame `"I saw a {word} near the river."`, subword token indices matching $w_{c,\ell}$ are located and mean-pooled, *"falling back to full-sequence mean-pooling only when subword boundaries merge across token pieces."*
   - Report the exact fallback rate across the $13,635$ `(concept, lang)` pairs.

3. **Multiple Hypothesis Testing & Directionality of Tests:**
   - All Mann-Whitney $U$ tests (`swadesh_vs_non_swadesh_comparison`, `colexification_test`, `offset_invariance`) are one-sided (`alternative="greater"`), whereas Pearson/Spearman tests are two-sided. State this consistently across all sections.

---

## 4. Actionable Checklist of Required Control Conditions & Revisions

- [ ] **1. CSM Controls & Unification (Section 4.1):**
  - Unify the centroid L2-normalization discrepancy between `benchmarks.py:conceptual_store_metric` and `precompute_paper_data.py:_conceptual_store_ratio`.
  - Fix `_bootstrap_ratio` (`benchmarks.py:852`) so resampled duplicate concepts (`idx[a] == idx[b]`) are not compared to themselves as "between-concept" pairs.
  - Add (a) a **concept-label permutation null** ($1,000$ permutations per language), (b) a **global mean-centering baseline** ($\mathbf{e}_{c,\ell} - \bar{\mathbf{e}}$), and (c) **cross-script-only CSM** ($\text{script}(\ell) \neq \text{script}(\ell')$) on both bare words and carrier frames.

- [ ] **2. Word-Length / Subword-Fertility & True Concrete-Noun Controls (Section 4.3):**
  - Compute exact **SentencePiece subword token counts (fertility)** and **character lengths** for all Swadesh and non-Swadesh items across languages.
  - Remove abstract/emotion nouns (*anger, fear, hunger, thirst, dream, harvest, shadow*) and ambiguous homographs (*well*) from `non_swadesh_controlled.json` (or analyze a strictly concrete, length/fertility-matched subset).
  - Report the multiple regression / ANCOVA of convergence on `Swadesh_Status + Subword_Fertility + Word_Length + Ortho_Sim` in both carrier-frame and bare-word regimes (and across cross-script pairs), and revise the claims in Abstract, Intro, Section 4.3, and Discussion accordingly.

- [ ] **3. Cross-Script & Partial Mantel Phylogenetic Tests (Section 4.4):**
  - Replace the invalid i.i.d. `spearmanr` $p$-value ($p = 2.40 \times 10^{-40}$) for the $321$ within-subfamily pairs with a **within-subfamily permutation test** ($9,999$ permutations).
  - Report the within-subfamily correlation restricted to **cross-script language pairs** ($n = 53$, $\rho_s = 0.574$) and a **partial Mantel test controlling for shared script ($S_{\ell,\ell'} \in \{0,1\}$)** across both carrier-frame and bare-word regimes.

- [ ] **4. String-Identity-Filtered & POS-Controlled CLICS³ Colexification Test (Section 4.5):**
  - Fix the text/code discrepancy between **within-language similarity** $\frac{1}{N}\sum_\ell \cos(\mathbf{e}_{a,\ell}, \mathbf{e}_{b,\ell})$ and **cross-lingual centroid similarity** $\cos(\boldsymbol{\mu}_a, \boldsymbol{\mu}_b)$ (report both, including on language-centered bare words!).
  - Exclude all language-pair instances where $w_{a,\ell} = w_{b,\ell}$ (identical translation strings in `swadesh_corpus.json`) or where within-language normalized edit distance $< 0.2$.
  - Report the CLICS³ comparison restricted to **same-POS pairs** ($n=137$ vs. $990$) and **same-category pairs** ($n=92$ vs. $384$).

- [ ] **5. Bare-Word & Length-Controlled Berlin & Kay Color Analysis (Section 4.6):**
  - Fix `generate_stats.py:302` and `generate_figures.py:764, 793` so `grey` is coded as `5.0` (matching `04-results.tex:223`) and plotted with an open diamond marker (matching `04-results.tex:217`), and align the language set to the same 135 languages (fixing `load_color_terms()` in `benchmarks.py:416`).
  - Re-run the color PCA on **decontextualized bare words** (and language-centered bare words) to prove the evolutionary hierarchy is not an artifact of `"I saw a {color} near the river."`
  - Report the exact permutation $p$-value and partial correlation controlling for mean word length / subword fertility.

- [ ] **6. True Language-Centered Perturbation Coupling & Distance/POS-Matched Offset Controls (Section 4.6):**
  - Fix `benchmarks.py:976–977` so per-language shifts $\hat{\boldsymbol{\ell}}$ are subtracted before computing `perturbation_coupling` $\cos(\boldsymbol{\epsilon}_{a,\ell}, \boldsymbol{\epsilon}_{b,\ell})$.
  - Replace or supplement the 7 non-overlapping/mixed-POS controls with **genuinely distance-matched and POS-matched unrelated Swadesh pairs** (same $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|_2$ distribution and same POS pairings as the 15 semantic pairs), remove the missing `("smoke", "leg")` entry from `DEFAULT_OFFSET_PAIRS`, and test both the theoretical concave baseline $D/\sqrt{D^2 + c}$ and bare-word offset consistency.

- [ ] **7. Layer Indexing & Carrier-Slice Cleanup (Section 3.1 & Section 4.1):**
  - Clarify that Layers $0 \dots 11$ in Figure 3 correspond to Transformer Encoder Blocks $1 \dots 12$ (or include `hidden_states[0]` as the true input embedding layer), and report the fallback rate of `_find_word_token_indices`.
