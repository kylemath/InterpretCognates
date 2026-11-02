# Adversarial Peer Review Report (Round 1): *Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*

**Overall Assessment:** **Major Revision / Reject in Current Form.**  
While the manuscript addresses an important interdisciplinary question and builds a genuinely useful open-source toolkit (`InterpretCognates`), a line-by-line audit of the LaTeX source (`paper/sections/*.tex`), statistical macros (`paper/output/stats.tex`), data-generation pipeline (`paper/scripts/precompute_paper_data.py`, `generate_stats.py`, `generate_figures.py`), backend implementation (`backend/app/modeling.py`, `benchmarks.py`), and companion site (`docs/index.html`) reveals **critical discrepancies between what the paper claims and what the code actually executes**. Several flagship claims rest on code-switched English prompts falsely described as translated sentences, uncorrected embeddings falsely described as isotropy-corrected, negative controls that outperform the test pairs, a statistical test ($t = 15.69, p = 9.99 \times 10^{-29}$) interpreted as its exact opposite, and a misread PCA axis that misses the paper's strongest actual finding.

Below are the detailed reports from the three domain reviewers and the Style/Structure Editor.

---

## 1. Reviewer A — Mechanistic Interpretability & Embedding Geometry Skeptic

### A1. Model Size Contradiction: 3.3B vs. 600M Distilled
* **The Claim:** Section 1 (`01-introduction.tex`, L8) introduces the study's subject as: *"NLLB-200 is a 3.3-billion-parameter encoder–decoder Transformer trained by Meta to translate directly between 200 languages."*
* **The Code:** `backend/app/modeling.py` (L11) hardcodes `MODEL_NAME = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")`, and `docs/data/layerwise_metrics.json` (L2) confirms `"model": "facebook/nllb-200-distilled-600M"`. Only in Section 3.1 (`03-methods.tex`, L6) and Section 5.2 (`05-discussion.tex`, L42) does the text admit the experiments use the **600M distilled student model** (12 encoder layers, $d=1024$), not the 3.3B model (24 encoder layers, $d=2048$).
* **Required Fix:** State explicitly in the Abstract and Introduction that all experiments probe `nllb-200-distilled-600M`. Distillation from a 54.5B MoE / 3.3B dense teacher into a 600M dense student compresses representations and forces all 200 languages through shared dense parameters, which is a key mechanistic detail rather than an afterthought.

### A2. The Carrier Sentence Is *Not* Translated — It Is Code-Switched English
* **The Claim:** Section 3.1 (`03-methods.tex`, L12) states: *"we place each target word in a fixed carrier sentence of the form 'I saw a {word} near the river', translated into each target language."* Section 5.2 (`05-discussion.tex`, L34–36) elaborates at length on what happens *"When translated into languages with different word orders, case systems, or zero-article grammars..."* The companion website (`docs/index.html`, L141–143) makes the identical claim.
* **The Code:** In `backend/app/modeling.py` (L117, L165, L201, L252), `DEFAULT_CONTEXT_TEMPLATE = "I saw a {word} near the river."` and `embed_word_in_context` simply executes:
  ```python
  tokenizer.src_lang = lang
  sentence = context_template.format(word=word)
  encoded = tokenizer(sentence, return_tensors="pt").to(_DEVICE)
  ```
  **The carrier sentence is never translated.** Every non-English word (e.g., Hindi `पानी`, Russian `вода`, Chinese `水`) is spliced directly into an **untranslated English string** (`"I saw a पानी near the river."`), while the tokenizer is given the non-English BCP-47 tag (`tokenizer.src_lang = "hin_Deva"`).
* **Secondary Undisclosed Template:** For Figure 1 (`sample_concept.json`, `"water"` manifold), `precompute_paper_data.py` (L857) uses a *different* untranslated English template (`"I drink {word}."`, referenced in `docs/index.html` L180), which is never mentioned in the paper.
* **Silent Mean-Pooling Fallback Bug:** In `modeling.py` (`_find_word_token_indices`, L120–149; fallback at L181, L228, L264), if the target word's isolated SentencePiece subword IDs do not match an exact contiguous subsequence inside the English sentence (which occurs when SentencePiece merges word-boundary whitespace or punctuation differently in context), the code **silently falls back to `_pool_hidden` over the entire sentence**, averaging the English carrier tokens (`I`, `saw`, `a`, `near`, `the`, `river`) into the target concept's embedding.
* **Required Fix:** Remove all false claims that the carrier sentence was translated into each target language. Describe the procedure accurately as a fixed English syntactic frame (code-switched probing) with subword extraction, report how often `_find_word_token_indices` hits the full-sentence fallback, and interpret the decontextualized baseline (Section 4.6) accordingly.

### A3. Direct Statistical Self-Contradiction in Section 4.6 (Carrier Sentence Robustness)
* **The Claim:** Section 4.6 (`04-results.tex`, L95–97) states: *"A paired $t$-test confirms that the two conditions do not differ significantly in central tendency ($t = \CarrierBaselineTstat{}$, $p = \CarrierBaselineTp{}$). Figure 6 shows that the vast majority of concepts fall near the identity line..."*
* **The Numbers (`stats.tex`, L75–77 & `decontextualized_convergence.json`):**
  * `\CarrierBaselineTstat` $= 15.69$
  * `\CarrierBaselineTp` $= 9.99 \times 10^{-29}$
  * `\CarrierBaselineMeanDiff` $= +0.128$ (contextualized mean $= 0.584$ vs. decontextualized mean $= 0.455$).
* **The Critique:** $t = 15.69, p = 9.99 \times 10^{-29}$ is an **overwhelmingly significant difference in central tendency**, the exact opposite of what the prose claims. Embedding target words inside a shared untranslated English carrier frame systematically inflates cross-lingual cosine similarity by $+0.128$ across almost all 101 concepts (in Figure 6a, virtually every point lies well above the identity line, not on it). What remains invariant is the **relative rank ordering** of concepts (Spearman $\rho = 0.867, p = 1.12 \times 10^{-31}$), not the central tendency.
* **Figure 6b Mismatch:** The caption of Figure 6b (`04-results.tex`, L104) calls it a *"Slopegraph of the top-20 concepts"*, whereas `generate_figures.py` (L1825) actually renders a horizontal dumbbell plot sorted solely by contextualized convergence.

### A4. Silent Mixing of ABTT-Corrected and Raw Uncorrected Embeddings
* **The Claim:** Section 3.2 (`03-methods.tex`, L36–38), Section 4.5 (`04-results.tex`, L89), and `docs/index.html` (L154) state that ABTT isotropy correction ($k=3$) is applied *"throughout our analyses"*, and Section 5.2 (`05-discussion.tex`, L50–54) explicitly warns that raw cosine similarity in NLLB-200 is *"inflated and poorly calibrated... Analyses that rely on raw cosine similarity without isotropy correction should be interpreted with caution."*
* **The Code (`precompute_paper_data.py`):**
  * **Uses ABTT-corrected embeddings ($k=3$):** ONLY Section 4.2/4.4 Swadesh convergence ranking (`swadesh_convergence.json`, $\mu = 0.58$), Section 4.5 Isotropy validation (`isotropy_sensitivity.json`), and Section 4.6 Decontextualized baseline (`decontextualized_convergence.json`).
  * **Uses RAW uncorrected embeddings (`sw_emb_ctx`):**
    1. **Figure 1 / Section 4.1 ("Water" manifold):** `precompute_paper_data.py` (L861) calls `mdl.sentence_similarity_matrix(vecs)` instead of `sentence_similarity_matrix_corrected(vecs)`. That is why the paper caption reports cosine similarities $>0.93$, while `docs/index.html` (Figure 2 caption, L250) falsely labels the exact same raw matrix *"Isotropy-corrected cosine similarity matrix"*!
    2. **Section 4.3 (Swadesh vs. Non-Swadesh & Controlled Non-Swadesh):** `_compute_swadesh_vs_non_swadesh` (L352–376) passes raw `sw_emb` and `nsw_emb` directly to `bm.swadesh_convergence_ranking`. Consequently, the Swadesh mean jumps from $\mathbf{0.58}$ in Section 4.2 to $\mathbf{0.80}$ in Section 4.3 (`\SwadeshCompMean` and `\ControlledSwadeshCompMean`) without a word of explanation!
    3. **Section 4.7 (Layer-wise Emergence):** `_compute_layerwise_metrics` (L517) computes `_cosine_mean_upper` on raw layer activations without ABTT correction, yielding a final-layer convergence of $\mathbf{0.80}$ (`\LayerwiseFinalConv`) that contradicts Section 4.2's $\mathbf{0.58}$.
    4. **Section 4.8 (Phylogenetic Mantel test & Concept Map):** Uses raw `sw_emb_ctx` (L630, L644).
    5. **Section 4.9 (CLICS³ Colexification):** Uses raw `sw_emb_ctx` (L684).
    6. **Section 4.10 (Conceptual Store Metric):** Uses raw `sw_emb_ctx` (L687).
    7. **Section 4.11 (Color Circle):** Uses raw contextual embeddings (`vecs`, L805–822), with neither ABTT nor per-language mean-centering applied before PCA.
    8. **Section 4.12 (Semantic Offset Invariance):** Uses raw `sw_emb_ctx` (L690).
* **Required Fix:** Either apply ABTT consistently where claimed or explicitly state in Methods (and in a master summary table) which experiments operate on ABTT-corrected vectors, which operate on per-language mean-centered vectors, and which operate on raw vectors—and reconcile the $\mu = 0.58$ vs. $\mu = 0.80$ discrepancy across Sections 4.2, 4.3, and 4.7.

### A5. Layer-wise Emergence (Section 4.7): Non-Monotonicity, Trivial Heuristics, and a Fatal Sorting Bug in Figure 7c
* **False Monotonicity Claim:** Section 4.7 (`04-results.tex`, L112) claims: *"semantic convergence increases monotonically from early layers (mean convergence $= 0.35$) to the final layer (mean convergence $= 0.80$)."* Inspection of `docs/data/layerwise_metrics.json` (L46–199) shows convergence is **not** monotonic: it increases from Layer 0 ($0.3474$) to a peak at **Layer 10 ($0.8641$)**, and then **drops sharply at Layer 11 ($0.8022$)**! This final-layer drop (a well-known decoder-preparation phenomenon in encoder–decoder Transformers) is completely ignored in the text.
* **Trivial `argmax(diff)` Heuristic:** `\LayerwiseEmergenceLayer = 1` is computed in `precompute_paper_data.py` (L558–559) as `int(np.argmax(np.diff(conv_means)) + 1)`. Because the largest single-step jump is simply between encoder block 0 ($0.347$) and block 1 ($0.468$), the macro outputs `1`. The prose (`04-results.tex`, L112–114; `06-conclusion.tex`, L6) then claims *"a sharp rise around layer 1... parallels the 'NLP pipeline' effect documented by Tenney et al. (2019)"* and even calls Layer 1 a *"phase transition"* in the Conclusion! Layer 1 is the *second earliest* layer in a 12-layer encoder; claiming that emergence at Layer 1 supports upper-layer semantic abstraction is self-contradictory.
* **Inconsistent CSM Definition and Sample Size:** In Section 4.10 (135 languages), the mean-centered Conceptual Store Metric is $\mathbf{2.69}$ (raw $2.25$). In Section 4.7 (39-language subset, `\LayerwiseFinalCSM`), the final-layer mean-centered CSM is $\mathbf{4.01}$ (raw $3.44$). Beyond the 39- vs. 135-language difference, `_conceptual_store_ratio` in `precompute_paper_data.py` (L430–436) L2-normalizes each language vector *before* averaging to form concept centroids, whereas `conceptual_store_metric` in `benchmarks.py` (L813) averages unnormalized vectors.
* **Fatal Sorting Bug in Figure 7c (`generate_figures.py`, L1941–1945):**
  In `fig_layerwise_trajectory`, concepts are supposed to be sorted by their final-layer convergence:
  ```python
  final_layer = max(layer_idx)  # int 11
  all_concepts = sorted(
      concept_data.keys(),
      key=lambda c: concept_data[c].get(final_layer, 0),
      reverse=True)
  ```
  Because `concept_data` (`concept_trajectories`) is loaded from JSON, its dictionary keys are **strings** (`"0"`, `"1"`, ..., `"11"`), not integers! The script author noticed this 11 lines later when populating the matrix (L1955: `val = concept_data[c].get(li, concept_data[c].get(str(li), 0))`), **but forgot to fix `concept_data[c].get(final_layer, 0)` on L1944!** As a result, `concept_data[c].get(11, 0)` evaluates to `0` for *every single concept*, leaving `all_concepts` in the raw dictionary insertion order of `swadesh_100.json` (which starts with Pronouns: `I`, `you`, `he`, `we`, `who`, `what`, `this`, `that`!). Yet Section 4.7 (L118) and the Figure 7c caption (L123) claim: *"Per-concept convergence heatmap reveals that concrete, perceptually grounded concepts (top rows) achieve high cross-lingual convergence earlier than abstract or polysemous concepts (bottom rows)"*—when the top rows of Figure 7c are literally the pronouns `I`, `you`, `he`, `we`, `who`, `what`!

### A6. Semantic Offset Invariance (Section 4.12): Negative Controls Outperform Semantic Pairs and Are Concealed from the Reader
* **Undisclosed Negative Controls:** In `backend/app/benchmarks.py` (L886–912), `DEFAULT_OFFSET_PAIRS` is explicitly partitioned into two groups:
  * 15 `# --- Semantically related pairs (expected high consistency) ---` (`man--woman`, `one--two`, `I--we`, `sun--moon`, `fire--water`, `big--small`, `hot--cold`, `good--new`, `dog--fish`, `die--kill`, `come--give`, `eye--ear`, `black--white`, `night--sun`, `eat--drink`)
  * 8 `# --- Semantically unrelated pairs (negative controls) ---` (`stone--sleep`, `rain--tooth`, `tree--red`, `leaf--come`, `cloud--bone`, `path--tongue`, `seed--night`, `smoke--leg`; `smoke--leg` drops out because `leg` is not in `swadesh_100.json`, leaving 7 negative controls and 22 total pairs).
  **The manuscript (`04-results.tex`, L213–244) never discloses that 7 of the 22 pairs are random/unrelated negative controls**, instead presenting all 22 pairs as tests of "semantic relationships"!
* **Negative Controls Beat Semantic Pairs:** Inspecting `docs/data/offset_invariance.json` reveals why the negative controls were quietly swept under the rug:
  1. `fire--water`: $0.9369$ (norm $= 14.15$)
  2. `night--sun`: $0.9263$ (norm $= 14.61$)
  3. **`tree--red` (Negative Control): $0.9151$ (norm $= 15.22$)**
  4. **`stone--sleep` (Negative Control): $0.9116$ (norm $= 15.14$)**
  5. `dog--fish`: $0.9035$ (norm $= 13.79$)
  6. **`seed--night` (Negative Control): $0.8905$ (norm $= 14.62$)**
  7. `man--woman`: $0.8902$ (norm $= 9.26$)
  8. **`rain--tooth` (Negative Control): $0.8761$ (norm $= 15.47$)**
  9. **`path--tongue` (Negative Control): $0.8598$ (norm $= 12.38$)**
  10. `sun--moon`: $0.8591$ (norm $= 11.32$)
  11. **`cloud--bone` (Negative Control): $0.8544$ (norm $= 13.38$)**
  12. `big--small`: $0.8460$ (norm $= 9.45$)
  ...
  17. `hot--cold`: $0.7927$ (norm $= 9.19$)
  19. `black--white`: $0.7719$ (norm $= 8.80$)
  22. `I--we`: $0.6963$ (norm $= 6.92$)
  **The 7 unrelated negative controls have a *higher* mean consistency ($0.8638$) than the 15 semantically related pairs ($0.8228$)!**
* **The Geometric Explanation:** For any two concepts $a, b$ whose embeddings in language $\ell$ decompose as $\mathbf{e}_{a,\ell} = \boldsymbol{\mu}_a + \boldsymbol{\ell} + \boldsymbol{\epsilon}_{a,\ell}$, the within-language difference vector cancels the language centroid $\boldsymbol{\ell}$:
  $$\mathbf{d}_\ell(a,b) = \mathbf{e}_{b,\ell} - \mathbf{e}_{a,\ell} = (\boldsymbol{\mu}_b - \boldsymbol{\mu}_a) + (\boldsymbol{\epsilon}_{b,\ell} - \boldsymbol{\epsilon}_{a,\ell}).$$
  Therefore, the cosine similarity between $\mathbf{d}_\ell(a,b)$ and the centroid offset $\bar{\mathbf{d}}(a,b) = \boldsymbol{\mu}_b - \boldsymbol{\mu}_a$ is governed by the signal-to-noise ratio $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\| / \|\boldsymbol{\epsilon}_{b,\ell} - \boldsymbol{\epsilon}_{a,\ell}\|$. Unrelated concepts from distant semantic domains (e.g., `tree--red`, `stone--sleep`, `rain--tooth`) have **much larger between-centroid distance $\|\boldsymbol{\mu}_b - \boldsymbol{\mu}_a\|$** ($14.6\text{--}15.5$) than semantically related pairs that share a domain (`hot--cold` $= 9.19$, `black--white` $= 8.80$, `die--kill` $= 7.98$, `I--we` $= 6.92$). Thus, raw cosine offset consistency measures **first-order cluster separation relative to within-concept dispersion**, *not* second-order analogy structure!
* **Hardcoded Whitelist in `precompute_paper_data.py` (L693–704):** To prevent `tree--red`, `stone--sleep`, and `rain--tooth` from appearing in Figure 12 (`fig_offset_vector_demo.pdf`), `precompute_paper_data.py` explicitly filters `pairs` against a hardcoded `semantic_set` whitelist before selecting `top_k`—and then sorts by `centroid_offset_norm` (L704) rather than `mean_consistency`, taking `[:5]` pairs while the manuscript text (L237) and caption (L242) falsely claim it plots the *"top-4 concept pairs"*.

---

## 2. Reviewer B — Historical Linguistics, Typology & Statistical Rigor Skeptic

### B1. Swadesh vs. Non-Swadesh Comparison (Section 4.3): Four Compounding Methodological Flaws
1. **Unacknowledged Language Drop ($135 \to 39$ languages):** Both `non_swadesh_60.json` and `non_swadesh_controlled.json` contain only 40 languages (39 after excluding `uzb_Latn`). `_compute_swadesh_vs_non_swadesh` (`precompute_paper_data.py`, L354) takes the intersection of language sets, **silently discarding 96 of the 135 languages** (including almost all low-resource African, Austronesian, and Indigenous American languages) and leaving a 39-language sample that is 38.5% Indo-European (15/39). The manuscript never mentions that Section 4.3 is computed on 39 languages rather than 135.
2. **Unacknowledged Raw vs. ABTT-Corrected Switch:** As noted in A4, Section 4.3 computes convergence on raw embeddings (`\SwadeshCompMean` $= 0.80$) immediately after Section 4.2 reported $\mu = 0.58$ on ABTT-corrected embeddings.
3. **Severe Part-of-Speech (POS) and Concreteness Confound:** The 101-concept Swadesh list spans nouns, verbs (`Actions`, corrected $\mu = 0.54$), adjectives (`Properties`, $\mu = 0.57$), numerals, demonstrative/interrogative/personal pronouns (`Pronouns`, $\mu = 0.45$), and negation (`not`). Furthermore, every word is embedded in a nominal slot (`"I saw a {word} near the river."`), which is syntactically anomalous for verbs (`die`, `eat`, `know`), pronouns (`I`, `you`, `we`), and negation (`not`). By contrast, `non_swadesh_controlled.json` (L3) consists of **"60 frequency-matched, non-loanword concrete nouns"** (`hammer`, `needle`, `cow`, `pig`, `hill`, `lake`, etc.). Comparing a multi-POS Swadesh list (weighed down by function words and verbs in a noun frame) against 60 concrete nouns is fatally confounded by POS and concreteness.
4. **Spinning a Null Result ($p = 0.087$):** Even with raw embeddings on 39 languages, the one-sided Mann-Whitney $U$ test for Swadesh ($\mu = 0.80$) vs. Controlled Non-Swadesh ($\mu = 0.78$) is **statistically non-significant** ($U = 3420, p = 0.087, d = 0.23$). Claiming in Section 4.3 (L38) that this is *"consistent with the hypothesis that culturally stable, universally attested concepts develop robust cross-lingual representations"* overstates a marginal null result driven by POS mismatch.
5. **Arbitrary Slicing and Naïve Phonetic Normalization in Surface-Form Regression (`generate_stats.py` L224–257; `generate_figures.py` L204–227):**
   * The corpus contains 84 Latin-script languages (`endswith('_Latn')`), ordered by family in `swadesh_corpus.json`. The scripts slice `sample = forms[:40]`, arbitrarily taking the first 40 Latin-script languages in file order (heavily Romance/Germanic/Slavic), while `docs/index.html` (L346, L374, L400) claims the regression uses *"21 Latin-script languages (210 pairs)"*!
   * Worse, "phonological similarity" (`\DecompRsqPhon`) does not use IPA or ASJP transcriptions—it applies `str.maketrans('bdgvzqcyw', 'pptfskkiu')` directly to Latin-script orthography (`_phonetic_normalize`)! Since ASJP phonetic transcriptions are already loaded in `benchmarks.py` for 88 languages, why use a toy ASCII substitution on Latin orthography and call it "phonological similarity"?

### B2. Phylogenetic Correlation (Section 4.8): Global $\rho = 0.13$ vs. Within-Subfamily $\rho = 0.65$, NLD vs. LDND, and ASJP Saturation
* **NLD Misdescribed as LDND:** `benchmarks.py` (L94, L1060) states that it computes ASJP **LDND** (*Levenshtein Distance Normalized Divided*, Jäger 2018; Wichmann et al. 2010), which divides the mean Normalized Levenshtein Distance (NLD) of synonymous pairs by the mean NLD of *off-diagonal (non-synonymous)* concept pairs to factor out chance phoneme-inventory overlap. However, `compute_asjp_distance_matrix` (`benchmarks.py`, L1131–1179) only computes raw **NLD** (`_levenshtein(fi, fj) / max(len(fi), len(fj))`), omitting the off-diagonal division entirely!
* **Why Overall Mantel $\rho = 0.13$ Is Diluted by Cross-Family Saturation:**
  In `docs/index.html` (L452), the original hypothesis predicted $\rho > 0.3, p < 0.01$. Across all $\binom{88}{2} = 3,828$ language pairs, the global Mantel correlation is modest ($\rho = 0.131, p = 0.020$). However, look at the three phylogenetic tiers plotted in Figure 9 (`fig_mantel_scatter.pdf`):
  * **Same subfamily ($n = 321$ pairs):** Spearman $\boldsymbol{\rho = 0.653}$ ($p = 2.40 \times 10^{-40}$)
  * **Cross-branch Indo-European ($n = 374$ pairs):** Spearman $\rho = 0.097$ ($p = 0.060$, NS)
  * **Cross-family ($n = 3,133$ pairs, $81.8\%$ of all pairs):** Spearman $\rho = 0.036$ ($p = 0.047$)
  In historical linguistics, it is a foundational fact that lexical phonetic distances on the Swadesh list **hit a saturation ceiling beyond ~4,000–6,000 years BP**: unrelated language families have ASJP NLD clustered tightly around $0.89 \pm 0.03$ (noise). Because $81.8\%$ of all pairs in the $88 \times 88$ matrix are unrelated cross-family pairs where ASJP distance has zero phylogenetic resolution, computing a single rank correlation across all $3,828$ pairs buries the massive **within-subfamily genealogical correlation ($\rho = 0.65, p = 2.4 \times 10^{-40}$)**!
* **Prose Contradicts Figure 9:** Section 4.8 (L144–145) claims: *"Same-subfamily pairs (e.g., French–Spanish) cluster at low ASJP distance with tight embedding distances; cross-branch Indo-European pairs (e.g., English–Russian) occupy the middle range; and cross-family pairs (e.g., English–Chinese) dominate the high-distance region. Separate per-group regression lines reveal that the positive trend is driven primarily by the contrast between these tiers rather than a uniform linear relationship."*
  Look at the actual data in Figure 9:
  1. The mean embedding distance for **Cross-Branch IE ($0.242$)** is actually *lower* than for **Same-Subfamily ($0.279$)**! Why? Because "Same-Subfamily" includes 24 Niger-Congo and 17 Austronesian languages (many lower-resource in NLLB, which have higher embedding distances), whereas "Cross-Branch IE" consists mostly of high-resource European languages!
  2. Consequently, the positive trend is **not** driven by a monotonic step across the three tiers ($0.279 \to 0.242 \to 0.290$ is non-monotonic!), but rather by the **strong linear relationship *within* subfamilies ($\rho = 0.65$)**! The text in L144–145 states the exact opposite of what Figure 9 shows.
* **Figure 8 Caption Error:** The caption of Figure 8 (`fig_phylogenetic.pdf`, L135) states that the heatmap and dendrogram show `\MantelNumLangs{}` ($88$) languages, when `fig_phylogenetic` actually plots all **$\mathbf{135}$** languages (`\PhyloNumLanguages{}`).

### B3. CLICS³ Colexification Test (Section 4.9): Only 54 of 101 Swadesh Concepts Mapped + 7 Non-Swadesh Concepts Injected
* **The Claim:** Section 4.9 (`04-results.tex`, L171) states: *"Treating colexification as a continuous variable across all \ColexNumPairs{} [1,431] Swadesh concept pairs..."*
* **The Math & Code Problem:** Our Swadesh corpus has $101$ concepts, which yields $\binom{101}{2} = \mathbf{5,050}$ pairs—not $1,431$! Why does `\ColexNumPairs` equal $1,431 = \binom{54}{2}$?
  Inspecting `_CLICS_GLOSS_TO_SWADESH` in `backend/app/benchmarks.py` (L460–480) reveals two serious bugs:
  1. **47 of the 101 Swadesh concepts are completely omitted from `_CLICS_GLOSS_TO_SWADESH`!** The dictionary only maps **54** of the 101 Swadesh concepts, leaving out 47 Swadesh concepts (`I`, `all`, `ash`, `bird`, `bite`, `burn`, `cold`, `dog`, `dry`, `ear`, `feather`, `fish`, `fly`, `full`, `give`, `grease`, `he`, `head`, `hear`, `hot`, `kill`, `know`, `lie`, `louse`, `man`, `many`, `mountain`, `name`, `not`, `one`, `person`, `round`, `sand`, `say`, `sit`, `stand`, `star`, `swim`, `that`, `this`, `two`, `walk`, `we`, `what`, `who`, `woman`, `you`)—many of which have canonical colexifications in CLICS³ (e.g., `hear`–`know`, `man`–`person`, `woman`–`wife`, `earth`–`sand`, `feather`–`hair`, `kill`–`die`).
  2. **7 non-Swadesh concepts are hardcoded into `_CLICS_GLOSS_TO_SWADESH`:** The dictionary includes `arm`, `leg`, `blue`, `brown`, `grey`, `orange`, and `purple`, which are **not in `swadesh_100.json`**! When `colexification_test` intersects `_SWADESH_CONCEPTS` with `concept_embeddings` (L686–688), those 7 concepts are silently dropped. Yet `docs/index.html` (L755) explicitly cites *"foot"–"leg" at 52 families, "hand"–"arm" at 48 families* as flagship examples in the CLICS³ test, even though neither pair is present in `colexification.json`!
  3. **Unreported Exclusion of 488 Intermediate Pairs in Binary Test & Unweighted Pooled SD:** In the Mann-Whitney $U$ test (`benchmarks.py`, L708–712), `colex_sims` filters for `frequency >= 3` ($n = 65$ pairs) and `non_colex_sims` filters for `frequency == 0` ($n = 878$ pairs), silently dropping the $488$ pairs attested in $1\text{--}2$ language families ($65 + 878 + 488 = 1,431$). Furthermore, `generate_stats.py` (L139) computes Cohen's $d = 0.96$ using an unweighted variance average `np.sqrt((np.var(col) + np.var(ncol)) / 2)` despite a $13.5\times$ sample size imbalance ($n_1=65, n_2=878$).

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