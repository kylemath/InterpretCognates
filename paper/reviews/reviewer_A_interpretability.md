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
