# Interpret Cognates

Interactive web applet for probing how multilingual translation networks represent equivalent concepts across languages.

## Why this project

- Compare semantic neighborhoods for translated concepts (shared vs language-specific structure).
- Visualize multilingual representation geometry ("manifold clouds").
- Inspect source-target token relationships via cross-attention.
- Build a foundation for historical-linguistic hypotheses (for example, potential broad Indo-European regularities).

## Model choice

This app uses **Meta NLLB-200** (open source) through Hugging Face Transformers.

- Highest quality option: `facebook/nllb-200-3.3B` (requires strong GPU/CPU RAM).
- Practical default in code: `facebook/nllb-200-distilled-600M` (faster local iteration).
- Better quality midpoint: `facebook/nllb-200-distilled-1.3B`.

Set the model with:

```bash
export NLLB_MODEL=facebook/nllb-200-distilled-1.3B
```

## Current MVP features

- Translate a source concept-in-context to multiple target languages.
- Compute multilingual sentence embeddings from NLLB encoder states.
- Project embeddings to 2D with PCA for concept-cloud visualization.
- Compute pairwise cosine similarity matrix across languages.
- Visualize source-target cross-attention heatmaps per target language.

## Run locally

From repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

`http://127.0.0.1:8000`

## NLLB language code examples

- English: `eng_Latn`
- Spanish: `spa_Latn`
- French: `fra_Latn`
- German: `deu_Latn`
- Italian: `ita_Latn`
- Portuguese: `por_Latn`
- Russian: `rus_Cyrl`
- Arabic: `arb_Arab`
- Hindi: `hin_Deva`
- Chinese (Simplified): `zho_Hans`

## Suggested roadmap

1. Add a small curated concept benchmark (Swadesh-like + abstract terms).
2. Add neighborhood analysis (nearest tokens/phrases per language).
3. Add trajectory view (concept shifts across multiple context prompts).
4. Add optional UMAP projection and clustering.
5. Add hypothesis testing notebook for proto-language structure signals.
