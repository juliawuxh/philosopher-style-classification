# philosopher-style-classification


This repo contains code and a short paper on cross-work authorship attribution for philosophical texts. I fine-tune a DistilBERT classifier to distinguish between **Immanuel Kant** vs **Friedrich Nietzsche**'s writings and test whether performance persists under **semantic control** (topic modeling + embedding similarity). I also run a small **LIME** interpretability analysis to inspect token-level drivers of predictions.

## What’s in here
- **Data prep**: Gutenberg cleaning + token chunking (125 tokens)
- **Model**: `distilbert-base-uncased` fine-tuned for binary classification
- **Evaluation**
  - H1: cross-work test (held-out books)
  - H2: semantic control
    - BERTopic topic overlap diagnostic + topic-controlled subset
    - cosine similarity control (opposite-author nearest neighbor)
- **Interpretability**: LIME explanations + aggregated feature plots/tables

## Reproducibility (typical workflow)
1. Install dependencies
2. Run preprocessing (download/clean/chunk)
3. Train + evaluate (H1)
4. Run semantic controls (H2)
5. Run LIME analysis (H3)
6. Build figures/tables used in the paper

> Note: Results depend on the exact corpus versions, random seeds, and filtering thresholds.
> Note: Not all notebook cells are executed with saved outputs. Some experiments require GPU access, which may not be available by default on Colab or Kaggle; users are expected to rerun the notebooks in their own environment to reproduce full results.

## Dependencies
- Python 3.8+
- `transformers`, `datasets`, `torch`
- `pandas`, `numpy`, `matplotlib`
- `bertopic`, `sentence-transformers`
- `scikit-learn`
- `lime`

## Outputs
Key outputs are saved as:
- confusion matrix / metrics summaries
- topic diagnostics + topic-controlled subset metrics
- similarity curve figure
- LIME plots + summary tables
