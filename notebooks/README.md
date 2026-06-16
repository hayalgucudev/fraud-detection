# Notebooks

Run these notebooks in order for the full end-to-end pipeline.

| Notebook | Purpose |
|---|---|
| `eda-fraud-data.ipynb` | Exploratory analysis of e-commerce fraud data |
| `eda-creditcard.ipynb` | Exploratory analysis of credit card fraud data |
| `feature-engineering.ipynb` | Feature construction, SMOTE, and processed CSV export |
| `modeling.ipynb` | Baseline + ensemble models, cross-validation, model selection |
| `shap-explainability.ipynb` | SHAP global/local plots and business recommendations |

## Figures

Saved plots are written to `figures/` (model metrics, SHAP visualizations).

## Prerequisites

1. Place raw datasets in `data/raw/`.
2. Run `python scripts/run_preprocessing.py` or execute `feature-engineering.ipynb`.
3. After `modeling.ipynb`, model artifacts are saved to `models/` (gitignored).
