# Fraud Detection — Adey Innovations Inc.

**10 Academy KAIM 9 | Week 5 & 6 Challenge**

An end-to-end fraud detection system for e-commerce and bank credit card transactions. The project covers geolocation enrichment, behavioral feature engineering, class imbalance handling, model comparison, and SHAP-based interpretability.

---

## Project Overview

Adey Innovations Inc. serves e-commerce and banking clients who need reliable, real-time fraud detection. This repository builds two independent fraud classification pipelines:

| Dataset | Description | Records | Fraud Rate |
|---|---|---|---|
| `Fraud_Data.csv` | E-commerce transactions with user, device, and behavioral context | ~151,000 | ~9.4% |
| `creditcard.csv` | Bank credit card transactions with PCA-anonymized features | ~284,000 | ~0.17% |

---

## Project Structure

```
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/workflows/
│   └── unittests.yml
├── data/                       # gitignored
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned + feature-engineered
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   ├── data_preprocessing.py   # Cleaning, merging, encoding, SMOTE
│   ├── feature_engineering.py  # Feature construction
│   ├── train.py                # Model training CLI
│   └── explain.py              # SHAP utilities
├── tests/
│   └── test_preprocessing.py
├── scripts/
│   ├── run_preprocessing.py
│   └── README.md
├── models/                     # Saved model artifacts (gitignored)
├── requirements.txt
├── REPORT.md                   # End-to-end project report
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Download datasets and place in `data/raw/`:
- [Fraud_Data.csv](https://www.kaggle.com/datasets/...) — e-commerce transactions
- [IpAddress_to_Country.csv](https://www.kaggle.com/datasets/...) — IP geolocation lookup
- [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — credit card fraud

---

## Usage

### 1. Preprocessing

```bash
python scripts/run_preprocessing.py
```

Outputs `data/processed/fraud_data_processed.csv` and `data/processed/creditcard_processed.csv`.

### 2. Notebooks (recommended workflow)

Run in order:

1. `notebooks/eda-fraud-data.ipynb` — EDA for e-commerce data
2. `notebooks/eda-creditcard.ipynb` — EDA for credit card data
3. `notebooks/feature-engineering.ipynb` — Feature engineering + SMOTE demo
4. `notebooks/modeling.ipynb` — Train and compare models
5. `notebooks/shap-explainability.ipynb` — SHAP analysis and recommendations

### 3. Training (CLI alternative)

```bash
python -m src.train
```

Saves to `models/`:
- `fraud_best_model.pkl`, `cc_best_model.pkl`
- `fraud_X_test.pkl`, `fraud_y_test.pkl`, `cc_X_test.pkl`, `cc_y_test.pkl`

### 4. Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Key Results

### Task 1 — Data & Features

| Item | Fraud_Data | creditcard |
|---|---|---|
| Fraud rate | 9.4% (~14:1) | 0.17% (~577:1) |
| Resampling | SMOTE on train only | SMOTE on train only |
| Top engineered features | `time_since_signup`, `hour_of_day`, `transaction_velocity_1h`, `country` | `Amount`, `Time`, PCA components V1–V28 |
| Geolocation | 138 countries via IP range lookup | N/A |

### Task 2 — Model Performance (test set)

| Dataset | Best Model | PR-AUC | F1 | ROC-AUC |
|---|---|---|---|---|
| Fraud_Data | Random Forest | **0.6267** | **0.7004** | 0.7658 |
| creditcard | Random Forest | **0.7894** | **0.6724** | 0.9805 |

**Selection rationale:** PR-AUC is the primary metric for imbalanced fraud detection. Random Forest achieved the highest PR-AUC and F1 on both held-out test sets, outperforming Logistic Regression and XGBoost despite XGBoost's stronger cross-validation scores.

### Task 3 — SHAP Insights

Top fraud drivers (Fraud_Data): `time_since_signup`, `hour_of_day`, `transaction_velocity_1h`, `country`.

See `REPORT.md` for the full narrative, visualizations, and business recommendations.

---

## Deliverables

| Deliverable | Location |
|---|---|
| Cleaned datasets | `data/processed/` |
| EDA notebooks | `notebooks/eda-*.ipynb` |
| Feature engineering | `notebooks/feature-engineering.ipynb`, `src/` |
| Trained models | `models/` (run `modeling.ipynb` or `src.train`) |
| Evaluation figures | `notebooks/figures/` |
| SHAP plots | `notebooks/figures/shap/` |
| End-to-end report (PDF) | `REPORT.pdf` — generate with `python scripts/generate_report_pdf.py` |

---

## References

- [Kaggle: Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [SHAP Documentation](https://shap.readthedocs.io/)
