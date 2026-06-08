# Fraud Detection — Adey Innovations Inc.

**10 Academy KAIM 9 | Week 5 & 6 Challenge**

An end-to-end fraud detection system for e-commerce and bank credit card transactions. The project handles geolocation enrichment, behavioral feature engineering, class imbalance, and model interpretability with SHAP.

---

## Project Overview

Adey Innovations Inc. serves e-commerce and banking clients who need reliable, real-time fraud detection. This project builds two independent fraud classification pipelines:

| Dataset | Description | Records | Fraud Rate |
|---|---|---|---|
| `Fraud_Data.csv` | E-commerce transactions with user, device, and behavioral context | ~150,000 | ~9% |
| `creditcard.csv` | Bank credit card transactions with PCA-anonymized features | ~285,000 | ~0.17% |

---

## Project Structure

```
fraud-detection/
├── .github/workflows/unittests.yml   # CI — pytest on every push
├── data/                              # gitignored
│   ├── raw/                           # Original datasets
│   └── processed/                    # Cleaned + feature-engineered
├── notebooks/
│   ├── eda-fraud-data.ipynb          # EDA for Fraud_Data.csv
│   ├── eda-creditcard.ipynb          # EDA for creditcard.csv
│   ├── feature-engineering.ipynb     # Feature engineering pipeline
│   ├── modeling.ipynb                # Model training and evaluation
│   └── shap-explainability.ipynb     # SHAP interpretability
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Cleaning, merging, encoding
│   └── feature_engineering.py        # Feature construction
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py         # Unit tests
├── scripts/
│   └── run_preprocessing.py          # CLI runner
├── models/                            # Saved model artifacts
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download datasets and place in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

---

## Usage

```bash
# Run preprocessing pipeline
python scripts/run_preprocessing.py

# Run notebooks in order
jupyter notebook notebooks/eda-fraud-data.ipynb
jupyter notebook notebooks/eda-creditcard.ipynb
jupyter notebook notebooks/feature-engineering.ipynb

# Run tests
pytest tests/ -v
```

---

## Key Results (Interim-1)

- Fraud_Data: 9.4% fraud rate. After SMOTE on training set: balanced 50/50.
- creditcard: 0.17% fraud rate. After SMOTE: balanced training set.
- Top engineered features: `time_since_signup`, `hour_of_day`, `transaction_velocity_1h`, `country`.
- Geolocation enrichment: 138 unique countries identified from IP addresses.

---

## References

- [Kaggle: Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)