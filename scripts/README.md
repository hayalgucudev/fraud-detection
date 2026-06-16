# Scripts

CLI entry points for reproducible pipeline steps.

## `run_preprocessing.py`

Runs the full preprocessing pipeline for both datasets and writes processed CSVs to `data/processed/`.

```bash
python scripts/run_preprocessing.py
python scripts/run_preprocessing.py --fraud data/raw/Fraud_Data.csv --ip data/raw/IpAddress_to_Country.csv --cc data/raw/creditcard.csv --out data/processed/
```

## Training (via `src/train.py`)

```bash
python -m src.train
python -m src.train --fraud data/processed/fraud_data_processed.csv --cc data/processed/creditcard_processed.csv --models models/
```

Saves `fraud_best_model.pkl`, `cc_best_model.pkl`, and test-set pickles used by the SHAP notebook.
