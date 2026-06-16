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

## `generate_report_pdf.py`

Builds `REPORT.pdf` from `REPORT.md` with all figures embedded (for final submission).

```bash
python scripts/generate_report_pdf.py
```

Upload `REPORT.pdf` to Google Drive and share a public "Anyone with the link" URL for submission.

