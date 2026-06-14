"""
train.py
========
Model building, training, and evaluation for both fraud datasets.

Models trained:
  1. Logistic Regression  (interpretable baseline)
  2. Random Forest        (ensemble)
  3. XGBoost              (gradient boosting)

Evaluation metrics (imbalanced-data appropriate):
  - Precision-Recall AUC
  - F1-Score
  - ROC-AUC
  - Confusion Matrix

Usage
-----
    python src/train.py \
        --fraud data/processed/fraud_data_processed.csv \
        --cc    data/processed/creditcard_processed.csv
"""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_FOLDS      = 5


# ── Helpers ────────────────────────────────────────────────────────────────

def load_dataset(path: str, target: str):
    """Load processed CSV, return X and y."""
    df = pd.read_csv(path)
    drop = [c for c in df.columns if c.startswith("Unnamed")]
    df   = df.drop(columns=drop, errors="ignore")
    X    = df.drop(columns=[target])
    y    = df[target]
    logger.info("Loaded %s: %d rows, %d features, fraud=%.2f%%",
                os.path.basename(path), len(df), X.shape[1], y.mean()*100)
    return X, y


def evaluate(model, X_test, y_test, name=""):
    """Return dict of evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model":       name,
        "roc_auc":     roc_auc_score(y_test, y_prob),
        "pr_auc":      average_precision_score(y_test, y_prob),
        "f1":          f1_score(y_test, y_pred, zero_division=0),
        "y_pred":      y_pred,
        "y_prob":      y_prob,
        "report":      classification_report(y_test, y_pred, zero_division=0),
        "conf_matrix": confusion_matrix(y_test, y_pred),
    }


def cross_validate_model(model, X, y, name=""):
    """5-fold stratified CV, return mean ± std for PR-AUC and F1."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring={"pr_auc": "average_precision", "f1": "f1"},
        n_jobs=-1,
    )
    logger.info(
        "%s CV — PR-AUC: %.4f ± %.4f | F1: %.4f ± %.4f",
        name,
        scores["test_pr_auc"].mean(),  scores["test_pr_auc"].std(),
        scores["test_f1"].mean(),       scores["test_f1"].std(),
    )
    return {
        "pr_auc_mean": scores["test_pr_auc"].mean(),
        "pr_auc_std":  scores["test_pr_auc"].std(),
        "f1_mean":     scores["test_f1"].mean(),
        "f1_std":      scores["test_f1"].std(),
    }


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model → %s", path)


# ── Model definitions ──────────────────────────────────────────────────────

def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=9,   # ~9:1 imbalance for fraud_data
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


# ── Training pipeline ──────────────────────────────────────────────────────

def train_pipeline(X, y, dataset_name: str, model_dir: str):
    """Full train/evaluate pipeline for one dataset."""
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("=" * 60)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train: %d | Test: %d | Train fraud: %.2f%%",
                len(X_train), len(X_test), y_train.mean()*100)

    results  = []
    cv_results = []
    best_model = None
    best_pr_auc = -1
    best_name   = ""

    for name, model in get_models().items():
        logger.info("Training %s ...", name)

        # Fit on full training set
        model.fit(X_train, y_train)

        # Hold-out evaluation
        res = evaluate(model, X_test, y_test, name=name)
        results.append({
            "Model":   name,
            "ROC-AUC": f"{res['roc_auc']:.4f}",
            "PR-AUC":  f"{res['pr_auc']:.4f}",
            "F1":      f"{res['f1']:.4f}",
        })
        logger.info("%s — ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f",
                    name, res["roc_auc"], res["pr_auc"], res["f1"])
        logger.info("\n%s", res["report"])

        # Cross-validation
        cv = cross_validate_model(model, X_train, y_train, name=name)
        cv_results.append({
            "Model":        name,
            "CV PR-AUC":    f"{cv['pr_auc_mean']:.4f} ± {cv['pr_auc_std']:.4f}",
            "CV F1":        f"{cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}",
        })

        # Track best by PR-AUC
        if res["pr_auc"] > best_pr_auc:
            best_pr_auc = res["pr_auc"]
            best_model  = model
            best_name   = name

    # Save best model
    save_model(
        best_model,
        os.path.join(model_dir, f"{dataset_name}_best_model.pkl")
    )

    # Print comparison table
    print(f"\n{'=' * 70}")
    print(f"Results — {dataset_name}")
    print(f"{'=' * 70}")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nCross-Validation Results:")
    print(pd.DataFrame(cv_results).to_string(index=False))
    print(f"\nBest model: {best_name} (PR-AUC = {best_pr_auc:.4f})")
    print(f"{'=' * 70}\n")

    return best_model, best_name, results, cv_results, X_test, y_test


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraud", default="data/processed/fraud_data_processed.csv")
    parser.add_argument("--cc",    default="data/processed/creditcard_processed.csv")
    parser.add_argument("--models", default="models/")
    args = parser.parse_args()

    os.makedirs(args.models, exist_ok=True)

    # Fraud_Data pipeline
    X_f, y_f = load_dataset(args.fraud, target="class")
    train_pipeline(X_f, y_f, "fraud_data", args.models)

    # creditcard pipeline
    X_c, y_c = load_dataset(args.cc, target="Class")
    train_pipeline(X_c, y_c, "creditcard", args.models)
