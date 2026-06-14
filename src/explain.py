"""
explain.py
==========
SHAP-based model explainability utilities.

Functions
---------
get_shap_values(model, X)           → shap_values array
plot_summary(shap_values, X, ...)   → SHAP summary (beeswarm) plot
plot_force(shap_values, X, idx, ...)→ SHAP force plot for one prediction
plot_waterfall(shap_values, X, idx) → SHAP waterfall plot
top_features(shap_values, X, n=10) → DataFrame of top-n features by mean |SHAP|
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_shap_explainer(model, X_background):
    """
    Return the appropriate SHAP explainer for the model type.

    - Tree-based models (RandomForest, XGBoost): TreeExplainer (fast, exact)
    - Linear models (LogisticRegression): LinearExplainer
    - Any other: KernelExplainer with background sample
    """
    model_type = type(model).__name__
    logger.info("Creating SHAP explainer for %s ...", model_type)

    if model_type in ("RandomForestClassifier", "XGBClassifier",
                      "GradientBoostingClassifier", "LGBMClassifier"):
        explainer = shap.TreeExplainer(model)
    elif model_type in ("LogisticRegression", "LinearSVC"):
        explainer = shap.LinearExplainer(model, X_background)
    else:
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer  = shap.KernelExplainer(model.predict_proba, background)

    logger.info("Explainer created: %s", type(explainer).__name__)
    return explainer


def get_shap_values(model, X, background=None):
    """
    Compute SHAP values for X.

    Returns
    -------
    shap_values : np.ndarray of shape (n_samples, n_features)
                  For binary classification returns values for class=1.
    explainer   : the fitted SHAP explainer
    """
    if background is None:
        background = X

    explainer   = get_shap_explainer(model, background)
    shap_values = explainer.shap_values(X)

    # For binary tree models shap_values is a list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    logger.info("SHAP values computed: shape=%s", np.array(shap_values).shape)
    return shap_values, explainer


def plot_summary(shap_values, X, title="SHAP Summary Plot",
                 max_display=15, save_path=None):
    """
    Beeswarm summary plot showing global feature importance.
    Each dot is one sample; colour = feature value; x-axis = SHAP value.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=max_display,
                      show=False, plot_type="dot")
    plt.title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved SHAP summary plot → %s", save_path)
    plt.show()


def plot_bar_importance(shap_values, X, title="SHAP Feature Importance (Mean |SHAP|)",
                        max_display=15, save_path=None):
    """Bar chart of mean absolute SHAP values (global importance)."""
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X, plot_type="bar",
                      max_display=max_display, show=False)
    plt.title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved SHAP bar plot → %s", save_path)
    plt.show()


def plot_force(explainer, shap_values, X, idx,
               title="SHAP Force Plot", save_path=None):
    """
    Force plot for a single prediction at index idx.
    Shows which features pushed the prediction above/below the baseline.
    """
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value if hasattr(explainer, "expected_value")
        else explainer.expected_value[1],
        shap_values[idx],
        X.iloc[idx] if hasattr(X, "iloc") else X[idx],
        matplotlib=True,
        show=False,
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved force plot → %s", save_path)
    plt.show()


def plot_waterfall(explainer, shap_values, X, idx,
                   title="SHAP Waterfall Plot", save_path=None):
    """
    Waterfall plot for a single prediction — shows contribution of
    each feature from base value to final prediction.
    """
    feature_names = list(X.columns) if hasattr(X, "columns") else None
    expected_val  = (explainer.expected_value
                     if not isinstance(explainer.expected_value, list)
                     else explainer.expected_value[1])

    sv = shap.Explanation(
        values        = shap_values[idx],
        base_values   = expected_val,
        data          = X.iloc[idx].values if hasattr(X, "iloc") else X[idx],
        feature_names = feature_names,
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(sv, show=False, max_display=15)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved waterfall plot → %s", save_path)
    plt.show()


def top_features(shap_values, X, n=10):
    """
    Return DataFrame of top-n features by mean |SHAP value|.

    Parameters
    ----------
    shap_values : np.ndarray (n_samples, n_features)
    X           : DataFrame with feature names
    n           : number of top features to return

    Returns
    -------
    DataFrame with columns [Feature, Mean_Abs_SHAP]
    """
    feature_names = list(X.columns) if hasattr(X, "columns") \
        else [f"f{i}" for i in range(shap_values.shape[1])]

    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "Feature":       feature_names,
        "Mean_Abs_SHAP": mean_abs,
    }).sort_values("Mean_Abs_SHAP", ascending=False).head(n).reset_index(drop=True)

    logger.info("Top %d features by mean |SHAP|:\n%s", n, df.to_string(index=False))
    return df
