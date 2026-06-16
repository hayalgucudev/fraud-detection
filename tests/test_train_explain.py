import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.explain import top_features
from src.train import evaluate, get_models, load_dataset


def test_get_models_keys():
    assert set(get_models()) == {"LogisticRegression", "RandomForest", "XGBoost"}


def test_evaluate_returns_metrics():
    X, y = make_classification(
        n_samples=100, n_features=4, weights=[0.9, 0.1], random_state=42
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    metrics = evaluate(model, X, y, name="RF")
    assert metrics["pr_auc"] > 0
    assert metrics["model"] == "RF"


def test_load_dataset(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "class": [0, 1]})
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    X, y = load_dataset(str(path), target="class")
    assert list(X.columns) == ["a"]
    assert len(y) == 2


def test_top_features():
    rng = np.random.default_rng(42)
    shap_values = rng.normal(size=(10, 4))
    X = pd.DataFrame(shap_values, columns=[f"f{i}" for i in range(4)])
    df = top_features(shap_values, X, n=2)
    assert len(df) == 2
