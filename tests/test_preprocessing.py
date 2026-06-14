import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (apply_smote, apply_undersampling,
                                    enrich_with_country, ip_to_int,
                                    preprocess_creditcard)
from src.feature_engineering import (add_temporal_features,
                                     add_time_since_signup,
                                     add_transaction_velocity,
                                     encode_categoricals, label_encode_country)


@pytest.fixture
def sample_fraud():
    """Minimal synthetic Fraud_Data-style DataFrame with string IPs."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "signup_time": ["2024-01-01 08:00:00"] * 5,
            "purchase_time": [
                "2024-01-01 09:00:00",
                "2024-01-01 09:30:00",
                "2024-01-02 10:00:00",
                "2024-01-02 10:15:00",
                "2024-01-03 12:00:00",
            ],
            "purchase_value": [50.0, 120.0, 30.0, 80.0, 200.0],
            "device_id": ["D1", "D1", "D2", "D2", "D3"],
            "source": ["SEO", "Ads", "SEO", "Direct", "Ads"],
            "browser": ["Chrome", "Firefox", "Chrome", "Safari", "Chrome"],
            "sex": ["M", "M", "F", "F", "M"],
            "age": [25, 25, 32, 32, 45],
            "ip_address": [
                "192.168.1.1",
                "10.0.0.1",
                "8.8.8.8",
                "1.1.1.1",
                "172.16.0.1",
            ],
            "class": [0, 1, 0, 0, 1],
        }
    )


@pytest.fixture
def sample_ip_df():
    """
    Minimal IP-to-country mapping.
    Both bounds stored as float64 — matches the real IpAddress_to_Country.csv
    format and our enrich_with_country() implementation.
    """
    return pd.DataFrame(
        {
            # Use inet_aton-derived integers expressed as floats, matching
            # lower_bound_ip_address dtype in the real CSV (float64).
            "lower_bound_ip_address": [0.0, 167772160.0, 134744072.0],
            "upper_bound_ip_address": [167772159.0, 184549375.0, 134744072.0],
            "country": ["Unknown", "US", "US"],
        }
    )


@pytest.fixture
def sample_fraud_numeric():
    """Fraud DataFrame with numeric float IP addresses (real dataset format)."""
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "signup_time": ["2024-01-01 08:00:00", "2024-01-01 08:00:00"],
            "purchase_time": ["2024-01-01 09:00:00", "2024-01-02 10:00:00"],
            "purchase_value": [50.0, 30.0],
            "device_id": ["D1", "D2"],
            "source": ["SEO", "Ads"],
            "browser": ["Chrome", "Firefox"],
            "sex": ["M", "F"],
            "age": [25, 32],
            "ip_address": [167772161.0, 134744072.0],  # numeric float IPs
            "class": [0, 1],
        }
    )


@pytest.fixture
def sample_cc():
    """Minimal creditcard-style DataFrame."""
    np.random.seed(42)
    n = 100
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n)
    data["Amount"] = np.abs(np.random.randn(n) * 100)
    data["Class"] = np.where(np.random.rand(n) < 0.05, 1, 0)
    return pd.DataFrame(data)


def test_ip_to_int_valid():
    """Dotted-decimal strings must parse to correct float integer."""
    assert ip_to_int("0.0.0.0") == pytest.approx(0.0)
    assert ip_to_int("255.255.255.255") == pytest.approx(4294967295.0)
    assert ip_to_int("192.168.1.1") == pytest.approx(3232235777.0)


def test_ip_to_int_invalid_returns_nan():
    result = ip_to_int("not_an_ip")
    assert result is np.nan or (isinstance(result, float) and np.isnan(result))


def test_ip_to_int_empty_string():
    result = ip_to_int("")
    assert result is np.nan or (isinstance(result, float) and np.isnan(result))


def test_enrich_with_country_adds_column(sample_fraud, sample_ip_df):
    result = enrich_with_country(sample_fraud, sample_ip_df)
    assert "country" in result.columns


def test_enrich_with_country_no_nulls(sample_fraud, sample_ip_df):
    result = enrich_with_country(sample_fraud, sample_ip_df)
    assert result["country"].isnull().sum() == 0


def test_enrich_with_country_row_count_preserved(sample_fraud, sample_ip_df):
    result = enrich_with_country(sample_fraud, sample_ip_df)
    assert len(result) == len(sample_fraud)


def test_time_since_signup_column_created(sample_fraud):
    result = add_time_since_signup(sample_fraud)
    assert "time_since_signup" in result.columns


def test_time_since_signup_non_negative(sample_fraud):
    result = add_time_since_signup(sample_fraud)
    assert (result["time_since_signup"] >= 0).all()


def test_time_since_signup_value_correct(sample_fraud):
    result = add_time_since_signup(sample_fraud)
    # user 1 first purchase: 09:00 - 08:00 = 3600 seconds
    assert result.loc[0, "time_since_signup"] == pytest.approx(3600.0)


def test_temporal_features_columns_exist(sample_fraud):
    result = add_temporal_features(sample_fraud)
    assert "hour_of_day" in result.columns
    assert "day_of_week" in result.columns


def test_hour_of_day_range(sample_fraud):
    result = add_temporal_features(sample_fraud)
    assert result["hour_of_day"].between(0, 23).all()


def test_day_of_week_range(sample_fraud):
    result = add_temporal_features(sample_fraud)
    assert result["day_of_week"].between(0, 6).all()


def test_velocity_column_created(sample_fraud):
    result = add_transaction_velocity(sample_fraud, window_seconds=3600)
    assert "transaction_velocity_1h" in result.columns


def test_velocity_non_negative(sample_fraud):
    result = add_transaction_velocity(sample_fraud, window_seconds=3600)
    assert (result["transaction_velocity_1h"] >= 0).all()


def test_velocity_first_transaction_zero(sample_fraud):
    """A user's very first transaction must have velocity = 0."""
    result = add_transaction_velocity(sample_fraud, window_seconds=3600)
    result_sorted = result.sort_values(["user_id", "purchase_time"])
    first_txns = result_sorted.groupby("user_id").first()
    assert (first_txns["transaction_velocity_1h"] == 0).all()


def test_velocity_row_count_preserved(sample_fraud):
    result = add_transaction_velocity(sample_fraud)
    assert len(result) == len(sample_fraud)


def test_encode_categoricals_creates_dummies(sample_fraud):
    result = encode_categoricals(sample_fraud, ["source", "browser"])
    assert any(c.startswith("source_") for c in result.columns)


def test_encode_categoricals_original_dropped(sample_fraud):
    result = encode_categoricals(sample_fraud, ["source"])
    assert "source" not in result.columns


def test_label_encode_country_column_exists(sample_fraud):
    df = sample_fraud.copy()
    df["country"] = ["US", "ET", "NG", "US", "ET"]
    result = label_encode_country(df)
    assert "country_encoded" in result.columns


def test_label_encode_country_integer_dtype(sample_fraud):
    df = sample_fraud.copy()
    df["country"] = ["US", "ET", "NG", "US", "ET"]
    result = label_encode_country(df)
    assert pd.api.types.is_integer_dtype(result["country_encoded"])


def test_creditcard_class_last(sample_cc, tmp_path):
    path = str(tmp_path / "cc.csv")
    sample_cc.to_csv(path, index=False)
    result = preprocess_creditcard(path)
    assert result.columns[-1] == "Class"


def test_creditcard_no_missing(sample_cc, tmp_path):
    path = str(tmp_path / "cc.csv")
    sample_cc.to_csv(path, index=False)
    result = preprocess_creditcard(path)
    assert result.isnull().sum().sum() == 0


def test_creditcard_amount_scaled(sample_cc, tmp_path):
    """After StandardScaler, Amount mean should be near zero."""
    path = str(tmp_path / "cc.csv")
    sample_cc.to_csv(path, index=False)
    result = preprocess_creditcard(path)
    assert abs(result["Amount"].mean()) < 1.0


def test_smote_balances_classes():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.array([0] * 180 + [1] * 20)
    X_res, y_res = apply_smote(X, y)
    counts = dict(zip(*np.unique(y_res, return_counts=True)))
    assert counts[0] == counts[1]


def test_smote_increases_minority():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.array([0] * 180 + [1] * 20)
    _, y_res = apply_smote(X, y)
    assert (y_res == 1).sum() > 20


def test_smote_preserves_majority():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.array([0] * 180 + [1] * 20)
    _, y_res = apply_smote(X, y)
    assert (y_res == 0).sum() == 180


# ── Tests: apply_undersampling ────────────────────────────────────────────


def test_undersampling_reduces_majority():
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = np.array([0] * 450 + [1] * 50)
    X_res, y_res = apply_undersampling(X, y, sampling_strategy=0.5)
    assert (y_res == 0).sum() == (y_res == 1).sum() * 2


def test_undersampling_preserves_minority():
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = np.array([0] * 450 + [1] * 50)
    _, y_res = apply_undersampling(X, y, sampling_strategy=0.5)
    assert (y_res == 1).sum() == 50
