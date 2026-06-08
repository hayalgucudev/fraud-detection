"""
feature_engineering.py
=======================
Standalone feature engineering utilities for Fraud_Data.csv.

All functions operate on a DataFrame and return a new DataFrame.
They are designed to be importable into notebooks and the preprocessing
pipeline alike.
"""

import numpy as np
import pandas as pd


def add_time_since_signup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time elapsed (in seconds) between signup_time and purchase_time.

    A very short time_since_signup (e.g., < 3600 seconds = 1 hour) is a
    strong fraud signal: fraudsters often create accounts and immediately
    abuse them before detection systems flag the account.
    """
    df = df.copy()
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds()
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour_of_day and day_of_week from purchase_time.

    Fraud patterns often cluster at specific hours (late night / early morning)
    and days (weekends when fraud teams are understaffed).
    """
    df = df.copy()
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek  # 0 = Monday
    return df


def add_transaction_velocity(
    df: pd.DataFrame,
    window_seconds: int = 3600,
    col_name: str = "transaction_velocity_1h",
) -> pd.DataFrame:
    """
    Count the number of prior transactions by the same user_id
    within `window_seconds` before the current transaction.

    High velocity (many transactions in a short window) is a classic
    fraud signal — fraudsters exploit stolen credentials rapidly.

    Parameters
    ----------
    df             : DataFrame with user_id and purchase_time columns
    window_seconds : look-back window in seconds (default 3600 = 1 hour)
    col_name       : name of the output column

    Returns
    -------
    DataFrame with the new velocity column appended.
    """
    df = df.copy()
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df = df.sort_values(["user_id", "purchase_time"]).reset_index(drop=True)

    velocities = []
    for _, group in df.groupby("user_id"):
        times = group["purchase_time"].values.astype(np.int64) // 10**9
        result = []
        for i, t in enumerate(times):
            window_start = t - window_seconds
            count = np.sum((times[:i] >= window_start) & (times[:i] < t))
            result.append(int(count))
        velocities.extend(result)

    df[col_name] = velocities
    return df


def encode_categoricals(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One-hot encode specified categorical columns."""
    return pd.get_dummies(df, columns=cols, drop_first=False, dtype=int)


def label_encode_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode the 'country' column.
    OHE is avoided here because country cardinality (~138) would create
    too many sparse binary columns; label encoding preserves the signal
    in a single dense column.
    """
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    le = LabelEncoder()
    df["country_encoded"] = le.fit_transform(df["country"].astype(str))
    return df