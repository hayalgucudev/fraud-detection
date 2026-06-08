"""
data_preprocessing.py  (v3 — final fix)
=========================================
Root cause of all 3 errors:
  1. merge_asof requires identical dtypes on both sides.
     The solution: use float64 throughout — convert ip_int to float64
     and cast both lb/ub bounds to float64.  This avoids int64/float64
     mismatch regardless of pandas version.

  2. ip_address in Fraud_Data.csv is stored as float64 large integers
     (e.g. 732758368.79972).  These must be floored, not passed through
     inet_aton.

  3. ip_address in the unit-test fixture is dotted-decimal strings
     (e.g. "192.168.1.1").  These must go through inet_aton and be
     returned as float64.

  4. merge_asof rejects NaN keys — null ip_int rows are separated
     before the merge and rejoined as "Unknown" afterwards.
"""

import argparse
import logging
import os
import struct
import socket

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ── IP conversion ──────────────────────────────────────────────────────────

def ip_to_int(ip_str) -> float:
    """
    Convert an IP address to a float64 integer representation.

    Handles two cases found in this project:
      • Numeric float/int  (Fraud_Data.csv stores IPs as float64, e.g. 732758368.79)
        → floor() and return as float
      • Dotted-decimal string  (unit-test fixtures use "192.168.1.1")
        → inet_aton → unpack → return as float

    Always returns float64 (or np.nan) so merge_asof keys are consistent.
    """
    try:
        # Try numeric path first
        val = float(str(ip_str).strip())
        if np.isnan(val):
            return np.nan
        if val > 65536:
            # Large numeric IP already (real dataset)
            return float(np.floor(val))
        # Small float — fall through to dotted-decimal parse
    except (ValueError, TypeError):
        pass

    try:
        return float(struct.unpack("!I", socket.inet_aton(str(ip_str).strip()))[0])
    except (socket.error, OSError, ValueError):
        return np.nan


def enrich_with_country(fraud_df: pd.DataFrame,
                         ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Range-based IP-to-country lookup using pd.merge_asof.

    Strategy: use float64 for ALL merge keys on both sides.
    This is the only dtype that works with both the real dataset
    (float64 IPs) and the unit-test fixture (string IPs converted
    to float via inet_aton), and avoids the int64/float64 mismatch.
    """
    logger.info("Converting IP addresses to integers ...")
    fraud_df = fraud_df.copy()

    raw = fraud_df["ip_address"]
    if pd.api.types.is_float_dtype(raw) or pd.api.types.is_integer_dtype(raw):
        # Real dataset: already numeric, just floor to remove decimals
        fraud_df["ip_int"] = np.floor(raw.astype(float))
    else:
        # String IPs (test fixture / alternative format)
        fraud_df["ip_int"] = raw.apply(ip_to_int)

    # ip_int must be float64 — convert explicitly
    fraud_df["ip_int"] = fraud_df["ip_int"].astype(float)

    n_null = fraud_df["ip_int"].isna().sum()
    if n_null:
        logger.warning("%d rows have null ip_int — will be labelled Unknown", n_null)

    # ── Prepare lookup table as float64 ──────────────────────────────────
    ip_df = ip_df.copy()
    ip_df["lb"] = ip_df["lower_bound_ip_address"].astype(float)
    ip_df["ub"] = ip_df["upper_bound_ip_address"].astype(float)
    ip_df = ip_df.sort_values("lb").reset_index(drop=True)

    # ── Separate null-ip rows (merge_asof rejects NaN keys) ──────────────
    null_mask  = fraud_df["ip_int"].isna()
    valid_df   = fraud_df[~null_mask].copy().sort_values("ip_int").reset_index(drop=True)
    invalid_df = fraud_df[null_mask].copy()

    # ── merge_asof — both keys are now float64 ────────────────────────────
    merged = pd.merge_asof(
        valid_df,
        ip_df[["lb", "ub", "country"]],
        left_on="ip_int",
        right_on="lb",
        direction="backward",
    )

    # Invalidate matches where ip_int > upper_bound
    outside = merged["ip_int"] > merged["ub"]
    merged.loc[outside, "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")
    merged = merged.drop(columns=["lb", "ub"], errors="ignore")

    # Restore null-ip rows
    invalid_df["country"] = "Unknown"
    result = pd.concat([merged, invalid_df], ignore_index=True)

    n_ok = (result["country"] != "Unknown").sum()
    logger.info("Geolocation: %d/%d matched (%.1f%%), %d unique countries",
                n_ok, len(result), n_ok / len(result) * 100,
                result["country"].nunique())
    return result


# ── Fraud_Data preprocessing ───────────────────────────────────────────────

def preprocess_fraud_data(fraud_path: str, ip_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline for Fraud_Data.csv."""
    logger.info("Loading Fraud_Data from %s ...", fraud_path)
    fraud = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)
    logger.info("Raw Fraud_Data shape: %s", fraud.shape)

    fraud["signup_time"]   = pd.to_datetime(fraud["signup_time"])
    fraud["purchase_time"] = pd.to_datetime(fraud["purchase_time"])

    n_before = len(fraud)
    fraud = fraud.drop_duplicates()
    logger.info("Removed %d duplicates (%d remaining)",
                n_before - len(fraud), len(fraud))

    if fraud.isnull().any().any():
        for col in fraud.select_dtypes(include=[np.number]).columns:
            fraud[col] = fraud[col].fillna(fraud[col].median())
        for col in fraud.select_dtypes(include=["object"]).columns:
            fraud[col] = fraud[col].fillna(fraud[col].mode()[0])
    else:
        logger.info("No missing values detected.")

    fraud = enrich_with_country(fraud, ip_df)

    logger.info("Engineering features ...")
    fraud["time_since_signup"] = (
        fraud["purchase_time"] - fraud["signup_time"]).dt.total_seconds()
    fraud["hour_of_day"] = fraud["purchase_time"].dt.hour
    fraud["day_of_week"] = fraud["purchase_time"].dt.dayofweek

    fraud = fraud.sort_values(["user_id", "purchase_time"]).reset_index(drop=True)

    def count_velocity(group, window_seconds):
        times  = group["purchase_time"].values.astype(np.int64) // 10**9
        result = []
        for i, t in enumerate(times):
            ws = t - window_seconds
            result.append(int(np.sum((times[:i] >= ws) & (times[:i] < t))))
        return pd.Series(result, index=group.index)

    logger.info("Computing 1h velocity ...")
    fraud["transaction_velocity_1h"] = fraud.groupby(
        "user_id", group_keys=False).apply(lambda g: count_velocity(g, 3600))
    logger.info("Computing 24h velocity ...")
    fraud["transaction_velocity_24h"] = fraud.groupby(
        "user_id", group_keys=False).apply(lambda g: count_velocity(g, 86400))

    logger.info("Encoding categoricals ...")
    fraud = pd.get_dummies(fraud, columns=["source", "browser", "sex"],
                            drop_first=False, dtype=int)
    le = LabelEncoder()
    fraud["country_encoded"] = le.fit_transform(fraud["country"].astype(str))

    drop_cols = ["signup_time", "purchase_time", "ip_address", "ip_int",
                 "device_id", "user_id", "country"]
    fraud = fraud.drop(columns=[c for c in drop_cols if c in fraud.columns])

    num_cols = [c for c in ["purchase_value", "age", "time_since_signup",
                             "hour_of_day", "day_of_week",
                             "transaction_velocity_1h", "transaction_velocity_24h",
                             "country_encoded"] if c in fraud.columns]
    fraud[num_cols] = StandardScaler().fit_transform(fraud[num_cols])

    target = fraud.pop("class")
    fraud["class"] = target

    logger.info("Final shape: %s", fraud.shape)
    logger.info("Class dist:\n%s",
                fraud["class"].value_counts(normalize=True).round(4).to_string())
    return fraud


# ── creditcard.csv preprocessing ──────────────────────────────────────────

def preprocess_creditcard(cc_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline for creditcard.csv."""
    logger.info("Loading creditcard.csv from %s ...", cc_path)
    cc = pd.read_csv(cc_path)
    logger.info("Raw shape: %s", cc.shape)

    n_before = len(cc)
    cc = cc.drop_duplicates()
    logger.info("Removed %d duplicates", n_before - len(cc))

    if cc.isnull().any().any():
        for col in cc.select_dtypes(include=[np.number]).columns:
            cc[col] = cc[col].fillna(cc[col].median())
    else:
        logger.info("No missing values.")

    cc[["Amount", "Time"]] = StandardScaler().fit_transform(cc[["Amount", "Time"]])

    target = cc.pop("Class")
    cc["Class"] = target

    logger.info("Final shape: %s", cc.shape)
    logger.info("Class dist:\n%s",
                cc["Class"].value_counts(normalize=True).round(4).to_string())
    return cc


# ── Imbalance handling ─────────────────────────────────────────────────────

def apply_smote(X_train, y_train, random_state=RANDOM_STATE):
    """Apply SMOTE to training set only."""
    logger.info("SMOTE before: %s",
                dict(zip(*np.unique(y_train, return_counts=True))))
    X_res, y_res = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
    logger.info("SMOTE after:  %s",
                dict(zip(*np.unique(y_res, return_counts=True))))
    return X_res, y_res


def apply_undersampling(X_train, y_train,
                         sampling_strategy=0.5, random_state=RANDOM_STATE):
    """Apply RandomUnderSampler to training set only."""
    logger.info("UnderSampler before: %s",
                dict(zip(*np.unique(y_train, return_counts=True))))
    X_res, y_res = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state).fit_resample(X_train, y_train)
    logger.info("UnderSampler after:  %s",
                dict(zip(*np.unique(y_res, return_counts=True))))
    return X_res, y_res


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraud",  default="data/raw/Fraud_Data.csv")
    parser.add_argument("--ip",     default="data/raw/IpAddress_to_Country.csv")
    parser.add_argument("--cc",     default="data/raw/creditcard.csv")
    parser.add_argument("--outdir", default="data/processed/")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    preprocess_fraud_data(args.fraud, args.ip).to_csv(
        os.path.join(args.outdir, "fraud_data_processed.csv"), index=False)
    preprocess_creditcard(args.cc).to_csv(
        os.path.join(args.outdir, "creditcard_processed.csv"), index=False)
    logger.info("Done.")