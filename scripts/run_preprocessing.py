
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import preprocess_fraud_data, preprocess_creditcard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraud", default="data/raw/Fraud_Data.csv")
    parser.add_argument("--ip",    default="data/raw/IpAddress_to_Country.csv")
    parser.add_argument("--cc",    default="data/raw/creditcard.csv")
    parser.add_argument("--out",   default="data/processed/")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fraud = preprocess_fraud_data(args.fraud, args.ip)
    fraud.to_csv(os.path.join(args.out, "fraud_data_processed.csv"), index=False)

    cc = preprocess_creditcard(args.cc)
    cc.to_csv(os.path.join(args.out, "creditcard_processed.csv"), index=False)

    print("Preprocessing complete.")