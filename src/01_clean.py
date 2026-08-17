"""
01_clean.py
Cleans the raw UCI Chronic Kidney Disease dataset.

Source: UCI ML Repository, "Chronic Kidney Disease" (Soundarapandian et al.)
https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

Raw data quirks handled here (typical of real clinical EHR exports):
  - Numeric lab values (wc, rc, pcv) stored as strings, some with stray whitespace/tabs
  - Categorical fields with inconsistent formatting: '\tyes', ' yes', 'yes' all mean the same thing
  - Target label has a typo variant: 'ckd\t' instead of 'ckd'
  - Missing values coded as blank / NaN across nearly every column (up to ~40% missing in some labs)
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/kidney_disease.csv"
OUT_PATH = "data/kidney_disease_clean.csv"

# Column groups per the UCI documentation
NUMERIC_TRUE = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo"]
NUMERIC_AS_STRING = ["pcv", "wc", "rc"]  # stored as object dtype due to stray characters
BINARY_YESNO = ["htn", "dm", "cad", "pe", "ane"]
BINARY_OTHER = {
    "rbc": {"normal": 1, "abnormal": 0},
    "pc": {"normal": 1, "abnormal": 0},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "appet": {"good": 1, "poor": 0},
}

def clean_str(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower().replace("\t", "")

def main():
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=["id"])

    # Normalize every object/string column: strip whitespace/tabs, lowercase
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "str":
            df[col] = df[col].apply(clean_str)

    # Force the "numeric but stored as string" columns to numeric
    for col in NUMERIC_AS_STRING:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target: fix the 'ckd\t' typo variant, then binary encode
    df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})

    # Yes/No fields -> binary
    for col in BINARY_YESNO:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # Other categorical fields -> binary via explicit mapping
    for col, mapping in BINARY_OTHER.items():
        df[col] = df[col].map(mapping)

    # Report missingness before imputation (kept for the write-up / EDA)
    missing_report = df.isna().mean().sort_values(ascending=False)
    missing_report.to_csv("data/missingness_report.csv", header=["pct_missing"])

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data: {df.shape}")
    print(f"Target balance:\n{df['classification'].value_counts(dropna=False)}")

if __name__ == "__main__":
    main()
