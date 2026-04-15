"""
01_preprocessing.py
-------------------
Data loading, cleaning, encoding, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "health_data.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the health dataset from CSV."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# 2. Basic Info & Missing Values
# ──────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """Print basic statistics and missing-value counts."""
    print("\n===== Data Types =====")
    print(df.dtypes)
    print("\n===== Missing Values =====")
    print(df.isnull().sum())
    print("\n===== Descriptive Statistics =====")
    print(df.describe())
    print("\n===== Target Distribution =====")
    print(df["risk_level"].value_counts())


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  [FIX] Filled '{col}' with median ({df[col].median():.2f})")
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [FIX] Filled '{col}' with mode ({mode_val})")
    return df


# ──────────────────────────────────────────────
# 3. Encoding Categorical Features
# ──────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns the encoded DataFrame and a dict of {col: LabelEncoder}.
    """
    df = df.copy()
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Keep 'risk_level' encoding separate (target)
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  [ENC] '{col}' → classes: {list(le.classes_)}")

    return df, encoders


# ──────────────────────────────────────────────
# 4. Feature Scaling
# ──────────────────────────────────────────────
def scale_features(
    X: pd.DataFrame, scaler: StandardScaler = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric features (mean=0, std=1)."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler


# ──────────────────────────────────────────────
# 5. Train / Test Split
# ──────────────────────────────────────────────
def split_data(
    df: pd.DataFrame, target: str = "risk_level", test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Split into X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# 6. Full Preprocessing Pipeline
# ──────────────────────────────────────────────
def run_preprocessing(path: str = DATA_PATH):
    """Execute the complete preprocessing pipeline and return artifacts."""
    print("=" * 60)
    print("  STEP 1 : DATA PREPROCESSING")
    print("=" * 60)

    df = load_data(path)
    inspect_data(df)

    print("\n--- Handling Missing Values ---")
    df = handle_missing_values(df)

    print("\n--- Encoding Categorical Features ---")
    df, encoders = encode_features(df)

    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n--- Scaling Features ---")
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled, _ = scale_features(X_test, scaler)

    print("\n[DONE] Preprocessing complete.\n")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "df_encoded": df,
    }


if __name__ == "__main__":
    artifacts = run_preprocessing()
    print("Returned keys:", list(artifacts.keys()))
