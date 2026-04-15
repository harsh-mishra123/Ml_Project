"""
02_outlier_clustering.py
------------------------
Outlier detection (IQR, Z-score, Isolation Forest) and
unsupervised clustering (K-Means, DBSCAN) for risk profiling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Outlier Detection — IQR Method
# ──────────────────────────────────────────────
def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Flag outliers using the IQR method for every numeric column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_counts[col] = n_outliers

    result = pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["outlier_count"])
    print("[IQR] Outlier counts per feature:")
    print(result[result["outlier_count"] > 0].to_string())
    return result


# ──────────────────────────────────────────────
# 2. Outlier Detection — Z-Score
# ──────────────────────────────────────────────
def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Flag outliers using Z-score > threshold."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    outlier_mask = z_scores > threshold
    outlier_counts = outlier_mask.sum(axis=0)
    result = pd.DataFrame(outlier_counts, columns=["outlier_count"])
    print(f"\n[Z-Score] Outliers (|z| > {threshold}):")
    print(result[result["outlier_count"] > 0].to_string())
    return result


# ──────────────────────────────────────────────
# 3. Outlier Detection — Isolation Forest
# ──────────────────────────────────────────────
def detect_outliers_isolation_forest(
    df: pd.DataFrame, contamination: float = 0.05
) -> np.ndarray:
    """Use Isolation Forest to detect anomalies. Returns -1 for outliers."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(df[numeric_cols])
    n_outliers = (labels == -1).sum()
    print(f"\n[Isolation Forest] Detected {n_outliers} outliers ({contamination*100:.1f}% contamination)")
    return labels


# ──────────────────────────────────────────────
# 4. Cap Outliers (Winsorize)
# ──────────────────────────────────────────────
def cap_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers at IQR boundaries."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        capped = df[col].clip(lower, upper)
        n_changed = (df[col] != capped).sum()
        if n_changed:
            print(f"  [CAP] '{col}': {n_changed} values capped to [{lower:.2f}, {upper:.2f}]")
        df[col] = capped

    return df


# ──────────────────────────────────────────────
# 5. K-Means Clustering
# ──────────────────────────────────────────────
def run_kmeans(X: pd.DataFrame, k_range: range = range(2, 8)) -> dict:
    """Run K-Means for several k values, return best model + labels."""
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)
        print(f"  K={k}  |  Inertia={km.inertia_:.1f}  |  Silhouette={sil:.4f}")

    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"\n  [BEST] K={best_k}  (silhouette={max(sil_scores):.4f})")

    best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    best_labels = best_km.fit_predict(X)

    # Elbow plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "o-")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(list(k_range), sil_scores, "s-", color="green")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "kmeans_elbow.png"), dpi=120)
    plt.close()

    return {"model": best_km, "labels": best_labels, "k": best_k}


# ──────────────────────────────────────────────
# 6. DBSCAN Clustering
# ──────────────────────────────────────────────
def run_dbscan(X: pd.DataFrame, eps: float = 0.8, min_samples: int = 5) -> dict:
    """Run DBSCAN and report cluster statistics."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"\n  [DBSCAN] Clusters: {n_clusters}  |  Noise points: {n_noise}")
    if n_clusters > 1:
        sil = silhouette_score(X, labels)
        print(f"  [DBSCAN] Silhouette: {sil:.4f}")
    return {"model": db, "labels": labels, "n_clusters": n_clusters}


# ──────────────────────────────────────────────
# 7. Full Outlier + Clustering Pipeline
# ──────────────────────────────────────────────
import os


def run_outlier_clustering(artifacts: dict) -> dict:
    """Execute outlier detection and clustering on preprocessed data."""
    print("=" * 60)
    print("  STEP 2 : OUTLIER DETECTION & CLUSTERING")
    print("=" * 60)

    X_train = artifacts["X_train"]
    df_encoded = artifacts["df_encoded"]

    print("\n--- IQR Outlier Detection ---")
    detect_outliers_iqr(df_encoded)

    print("\n--- Z-Score Outlier Detection ---")
    detect_outliers_zscore(df_encoded)

    print("\n--- Isolation Forest ---")
    iso_labels = detect_outliers_isolation_forest(df_encoded)

    print("\n--- Capping Outliers ---")
    df_capped = cap_outliers(df_encoded)

    print("\n--- K-Means Clustering ---")
    km_result = run_kmeans(X_train)

    print("\n--- DBSCAN Clustering ---")
    db_result = run_dbscan(X_train)

    print("\n[DONE] Outlier detection & clustering complete.\n")

    artifacts["df_capped"] = df_capped
    artifacts["kmeans"] = km_result
    artifacts["dbscan"] = db_result
    artifacts["isolation_labels"] = iso_labels

    return artifacts


if __name__ == "__main__":
    from importlib import import_module
    preprocess = import_module("01_preprocessing")
    artifacts = preprocess.run_preprocessing()
    artifacts = run_outlier_clustering(artifacts)
