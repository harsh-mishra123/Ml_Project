"""
05_final_pipeline.py
--------------------
Final evaluation, model comparison, feature importance,
and model persistence (save / load).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")


SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


# ──────────────────────────────────────────────
# 1. Model Comparison Table
# ──────────────────────────────────────────────
def compare_models(artifacts: dict, X_test, y_test) -> pd.DataFrame:
    """Build a comparison table of all trained models."""
    rows = []

    # Traditional ML models
    tuned = artifacts.get("tuned_models", {})
    for name, info in tuned.items():
        if info and "model" in info:
            y_pred = info["model"].predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            rows.append({"Model": name.upper(), "Accuracy": acc, "F1 (weighted)": f1})

    # Ensembles
    for ens_name in ["voting", "stacking"]:
        info = artifacts.get(ens_name, {})
        if info and "model" in info:
            y_pred = info["model"].predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            rows.append({"Model": ens_name.upper(), "Accuracy": acc, "F1 (weighted)": f1})

    # ANN models
    ann_results = artifacts.get("ann_results", {})
    for name, info in ann_results.items():
        if "accuracy" in info:
            rows.append(
                {
                    "Model": name.upper(),
                    "Accuracy": info["accuracy"],
                    "F1 (weighted)": info.get("f1", info["accuracy"]),
                }
            )

    df_compare = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    print("\n===== MODEL COMPARISON =====")
    print(df_compare.to_string(index=False))
    return df_compare


# ──────────────────────────────────────────────
# 2. Confusion Matrix Heatmap
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str = "Best Model"):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "..", f"confusion_matrix_{model_name}.png"),
        dpi=120,
    )
    plt.close()
    print(f"  [PLOT] Confusion matrix saved for {model_name}")


# ──────────────────────────────────────────────
# 3. Feature Importance (tree-based models)
# ──────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, model_name: str = "Model"):
    """Plot feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"  [SKIP] {model_name} has no feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[idx], color="teal")
    plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "..", f"feature_importance_{model_name}.png"),
        dpi=120,
    )
    plt.close()
    print(f"  [PLOT] Feature importance saved for {model_name}")


# ──────────────────────────────────────────────
# 4. Save & Load Best Model
# ──────────────────────────────────────────────
def save_model(model, name: str = "best_model"):
    """Persist a model to disk using joblib."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  [SAVE] Model saved → {path}")
    return path


def load_model(name: str = "best_model"):
    """Load a persisted model."""
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    model = joblib.load(path)
    print(f"  [LOAD] Model loaded ← {path}")
    return model


# ──────────────────────────────────────────────
# 5. Select Best Model
# ──────────────────────────────────────────────
def select_best_model(artifacts: dict, X_test, y_test) -> tuple:
    """Find the best model by test accuracy across all trained models."""
    best_name = None
    best_acc = 0.0
    best_model = None

    # Check traditional & ensemble models
    for source_key in ["tuned_models", "voting", "stacking"]:
        source = artifacts.get(source_key, {})
        if source_key == "tuned_models":
            for name, info in source.items():
                if info and "model" in info:
                    y_pred = info["model"].predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    if acc > best_acc:
                        best_acc = acc
                        best_name = name
                        best_model = info["model"]
        else:
            if source and "model" in source:
                y_pred = source["model"].predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_name = source_key
                    best_model = source["model"]

    # Check ANN (sklearn MLP only for save compatibility)
    ann = artifacts.get("ann_results", {})
    mlp_info = ann.get("sklearn_mlp", {})
    if mlp_info.get("accuracy", 0) > best_acc:
        best_acc = mlp_info["accuracy"]
        best_name = "sklearn_mlp"
        best_model = mlp_info["model"]

    print(f"\n  🏆 Best Model: {best_name.upper()}  |  Accuracy: {best_acc:.4f}")
    return best_name, best_model, best_acc


# ──────────────────────────────────────────────
# 6. Final Pipeline
# ──────────────────────────────────────────────
def run_final_pipeline(artifacts: dict) -> dict:
    """Run final evaluation, comparison, and model persistence."""
    print("=" * 60)
    print("  STEP 5 : FINAL EVALUATION & MODEL SELECTION")
    print("=" * 60)

    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    X_test_vals = X_test.values if hasattr(X_test, "values") else X_test
    y_test_vals = y_test.values if hasattr(y_test, "values") else y_test

    # Comparison table
    df_compare = compare_models(artifacts, X_test_vals, y_test_vals)

    # Select best
    best_name, best_model, best_acc = select_best_model(artifacts, X_test_vals, y_test_vals)

    # Confusion matrix for best model
    y_pred_best = best_model.predict(X_test_vals)
    plot_confusion_matrix(y_test_vals, y_pred_best, best_name)

    # Feature importance
    feature_names = list(X_test.columns) if hasattr(X_test, "columns") else [f"f{i}" for i in range(X_test_vals.shape[1])]
    plot_feature_importance(best_model, feature_names, best_name)

    # Save best model + scaler
    save_model(best_model, "best_model")
    save_model(artifacts["scaler"], "scaler")

    print("\n" + "=" * 60)
    print(f"  ✅ PIPELINE COMPLETE — Best: {best_name.upper()} ({best_acc:.4f})")
    print("=" * 60)

    artifacts["best_model"] = best_model
    artifacts["best_name"] = best_name
    artifacts["comparison"] = df_compare
    return artifacts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module

    preprocess = import_module("01_preprocessing")
    tuning = import_module("03_tuning_ensemble")

    artifacts = preprocess.run_preprocessing()
    artifacts = tuning.run_tuning_ensemble(artifacts)
    artifacts = run_final_pipeline(artifacts)
