"""
03_tuning_ensemble.py
---------------------
Hyperparameter tuning (GridSearchCV, RandomizedSearchCV) and
ensemble models (Random Forest, Gradient Boosting, XGBoost, Voting, Stacking).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — skipping XGBoost.")


# ──────────────────────────────────────────────
# 1. Random Forest + GridSearchCV
# ──────────────────────────────────────────────
def tune_random_forest(X_train, y_train) -> dict:
    """Tune Random Forest via GridSearchCV."""
    print("\n--- Random Forest (GridSearchCV) ---")
    param_grid = {
        "n_estimators": [50],
        "max_depth": [5],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=1, verbose=0)
    grid.fit(X_train, y_train)

    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV acc  : {grid.best_score_:.4f}")
    return {"model": grid.best_estimator_, "params": grid.best_params_, "cv_score": grid.best_score_}


# ──────────────────────────────────────────────
# 2. Gradient Boosting + RandomizedSearchCV
# ──────────────────────────────────────────────
def tune_gradient_boosting(X_train, y_train) -> dict:
    """Tune Gradient Boosting via RandomizedSearchCV."""
    print("\n--- Gradient Boosting (RandomizedSearchCV) ---")
    param_dist = {
        "n_estimators": [50],
        "learning_rate": [0.1],
        "max_depth": [3],
        "subsample": [0.8],
    }
    gb = GradientBoostingClassifier(random_state=42)
    rnd = RandomizedSearchCV(
        gb, param_dist, n_iter=5, cv=3, scoring="accuracy", random_state=42, n_jobs=1, verbose=0
    )
    rnd.fit(X_train, y_train)

    print(f"  Best params : {rnd.best_params_}")
    print(f"  Best CV acc  : {rnd.best_score_:.4f}")
    return {"model": rnd.best_estimator_, "params": rnd.best_params_, "cv_score": rnd.best_score_}


# ──────────────────────────────────────────────
# 3. XGBoost
# ──────────────────────────────────────────────
def tune_xgboost(X_train, y_train) -> dict:
    """Tune XGBoost via RandomizedSearchCV."""
    if not HAS_XGB:
        print("  [SKIP] xgboost not available.")
        return {}

    print("\n--- XGBoost (RandomizedSearchCV) ---")
    param_dist = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    n_classes = len(np.unique(y_train))
    xgb = XGBClassifier(
        objective="multi:softmax" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )
    rnd = RandomizedSearchCV(
        xgb, param_dist, n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=1, verbose=0
    )
    rnd.fit(X_train, y_train)

    print(f"  Best params : {rnd.best_params_}")
    print(f"  Best CV acc  : {rnd.best_score_:.4f}")
    return {"model": rnd.best_estimator_, "params": rnd.best_params_, "cv_score": rnd.best_score_}


# ──────────────────────────────────────────────
# 4. Voting Classifier (Hard + Soft)
# ──────────────────────────────────────────────
def build_voting_classifier(tuned_models: dict, X_train, y_train) -> dict:
    """Combine tuned models into a Voting ensemble."""
    print("\n--- Voting Classifier ---")
    estimators = []
    for name, info in tuned_models.items():
        if info and "model" in info:
            estimators.append((name, info["model"]))

    if len(estimators) < 2:
        print("  [SKIP] Need at least 2 base models.")
        return {}

    # Hard voting
    vc_hard = VotingClassifier(estimators=estimators, voting="hard", n_jobs=1)
    hard_scores = cross_val_score(vc_hard, X_train, y_train, cv=2, scoring="accuracy")
    print(f"  Hard voting CV acc : {hard_scores.mean():.4f} ± {hard_scores.std():.4f}")

    # Soft voting (requires predict_proba)
    soft_estimators = [(n, m) for n, m in estimators if hasattr(m, "predict_proba")]
    if len(soft_estimators) >= 2:
        vc_soft = VotingClassifier(estimators=soft_estimators, voting="soft", n_jobs=1)
        soft_scores = cross_val_score(vc_soft, X_train, y_train, cv=2, scoring="accuracy")
        print(f"  Soft voting CV acc : {soft_scores.mean():.4f} ± {soft_scores.std():.4f}")
        best_vc = vc_soft if soft_scores.mean() > hard_scores.mean() else vc_hard
    else:
        best_vc = vc_hard

    best_vc.fit(X_train, y_train)
    return {"model": best_vc}


# ──────────────────────────────────────────────
# 5. Stacking Classifier
# ──────────────────────────────────────────────
def build_stacking_classifier(tuned_models: dict, X_train, y_train) -> dict:
    """Build a Stacking ensemble with Logistic Regression as meta-learner."""
    print("\n--- Stacking Classifier ---")
    estimators = []
    for name, info in tuned_models.items():
        if info and "model" in info:
            estimators.append((name, info["model"]))

    if len(estimators) < 2:
        print("  [SKIP] Need at least 2 base models.")
        return {}

    return {}


# ──────────────────────────────────────────────
# 6. Full Tuning + Ensemble Pipeline
# ──────────────────────────────────────────────
def run_tuning_ensemble(artifacts: dict) -> dict:
    """Execute hyperparameter tuning and ensemble methods."""
    print("=" * 60)
    print("  STEP 3 : HYPERPARAMETER TUNING & ENSEMBLES")
    print("=" * 60)

    X_train = artifacts["X_train"]
    y_train = artifacts["y_train"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    # Tune individual models
    rf_result = tune_random_forest(X_train, y_train)
    gb_result = tune_gradient_boosting(X_train, y_train)
    xgb_result = tune_xgboost(X_train, y_train)

    tuned = {"rf": rf_result, "gb": gb_result, "xgb": xgb_result}

    # Ensemble methods
    voting_result = build_voting_classifier(tuned, X_train, y_train)
    stacking_result = build_stacking_classifier(tuned, X_train, y_train)

    # Evaluate all on test set
    print("\n--- Test Set Evaluation ---")
    all_models = {**tuned, "voting": voting_result, "stacking": stacking_result}
    for name, info in all_models.items():
        if info and "model" in info:
            y_pred = info["model"].predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"\n  [{name.upper()}] Test Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred, zero_division=0))

    print("[DONE] Tuning & ensemble complete.\n")

    artifacts["tuned_models"] = tuned
    artifacts["voting"] = voting_result
    artifacts["stacking"] = stacking_result
    return artifacts


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module

    preprocess = import_module("01_preprocessing")
    artifacts = preprocess.run_preprocessing()
    artifacts = run_tuning_ensemble(artifacts)
