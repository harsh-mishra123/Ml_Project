"""
04_ann_models.py
----------------
Artificial Neural Network models using TensorFlow / Keras.
Includes a simple ANN, a deeper ANN, and comparison with sklearn MLPClassifier.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks

    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[WARN] TensorFlow not installed — using sklearn MLP only.")


# ──────────────────────────────────────────────
# 1. Simple ANN (Keras)
# ──────────────────────────────────────────────
def build_simple_ann(input_dim: int, num_classes: int) -> "keras.Model":
    """Build a simple 2-hidden-layer ANN."""
    if not HAS_TF:
        return None

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# 2. Deep ANN (Keras)
# ──────────────────────────────────────────────
def build_deep_ann(input_dim: int, num_classes: int) -> "keras.Model":
    """Build a deeper ANN with 4 hidden layers."""
    if not HAS_TF:
        return None

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# 3. Train Keras Model
# ──────────────────────────────────────────────
def train_keras_model(
    model, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 16, name: str = "ANN"
):
    """Train a Keras model with early stopping and return history."""
    if model is None:
        return None, None

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )

    print(f"\n  Training {name} ({epochs} max epochs) ...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  {name} — Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.4f}")
    return model, history


# ──────────────────────────────────────────────
# 4. Plot Training History
# ──────────────────────────────────────────────
def plot_training_history(history, name: str = "ANN"):
    """Plot accuracy and loss curves."""
    if history is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"{name} — Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"{name} — Loss")
    axes[1].legend()

    plt.tight_layout()
    import os
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"{name}_training.png"), dpi=120)
    plt.close()


# ──────────────────────────────────────────────
# 5. sklearn MLP Baseline
# ──────────────────────────────────────────────
def train_sklearn_mlp(X_train, y_train, X_test, y_test) -> dict:
    """Train an sklearn MLPClassifier as baseline."""
    print("\n--- sklearn MLPClassifier ---")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  MLP Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"model": mlp, "accuracy": acc}


# ──────────────────────────────────────────────
# 6. Full ANN Pipeline
# ──────────────────────────────────────────────
def run_ann_models(artifacts: dict) -> dict:
    """Execute ANN training pipeline."""
    print("=" * 60)
    print("  STEP 4 : ARTIFICIAL NEURAL NETWORKS")
    print("=" * 60)

    X_train = artifacts["X_train"].values if hasattr(artifacts["X_train"], "values") else artifacts["X_train"]
    X_test = artifacts["X_test"].values if hasattr(artifacts["X_test"], "values") else artifacts["X_test"]
    y_train = artifacts["y_train"].values if hasattr(artifacts["y_train"], "values") else artifacts["y_train"]
    y_test = artifacts["y_test"].values if hasattr(artifacts["y_test"], "values") else artifacts["y_test"]

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    ann_results = {}

    # Simple ANN
    if HAS_TF:
        simple_ann = build_simple_ann(input_dim, num_classes)
        simple_ann, hist1 = train_keras_model(
            simple_ann, X_train, y_train, X_test, y_test, name="SimpleANN"
        )
        plot_training_history(hist1, "SimpleANN")
        if simple_ann:
            y_pred = np.argmax(simple_ann.predict(X_test, verbose=0), axis=1)
            acc = accuracy_score(y_test, y_pred)
            print(f"\n  [SimpleANN] Test Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred, zero_division=0))
            ann_results["simple_ann"] = {"model": simple_ann, "accuracy": acc}

        # Deep ANN
        deep_ann = build_deep_ann(input_dim, num_classes)
        deep_ann, hist2 = train_keras_model(
            deep_ann, X_train, y_train, X_test, y_test, name="DeepANN"
        )
        plot_training_history(hist2, "DeepANN")
        if deep_ann:
            y_pred = np.argmax(deep_ann.predict(X_test, verbose=0), axis=1)
            acc = accuracy_score(y_test, y_pred)
            print(f"\n  [DeepANN] Test Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred, zero_division=0))
            ann_results["deep_ann"] = {"model": deep_ann, "accuracy": acc}

    # sklearn MLP
    mlp_result = train_sklearn_mlp(X_train, y_train, X_test, y_test)
    ann_results["sklearn_mlp"] = mlp_result

    print("\n[DONE] ANN models complete.\n")

    artifacts["ann_results"] = ann_results
    return artifacts


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module

    preprocess = import_module("01_preprocessing")
    artifacts = preprocess.run_preprocessing()
    artifacts = run_ann_models(artifacts)
