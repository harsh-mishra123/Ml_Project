"""
main.py — HealthRisk_AI
========================
Master entry point that runs the complete ML pipeline:
  1. Preprocessing (clean, encode, scale, split)
  2. Outlier Detection & Clustering
  3. Hyperparameter Tuning & Ensemble Models
  4. ANN Models (Keras + sklearn MLP)
  5. Final Evaluation & Model Selection
"""

import sys
import os
import time

# Ensure src/ modules are importable
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

from importlib import import_module


def banner(text: str) -> None:
    """Print a styled banner."""
    width = 64
    print("\n" + "╔" + "═" * width + "╗")
    print("║" + text.center(width) + "║")
    print("╚" + "═" * width + "╝\n")


def main():
    start = time.time()
    banner("HealthRisk_AI — Full Pipeline")

    # ── Step 1: Preprocessing ──
    preprocessing = import_module("01_preprocessing")
    artifacts = preprocessing.run_preprocessing()

    # ── Step 2: Outlier Detection & Clustering ──
    outlier_clustering = import_module("02_outlier_clustering")
    artifacts = outlier_clustering.run_outlier_clustering(artifacts)

    # ── Step 3: Hyperparameter Tuning & Ensembles ──
    tuning_ensemble = import_module("03_tuning_ensemble")
    artifacts = tuning_ensemble.run_tuning_ensemble(artifacts)

    # ── Step 4: ANN Models ──
    ann_models = import_module("04_ann_models")
    artifacts = ann_models.run_ann_models(artifacts)

    # ── Step 5: Final Evaluation & Model Selection ──
    final_pipeline = import_module("05_final_pipeline")
    artifacts = final_pipeline.run_final_pipeline(artifacts)

    elapsed = time.time() - start
    banner(f"Pipeline finished in {elapsed:.1f}s")

    print("Saved artifacts:")
    print(f"  • Best model : saved_models/best_model.pkl")
    print(f"  • Scaler     : saved_models/scaler.pkl")
    print(f"  • Plots      : *.png in project root")
    print()


if __name__ == "__main__":
    main()
