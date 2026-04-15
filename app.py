"""
app.py — HealthRisk_AI Dashboard Backend
==========================================
Flask server providing API endpoints for the ML pipeline
and serving the frontend dashboard.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')

# Ensure src/ modules are importable
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

app = Flask(__name__)
CORS(app)

# ── Global State ──
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "health_data.csv")
pipeline_artifacts = {}
pipeline_status = {"step": 0, "total": 5, "message": "Not started", "running": False}


# ──────────────────────────────────────────────
# Helper: Convert numpy types for JSON
# ──────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


app.json_encoder = NumpyEncoder


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data-summary")
def data_summary():
    """Return dataset statistics."""
    try:
        df = pd.read_csv(DATA_PATH)
        numeric_df = df.select_dtypes(include=[np.number])

        summary = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "features": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": {col: int(v) for col, v in df.isnull().sum().items()},
            "stats": {},
            "target_distribution": df["risk_level"].value_counts().to_dict(),
        }

        for col in numeric_df.columns:
            summary["stats"][col] = {
                "mean": round(float(df[col].mean()), 2),
                "std": round(float(df[col].std()), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "median": round(float(df[col].median()), 2),
            }

        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/eda")
def eda_data():
    """Return EDA visualisation data."""
    try:
        df = pd.read_csv(DATA_PATH)
        numeric_df = df.select_dtypes(include=[np.number])

        # Correlation matrix
        corr = numeric_df.corr().round(3)
        correlation = {
            "labels": corr.columns.tolist(),
            "values": corr.values.tolist(),
        }

        # Distributions (histogram bins)
        distributions = {}
        for col in numeric_df.columns:
            counts, bin_edges = np.histogram(df[col].dropna(), bins=10)
            distributions[col] = {
                "counts": counts.tolist(),
                "bins": [round(float(b), 2) for b in bin_edges],
            }

        # Box-plot data by risk level
        box_data = {}
        key_features = ["age", "bmi", "blood_pressure_systolic", "cholesterol", "blood_glucose"]
        for feat in key_features:
            if feat in df.columns:
                box_data[feat] = {}
                for level in ["Low", "Medium", "High"]:
                    subset = df[df["risk_level"] == level][feat].dropna()
                    box_data[feat][level] = {
                        "q1": round(float(subset.quantile(0.25)), 2),
                        "median": round(float(subset.median()), 2),
                        "q3": round(float(subset.quantile(0.75)), 2),
                        "min": round(float(subset.min()), 2),
                        "max": round(float(subset.max()), 2),
                        "values": subset.tolist(),
                    }

        # Gender distribution
        gender_dist = df["gender"].value_counts().to_dict() if "gender" in df.columns else {}

        return jsonify({
            "correlation": correlation,
            "distributions": distributions,
            "box_data": box_data,
            "gender_distribution": gender_dist,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


import threading

def pipeline_worker():
    global pipeline_artifacts, pipeline_status
    try:
        pipeline_status = {"step": 0, "total": 5, "message": "Starting...", "running": True}
        results = {}

        # Step 1: Preprocessing
        pipeline_status["step"] = 1
        pipeline_status["message"] = "Preprocessing data..."
        from importlib import import_module
        preprocessing = import_module("01_preprocessing")
        pipeline_artifacts = preprocessing.run_preprocessing(DATA_PATH)
        results["preprocessing"] = {
            "train_size": int(pipeline_artifacts["X_train"].shape[0]),
            "test_size": int(pipeline_artifacts["X_test"].shape[0]),
            "features": int(pipeline_artifacts["X_train"].shape[1]),
        }

        # Step 2: Outlier Detection & Clustering
        pipeline_status["step"] = 2
        pipeline_status["message"] = "Detecting outliers & clustering..."
        outlier_mod = import_module("02_outlier_clustering")
        pipeline_artifacts = outlier_mod.run_outlier_clustering(pipeline_artifacts)
        results["clustering"] = {
            "kmeans_k": int(pipeline_artifacts["kmeans"]["k"]),
            "dbscan_clusters": int(pipeline_artifacts["dbscan"]["n_clusters"]),
        }

        # Step 3: Tuning & Ensembles
        pipeline_status["step"] = 3
        pipeline_status["message"] = "Tuning models & building ensembles..."
        tuning_mod = import_module("03_tuning_ensemble")
        pipeline_artifacts = tuning_mod.run_tuning_ensemble(pipeline_artifacts)

        # Step 4: ANN Models
        pipeline_status["step"] = 4
        pipeline_status["message"] = "Training neural networks..."
        ann_mod = import_module("04_ann_models")
        pipeline_artifacts = ann_mod.run_ann_models(pipeline_artifacts)

        # Step 5: Final Evaluation
        pipeline_status["step"] = 5
        pipeline_status["message"] = "Final evaluation..."
        final_mod = import_module("05_final_pipeline")
        pipeline_artifacts = final_mod.run_final_pipeline(pipeline_artifacts)

        results["best_model"] = pipeline_artifacts.get("best_name", "N/A")
        results["comparison"] = pipeline_artifacts["comparison"].to_dict(orient="records") if "comparison" in pipeline_artifacts else []

        pipeline_status = {"step": 5, "total": 5, "message": "Complete!", "running": False, "results": results}

    except Exception as e:
        pipeline_status = {"step": 0, "total": 5, "message": f"Error: {str(e)}", "running": False}

@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline():
    """Run the full ML pipeline in the background and return 202."""
    if pipeline_status["running"]:
        return jsonify({"status": "already_running"}), 400
    
    thread = threading.Thread(target=pipeline_worker)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "started"}), 202


@app.route("/api/pipeline-status")
def get_pipeline_status():
    return jsonify(pipeline_status)


@app.route("/api/model-results")
def model_results():
    """Return model comparison data if pipeline has been run."""
    if "comparison" not in pipeline_artifacts:
        return jsonify({"error": "Pipeline not yet run. Click 'Run Pipeline' first."}), 400

    comparison = pipeline_artifacts["comparison"].to_dict(orient="records")

    # Feature importance
    feature_importance = {}
    tuned = pipeline_artifacts.get("tuned_models", {})
    X_test = pipeline_artifacts["X_test"]
    feature_names = list(X_test.columns) if hasattr(X_test, "columns") else [f"f{i}" for i in range(X_test.shape[1])]

    for name, info in tuned.items():
        if info and "model" in info and hasattr(info["model"], "feature_importances_"):
            imp = info["model"].feature_importances_
            feature_importance[name] = {
                "features": feature_names,
                "importances": [round(float(x), 4) for x in imp],
            }

    return jsonify({
        "comparison": comparison,
        "feature_importance": feature_importance,
        "best_model": pipeline_artifacts.get("best_name", "N/A"),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Make a prediction using the best trained model."""
    if "best_model" not in pipeline_artifacts:
        return jsonify({"error": "Pipeline not yet run."}), 400

    try:
        data = request.json
        feature_order = list(pipeline_artifacts["X_train"].columns)

        # Build input array
        input_data = {}
        for feat in feature_order:
            if feat in data:
                input_data[feat] = float(data[feat])
            else:
                input_data[feat] = 0.0

        input_df = pd.DataFrame([input_data], columns=feature_order)

        # Scale
        scaler = pipeline_artifacts["scaler"]
        input_scaled = scaler.transform(input_df)

        # Predict
        model = pipeline_artifacts["best_model"]
        prediction = model.predict(input_scaled)[0]

        # Map back to label
        encoders = pipeline_artifacts.get("encoders", {})
        risk_encoder = encoders.get("risk_level", None)
        if risk_encoder:
            label = risk_encoder.inverse_transform([int(prediction)])[0]
        else:
            label_map = {0: "High", 1: "Low", 2: "Medium"}
            label = label_map.get(int(prediction), str(prediction))

        # Probabilities if available
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            if risk_encoder:
                for i, cls in enumerate(risk_encoder.classes_):
                    probabilities[cls] = round(float(probs[i]), 4)
            else:
                for i, p in enumerate(probs):
                    probabilities[f"Class {i}"] = round(float(p), 4)

        return jsonify({
            "prediction": label,
            "probabilities": probabilities,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
