# 🧬 HealthRisk AI
**Precision Clinical Intelligence & Machine Learning Pipeline Platform**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%x-orange?logo=scikit-learn&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Plugin-38B2AC?logo=tailwind-css&logoColor=white)
![Render](https://img.shields.io/badge/Deployed_on-Render-teal?logo=render&logoColor=white)

HealthRisk AI is a premium, clinical-grade patient risk stratification dashboard. It seamlessly binds a high-fidelity, interactive glassmorphism UI with a highly robust **5-step automated machine learning pipeline**. The platform automatically cleans data, detects clinical outliers, tunes hyperparameters natively, and generates ensembles to pinpoint a patient's exact risk level based on multidimensional health data.

---

## ✨ Key Features
- 🏎️ **Automated ML Pipeline:** 1-click execution straight from the browser that runs real-time preprocessing, clustering, and ensemble tuning.
- 🩺 **Live Patient Inference:** Allows doctors/users to type in custom clinical metrics and instantly receive risk predictions using the winning model from the pipeline.
- 📊 **Visual Analytics Engine:** Built-in exploratory data analysis (EDA) using `Chart.js` for dynamic distribution rendering, correlation matrices, and model feature importance.
- 🎨 **Cinematic Glassmorphism UI:** Professionally designed dark-theme interface utilizing Tailwind CSS, responsive container queries, and the modern `Manrope` font map.

---

## 🏗️ Architecture & Structure

The codebase strictly adheres to a modular design, decoupling the dataset, the ML algorithms, the API, and the static frontend.

```text
HealthRisk_AI/
│
├── app.py                     # Flask WSGI Server & API Router
├── main.py                    # Standalone CLI ML Pipeline Executor
├── Procfile                   # Cloud Deployment Config (Render/Heroku)
├── requirements.txt           # Python Package Dependencies
├── .python-version            # Enforced Environment Targeting (3.11.9)
├── .gitignore                 # Version Control Ignore Rules
│
├── data/
│   └── health_data.csv        # Core clinical multidimensional dataset
│
├── saved_models/
│   ├── best_model.pkl         # Auto-generated winning model (e.g. Random Forest)
│   └── scaler.pkl             # Standard scaler memory for inference normalization
│
├── src/                       # Modular Machine Learning Pipeline
│   ├── 01_preprocessing.py        # Data cleaning, encoding, and scaling
│   ├── 02_outlier_clustering.py   # DBSCAN, K-Means & Isolation Forest anomaly filtering
│   ├── 03_tuning_ensemble.py      # GridSearchCV, Random Forest, Gradient Boosting, Voting
│   ├── 04_ann_models.py           # Neural Networks (Sklearn MLP fallback built-in)
│   └── 05_final_pipeline.py       # Metrics Evaluation and automated Model Selection
│
├── static/                    # Frontend Assets
│   ├── css/
│   │   └── style.css          # (Supplemental custom logic integrated with Tailwind)
│   └── js/
│       └── app.js             # Async API caller, UI state manager, & Chart.js rendering
│
└── templates/
    └── index.html             # The monolithic Single Page Application (SPA) view
```

---

## ⚙️ The ML Pipeline Flow
The backend algorithms trigger via asynchronous REST API requests `/api/run-pipeline` through 5 crucial components:

1. **Preprocessing (`01`):** Loads `.csv`, maps targets (`High`, `Medium`, `Low` risk), interpolates missing data, and scales continuous integers via `StandardScaler`.
2. **Outlier Detection (`02`):** Purges statistical anomalies using Interquartile Range (IQR) and Isolation Forests. Maps natural patient segmentations using Silhouette-scored K-Means clustering.
3. **Hyperparameter Tuning (`03`):** Competes multiple algorithms sequentially (to prevent macOS/Flask multiprocessing thread deadlocks). Tests Random Forest, Gradient Boosting, Voting, and Stacking Classifiers natively.
4. **Deep Learning Fallback (`04`):** If system constraints prohibit massive `tensorflow` wheels, relies gracefully on Sklearn's lightweight `MLPClassifier` to generate Neural Network insights.
5. **Selection (`05`):** Calculates `f1-weighted` / `accuracy`. Dumps the top performing architecture into `.pkl` and spits JSON metrics back to the web UI.

---

## 🚀 Local Development / Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/HealthRisk_AI.git
   cd HealthRisk_AI
   ```

2. **Install Dependencies**
   *Python 3.11+ is recommended.*
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask Web Server**
   ```bash
   python app.py
   ```

4. **View the Dashboard**
   Navigate to `http://127.0.0.1:5000` in your web browser.

---

## 🌐 Production Deployment (Render)

This application is purposefully tuned for **[Render.com](https://render.com/)** free-tier deployment. 
1. Commit the repo to GitHub.
2. Log into Render -> "**New Web Service**" -> Attach Repository.
3. The server natively respects the `.python-version` (3.11.9) file and avoids Heavy-weight dependencies (like `tensorflow`) dropping the memory footprint below the 512MB limit.
4. Render natively reads the `Procfile` and boots up:
   ```bash
   gunicorn app:app --timeout 120 --workers 1 --threads 2
   ```

*Note: The frontend HTML handles dynamic URL routing natively; therefore, deploying the backend Flask app acts simultaneously as the frontend CDN without any cross-domain structural rewrites.*
