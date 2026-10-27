# Explainable ML for Predicting Smartphone Addiction Risk in Adults 📱

An end-to-end Machine Learning and Explainable AI (XAI) project designed to predict smartphone addiction risk in adults and provide transparent, game-theoretic justifications using **Random Forest**, **XGBoost**, and **SHAP (SHapley Additive exPlanations)**.

---

## 🌟 Key Highlights
- **Real-World Survey Dataset:** 7,500 adult survey records with 12 behavioral and psychological features.
- **High Performance Models:**
  - **Random Forest:** `93.73% Accuracy`, `0.989 ROC-AUC`
  - **XGBoost:** `93.53% Accuracy`, `0.989 ROC-AUC`
- **Explainable AI (XAI) with SHAP:**
  - **Global Interpretability:** Summary Beeswarm Plot & Feature Importance ranking.
  - **Local Interpretability:** Real-time SHAP Waterfall plots explaining *why* an individual is categorized at High or Low risk.
- **Interactive Web App:** Clean, modern Streamlit dashboard for real-time assessment and visualization.

---

## 📂 Project Architecture
```text
ML project/
├── data/
│   ├── smartphone_addiction.csv       # Raw survey dataset (7,500 rows)
│   └── processed/                      # Preprocessed train/test splits (80-20)
├── models/
│   ├── random_forest_model.joblib     # Trained Random Forest Classifier
│   ├── xgboost_model.joblib           # Trained XGBoost Classifier
│   ├── shap_explainer.joblib          # Saved TreeExplainer object
│   ├── preprocessing_meta.joblib      # Feature mappings & metadata
│   └── metrics_comparison.csv         # Benchmark evaluation metrics
├── plots/
│   ├── model_comparison.png           # Accuracy, F1, ROC-AUC comparison chart
│   ├── confusion_matrices.png         # Test set confusion matrices
│   ├── roc_curves.png                 # ROC Curves for RF and XGBoost
│   ├── shap_feature_importance_bar.png# Global feature importance
│   ├── shap_summary_beeswarm.png      # SHAP Beeswarm distribution plot
│   ├── shap_waterfall_high_risk.png   # Case study: High addiction risk
│   └── shap_waterfall_low_risk.png    # Case study: Healthy/Low risk
├── src/
│   ├── preprocess.py                  # Data cleaning and encoding
│   ├── train.py                       # Model training and evaluation
│   └── explain.py                     # SHAP TreeExplainer & plot generation
├── app.py                             # Interactive Streamlit Web Dashboard
├── PROJECT_REPORT_AND_VIVA.md         # Complete viva guide & project report notes
└── README.md                          # Documentation
```

---

## 🚀 How to Run the Project

### 1. Run Data Preprocessing
```bash
python src/preprocess.py
```

### 2. Train and Evaluate Random Forest & XGBoost
```bash
python src/train.py
```

### 3. Generate SHAP Explainability Plots
```bash
python src/explain.py
```

### 4. Launch the Interactive Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Benchmark Comparison

| Metric | Random Forest | XGBoost |
|---|---|---|
| **Accuracy** | **93.73%** | 93.53% |
| **Precision** | 96.45% | 95.82% |
| **Recall** | 94.63% | 95.01% |
| **F1-Score** | **95.53%** | 95.41% |
| **ROC-AUC** | 0.9892 | **0.9893** |

---

## 🧠 Key Findings from SHAP Explainability
1. **Primary Drivers:** Total daily screen time and weekend screen time are the single strongest predictors of smartphone addiction.
2. **Protective Habits:** Night sleep duration (> 7.5 hours) acts as a powerful buffer, negatively attributing to addiction risk.
3. **Compulsive Triggers:** High frequency of app unlocks and notifications per day significantly push individuals towards high risk.
4. **Stress Multiplier:** High perceived stress levels amplify the impact of high screen time on work and academic performance.
