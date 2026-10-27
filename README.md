# Explainable Machine Learning for Predicting Smartphone Addiction Risk in Adults

A comprehensive, end-to-end framework integrating predictive machine learning with game-theoretic Explainable Artificial Intelligence (XAI) to evaluate and interpret smartphone addiction risk among adult populations.

---

## 1. Executive Summary

Smartphone dependency and nomophobia pose increasing risks to psychological health, sleep architecture, and occupational productivity in adults. While high-capacity non-linear machine learning models effectively identify addiction indicators, their black-box nature creates barriers to clinical adoption and behavioural counseling.

This repository presents an interpretable predictive system that pairs **Random Forest** and **XGBoost** ensemble classifiers with **SHAP (SHapley Additive exPlanations)**. Evaluated on 7,500 adult survey records across 12 behavioral and psychological variables, the system achieves over 93.5% classification accuracy and 0.989 ROC-AUC, while generating local and global explanations of individual risk factors in real time.

---

## 2. System Architecture

```text
+-------------------------------------------------------------+
|                7,500 Adult Survey Records                   |
| (Demographics, Screen Time, App Opens, Sleep, Stress Level) |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 Preprocessing & Validation                  |
| - Categorical Ordinal & Binary Encoding                     |
| - Stratified 80/20 Train-Test Partitioning                  |
| - Feature Space Integrity Verification                      |
+-------------------------------------------------------------+
                              |
               +--------------+--------------+
               |                             |
               v                             v
+-----------------------------+ +-----------------------------+
|    Random Forest (Bagging)  | |      XGBoost (Boosting)     |
| - 150 Estimators            | | - 150 Estimators            |
| - Max Depth: 12             | | - Max Depth: 5, LR: 0.08    |
| - Min Samples Split: 5      | | - Subsample: 0.85           |
+-----------------------------+ +-----------------------------+
               |                             |
               +--------------+--------------+
                              |
                              v
+-------------------------------------------------------------+
|             Comparative Benchmark & Evaluation              |
| Accuracy, Precision, Recall, F1-Score, ROC-AUC, Conf. Matrix|
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|           TreeSHAP Explainability Engine (XAI)              |
| - Global Feature Attribution (Mean |SHAP| Bar Chart)        |
| - Distributional Feature Impact (Summary Beeswarm Plot)     |
| - Local Instance Attribution (Waterfall Explanations)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|               Interactive Web Application                   |
| - Real-time Risk Assessment & Confidence Scoring            |
| - On-the-fly SHAP Waterfall Generation via Streamlit        |
+-------------------------------------------------------------+
```

---

## 3. Dataset Specification

The underlying dataset comprises survey responses from 7,500 adult subjects. The feature space spans demographic attributes, daily screen interactions, mobile engagement frequency, and subjective well-being metrics.

### Feature Dictionary

| Feature Name | Type | Description |
|---|---|---|
| `age` | Integer | Chronological age of the adult respondent (18 - 35) |
| `gender` | Categorical | Self-reported gender identity (`Female`, `Male`, `Other`) |
| `daily_screen_time_hours` | Continuous | Total average active daily screen time (hours) |
| `social_media_hours` | Continuous | Daily time spent on social networking applications |
| `gaming_hours` | Continuous | Daily time allocated to mobile gaming |
| `work_study_hours` | Continuous | Screen time dedicated to professional or educational tasks |
| `sleep_hours` | Continuous | Self-reported average nightly sleep duration |
| `notifications_per_day` | Integer | Average frequency of mobile notifications received daily |
| `app_opens_per_day` | Integer | Daily device unlock and application launch frequency |
| `weekend_screen_time` | Continuous | Average daily screen time recorded over weekends |
| `stress_level` | Categorical | Perceived psychological stress (`Low`, `Medium`, `High`) |
| `academic_work_impact` | Binary | Self-reported negative impact on performance (`No`, `Yes`) |
| `addicted_label` | Binary Target | Ground truth classification (`0` = Low/Balanced, `1` = High Risk) |

---

## 4. Methodology

### 4.1. Preprocessing Pipeline
- **Categorical Encoding:** Ordinal variables (`stress_level`) are mapped to sequential numeric ranks (`Low`: 0, `Medium`: 1, `High`: 2). Binary variables (`academic_work_impact`) are mapped to `[0, 1]`. Multi-class categorical variables (`gender`) are consistently mapped across training and inference.
- **Stratified Partitioning:** A stratified 80/20 train-test split preserves class prevalence across subsets, yielding 6,000 training instances and 1,500 independent evaluation instances.

### 4.2. Machine Learning Algorithms
- **Random Forest:** Implemented as a bagging ensemble of decision trees. Individual trees are trained on bootstrap samples with random feature subsampling at each decision node, effectively reducing model variance and mitigating overfitting on tabular survey inputs.
- **XGBoost:** Implemented as a gradient-boosted decision tree architecture. Iterative trees minimize a regularized objective function combining negative gradient loss with L1 and L2 penalty terms, effectively capturing non-linear interactions between usage habits and stress indicators.

### 4.3. Game-Theoretic Explainability (SHAP)
To transition from opaque predictions to clinical transparency, the system employs **TreeSHAP** (Lundberg et al.), computing exact Shapley values derived from cooperative game theory:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

- **Global Interpretability:** Assesses population-wide influence via mean absolute SHAP values and directional beeswarm distributions.
- **Local Interpretability:** Decomposes an individual adult's predicted log-odds into baseline expectation $E[f(X)]$ and additive feature contributions $\phi_i$, visualized through waterfall plots.

---

## 5. Experimental Results and Benchmarking

Performance evaluation was conducted on 1,500 unseen, stratified test samples. Both models demonstrate robust discriminatory power and balanced classification.

### Quantitative Comparison

| Metric | Random Forest | XGBoost |
|---|---|---|
| **Accuracy** | **93.73%** | 93.53% |
| **Precision** | **96.45%** | 95.82% |
| **Recall** | 94.63% | **95.01%** |
| **F1-Score** | **95.53%** | 95.41% |
| **ROC-AUC** | 0.9892 | **0.9893** |

### Key Observations
- Random Forest achieved slightly higher precision (96.45%) and accuracy (93.73%), demonstrating high reliability against false positives.
- XGBoost achieved higher recall (95.01%) and an equivalent ROC-AUC (0.9893), optimizing sensitivity for high-risk detection.
- Both models maintain false positive and false negative rates below 6% on independent test data.

---

## 6. Explainability Findings

Analysis using TreeSHAP on test cohorts revealed clear behavioral and protective indicators:

1. **Dominant Risk Determinants:** `daily_screen_time_hours` and `weekend_screen_time` constitute the strongest positive drivers of smartphone addiction risk.
2. **Protective Thresholds:** Nightly sleep duration (`sleep_hours` >= 7.5 hours) consistently exerts negative attribution values, indicating strong buffer effects against dependency.
3. **Compulsive Behavioral Patterns:** Elevated `app_opens_per_day` and `notifications_per_day` reflect frequent compulsive checking habits that elevate predicted risk.
4. **Stress Modulation:** High self-reported stress levels amplify the impact of high screen exposure, shifting predictions toward high-risk thresholds.

---

## 7. Project Structure

```text
.
|-- data/
|   |-- smartphone_addiction.csv        # Master survey dataset (7,500 samples)
|   `-- processed/                      # Preprocessed train/test partitions
|-- models/
|   |-- metrics_comparison.csv          # Quantitative benchmark metrics
|   |-- preprocessing_meta.joblib       # Encoders and feature metadata
|   |-- random_forest_model.joblib      # Serialized Random Forest classifier
|   |-- shap_explainer.joblib           # Pre-fitted TreeExplainer instance
|   `-- xgboost_model.joblib            # Serialized XGBoost classifier
|-- plots/
|   |-- confusion_matrices.png          # Test set confusion matrix heatmaps
|   |-- model_comparison.png            # Multi-metric comparative bar chart
|   |-- roc_curves.png                  # Receiver Operating Characteristic curves
|   |-- shap_feature_importance_bar.png # Global feature importance ranking
|   |-- shap_summary_beeswarm.png       # Global feature impact distribution
|   |-- shap_waterfall_high_risk.png    # High-risk adult case attribution
|   `-- shap_waterfall_low_risk.png     # Low-risk adult case attribution
|-- src/
|   |-- explain.py                      # SHAP explanation and plotting pipeline
|   |-- preprocess.py                   # Data cleaning, encoding, and splitting
|   `-- train.py                        # Model training, validation, and benchmarking
|-- .gitignore                          # Version control exclusion rules
|-- app.py                              # Interactive Streamlit dashboard
|-- README.md                           # Project documentation
`-- requirements.txt                    # Python environment dependencies
```

---

## 8. Installation and Execution

### Prerequisites
- Python 3.10 to 3.13
- Virtual environment tool (`venv` or `conda`)

### Step 1: Clone Repository
```bash
git clone https://github.com/VIDYANKSHINI/ML_mini_project.git
cd ML_mini_project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Data Preprocessing
```bash
python src/preprocess.py
```

### Step 4: Train and Evaluate Models
```bash
python src/train.py
```

### Step 5: Generate SHAP Explainability Visualizations
```bash
python src/explain.py
```

### Step 6: Launch Web Dashboard
```bash
streamlit run app.py
```
Access the application interface at `http://localhost:8501`.

---

## 9. Interactive Web Dashboard Features

The application provides four dedicated operational views:
- **Live Risk Predictor:** Allows users to adjust behavioral sliders and evaluate addiction risk with instant probability metrics.
- **On-the-Fly SHAP Breakdown:** Computes individual waterfall attributions dynamically, visualizing specific risk and protective factors for each query.
- **Model Benchmarking:** Displays tabular metrics, confusion matrices, and ROC curves comparing Random Forest and XGBoost.
- **Global Explainability:** Displays population-level feature rankings and beeswarm distributions derived from TreeSHAP.

---

## 10. References and Standards

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Kwon, M., et al. (2013). The smartphone addiction scale: development and validation of a short version for adolescents. *PLoS ONE*, 8(12), e83558.
