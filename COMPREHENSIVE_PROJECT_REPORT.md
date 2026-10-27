# Comprehensive Project Report

# Explainable Machine Learning for Predicting Smartphone Addiction Risk in Adults

**Academic Domain:** Applied Machine Learning, Behavioral Health Informatics, and Explainable Artificial Intelligence (XAI)  
**Authors / Candidate:** Engineering Mini-Project Submission  
**Repository:** `https://github.com/VIDYANKSHINI/ML_mini_project.git`  
**Application Interface:** Interactive Streamlit Dashboard (`http://localhost:8501`)  

---

## Executive Summary

Excessive smartphone dependency among adults has grown into a widespread behavioral and psychological challenge, frequently precipitating sleep deprivation, cognitive fatigue, elevated stress levels, and occupational impairment. Although conventional machine learning algorithms achieve high predictive accuracy on behavioral datasets, their "black-box" architecture prevents clinical stakeholders and end-users from understanding the underlying reasoning behind predictions.

This project delivers an end-to-end, interpretable Machine Learning pipeline to predict smartphone addiction risk in adult cohorts using 7,500 real-world survey records. Two competitive ensemble learning paradigms—**Random Forest (Bagging)** and **XGBoost (Extreme Gradient Boosting)**—were developed, regularized, and benchmarked across 12 behavioral and lifestyle features. To resolve the black-box dilemma, the framework incorporates **SHAP (SHapley Additive exPlanations)** based on cooperative game theory. 

The resulting models demonstrate outstanding discriminatory performance on 1,500 independent, stratified test instances:
- **Random Forest:** 93.73% Accuracy, 96.45% Precision, 94.63% Recall, 95.53% F1-Score, and 0.9892 ROC-AUC.
- **XGBoost:** 93.53% Accuracy, 95.82% Precision, 95.01% Recall, 95.41% F1-Score, and 0.9893 ROC-AUC.

Through global SHAP beeswarm visualizations and local waterfall decompositions, the system uncovers that `daily_screen_time_hours` and `weekend_screen_time` act as the primary risk triggers, while nightly sleep duration (`sleep_hours` >= 7.5) functions as a decisive protective buffer. The entire pipeline is deployed via an interactive, production-grade Streamlit web dashboard providing real-time risk scoring and personalized feature-attribution graphs.

---

## Table of Contents

1. [Chapter 1: Introduction](#chapter-1-introduction)
   - 1.1 Context and Problem Overview
   - 1.2 Motivation: The Need for Explainability in Behavioral ML
   - 1.3 Problem Statement
   - 1.4 Project Objectives
   - 1.5 Scope and Boundaries
2. [Chapter 2: Literature Review & Theoretical Foundation](#chapter-2-literature-review--theoretical-foundation)
   - 2.1 Existing Smartphone Addiction Assessment Frameworks
   - 2.2 Ensemble Machine Learning Techniques
   - 2.3 Mathematical Formulation of Random Forest
   - 2.4 Mathematical Formulation of XGBoost
   - 2.5 Explainable AI (XAI) and Cooperative Game Theory (SHAP)
   - 2.6 TreeSHAP Algorithmic Efficiency
3. [Chapter 3: Dataset Architecture & Exploratory Data Analysis](#chapter-3-dataset-architecture--exploratory-data-analysis)
   - 3.1 Cohort Description and Attribute Schema
   - 3.2 Data Hygiene, Cleaning, and Null Verification
   - 3.3 Feature Transformation & Encoding Methodology
   - 3.4 Stratified Train-Test Splitting (80:20 Partitioning)
4. [Chapter 4: System Architecture & Workflow Pipeline](#chapter-4-system-architecture--workflow-pipeline)
   - 4.1 End-to-End System Diagram
   - 4.2 Data Pipeline Design
   - 4.3 Training & Hyperparameter Tuning Strategy
   - 4.4 Model Serialization and Artifact Versioning
5. [Chapter 5: Implementation Modules](#chapter-5-implementation-modules)
   - 5.1 Technology Stack & Development Environment
   - 5.2 Preprocessing Engine (`src/preprocess.py`)
   - 5.3 Model Training and Validation Module (`src/train.py`)
   - 5.4 SHAP Explainability Engine (`src/explain.py`)
   - 5.5 Interactive Web Dashboard (`app.py`)
6. [Chapter 6: Experimental Results & Benchmarking](#chapter-6-experimental-results--benchmarking)
   - 6.1 Evaluation Metrics Definition
   - 6.2 Quantitative Benchmark Comparison Table
   - 6.3 Confusion Matrix & Error Diagnosis
   - 6.4 ROC-AUC Curve Analysis
7. [Chapter 7: Explainable AI (SHAP) Insights & Discussion](#chapter-7-explainable-ai-shap-insights--discussion)
   - 7.1 Global Feature Importance (Mean |SHAP| Ranking)
   - 7.2 Directional Impact Analysis (Beeswarm Summary)
   - 7.3 Local Case Studies (Waterfall Decomposition)
   - 7.4 Behavioral & Psychological Implications
8. [Chapter 8: Conclusion & Future Scope](#chapter-8-conclusion--future-scope)
   - 8.1 Summary of Contributions
   - 8.2 Practical and Clinical Utility
   - 8.3 Limitations & Constraints
   - 8.4 Roadmap for Future Extensions
9. [References](#references)

---

## Chapter 1: Introduction

### 1.1 Context and Problem Overview
Over the past decade, smartphones have transformed from basic communication utilities into pervasive computational hubs that integrate social interaction, professional workflows, commerce, and multimedia entertainment. However, compulsive engagement with mobile interfaces has generated widespread behavioral anomalies often classified under **Problematic Smartphone Use (PSU)** or **Nomophobia** (No Mobile Phone Phobia). In adult demographics (ages 18 to 35+), smartphone addiction correlates strongly with impaired sleep architecture, delayed circadian rhythms, elevated anxiety, and decreased professional and academic throughput.

### 1.2 Motivation: The Need for Explainability in Behavioral ML
While modern supervised learning algorithms (e.g., deep neural networks, boosted trees) achieve high predictive capability, their widespread adoption in medical, psychological, and behavioral domains is hindered by their opacity. In high-stakes behavioral diagnosis:
1. **Clinical Justification:** A counselor cannot prescribe an intervention simply because an algorithm produced a "High Risk" probability. They require evidence indicating *which specific behaviors* triggered the classification.
2. **Actionability:** Users require personalized feedback (e.g., "reducing bedtime screen time by 1.5 hours will shift your status to low risk").
3. **Auditing and Bias Mitigation:** Machine learning models trained on survey data may inadvertently perpetuate demographic biases unless feature attributions are rigorously auditable.

Explainable AI (XAI) bridges this gap by rendering complex predictive models fully transparent, verifiable, and actionable.

### 1.3 Problem Statement
To design, train, evaluate, and interpret a machine learning system capable of accurately predicting smartphone addiction risk among adults, while generating individualized and population-wide feature-level explanations using game-theoretic Shapley attributions.

### 1.4 Project Objectives
1. **Data Pipeline Engineering:** Preprocess a structured survey dataset of 7,500 adult respondents, ensuring categorical encoding, statistical validation, and stratified splitting with zero data leakage.
2. **Model Development & Benchmarking:** Train and rigorously evaluate Random Forest (Bagging) and XGBoost (Gradient Boosting) classifiers against industry-standard metrics (Accuracy, F1-Score, ROC-AUC).
3. **XAI Integration:** Apply TreeSHAP to quantify the exact marginal contribution of each behavioral variable toward the final addiction classification.
4. **Interactive Deployment:** Build a responsive Streamlit web dashboard capable of performing real-time inference and rendering on-the-fly SHAP waterfall decompositions.

### 1.5 Scope and Boundaries
- **Target Population:** Adult smartphone users aged 18 to 35 years.
- **Classification Paradigm:** Binary risk classification (`0`: Low/Balanced Usage Risk, `1`: High Addiction Risk).
- **Data Scope:** Tabular survey responses capturing screen duration, application categories, notification load, sleep duration, and perceived stress. Physiological sensor telemetry (e.g., EEG, galvanic skin response) is outside the current scope.

---

## Chapter 2: Literature Review & Theoretical Foundation

### 2.1 Existing Smartphone Addiction Assessment Frameworks
Traditional diagnostic tools in behavioral psychology primarily rely on psychometric self-assessment scales, most notably the **Smartphone Addiction Scale - Short Version (SAS-SV)** (Kwon et al., 2013) and the **Nomophobia Questionnaire (NMP-Q)** (Yildirim & Correia, 2015). While validated, these manual screening questionnaires suffer from:
- Subjective recall bias.
- Static scoring cutoffs that fail to account for non-linear interactions between screen time, notification volume, and stress.
- Inability to simulate the projected benefit of targeted behavioral modifications.

Machine learning approaches overcome these limitations by evaluating complex multi-dimensional interaction surfaces.

### 2.2 Ensemble Machine Learning Techniques
Ensemble techniques combine multiple base estimators to construct a composite predictive model exhibiting superior generalization performance compared to any constituent model. They predominantly fall into two paradigms:
1. **Bagging (Bootstrap Aggregation):** Reduces variance by averaging predictions from multiple models trained on independent bootstrap partitions.
2. **Boosting:** Reduces bias by training models sequentially, where each successor prioritizes the residual errors generated by its predecessors.

### 2.3 Mathematical Formulation of Random Forest
Random Forest (Breiman, 2001) constructs an ensemble of $B$ decorrelated decision trees $\{T_1, T_2, \dots, T_B\}$. 

1. **Bootstrapping:** From a training set $D$ of size $N$, draw $B$ bootstrap samples $D_b$ with replacement ($|D_b| = N$). Approximately 63.2% of instances are sampled, with the remaining 36.8% constituting Out-Of-Bag (OOB) samples.
2. **Feature Subsampling (Random Subspace):** At each internal node during tree growth, evaluate a random subset of $m \ll M$ candidate features (typically $m = \lfloor\sqrt{M}\rfloor$) to determine the optimal split via Gini Impurity:
   $$I_G(p) = 1 - \sum_{k=1}^{K} p_k^2$$
3. **Ensemble Aggregation:** For binary classification $y \in \{0, 1\}$, the final probability is obtained through unweighted ensemble consensus:
   $$\hat{P}(Y=1|x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

### 2.4 Mathematical Formulation of XGBoost
XGBoost (Extreme Gradient Boosting; Chen & Guestrin, 2016) is a scalable tree-boosting system that optimizes an objective function comprising a convex loss function and an explicit regularization penalty:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

where the regularization term $\Omega(f_t)$ penalizes tree complexity:
$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$
Here, $T$ denotes the number of terminal leaves, $w_j$ represents leaf weights, $\gamma$ is the minimum loss reduction threshold for partitioning, and $\lambda$ is the L2 leaf-weight regularization factor.

Using second-order Taylor expansion around the previous prediction $\hat{y}_i^{(t-1)}$:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$ denote the first- and second-order gradients, respectively.

The optimal weight $w_j^*$ for leaf $j$ containing sample set $I_j$ is given by:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$
And the corresponding split quality score is calculated as:
$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

### 2.5 Explainable AI (XAI) and Cooperative Game Theory (SHAP)
SHAP (Shapley Additive exPlanations; Lundberg & Lee, 2017) conceptualizes model prediction as a cooperative game where input features are "players" and the predicted probability is the "payout". The Shapley value $\phi_i$ allocates credit to feature $i$ across all possible feature subsets $S \subseteq F \setminus \{i\}$:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

SHAP is the unique attribution method that simultaneously satisfies four fundamental axiomatic properties:
1. **Efficiency:** $\sum_{i=0}^{M} \phi_i(x) = f(x) - E[f(X)]$ (The sum of feature attributions equals the difference between individual prediction and expected base value).
2. **Symmetry:** If $f(S \cup \{i\}) = f(S \cup \{j\})$ for all $S$, then $\phi_i = \phi_j$.
3. **Dummy (Null Player):** If $f(S \cup \{i\}) = f(S)$ for all $S$, then $\phi_i = 0$.
4. **Additivity:** If two models are combined ($f + g$), the resulting attributions sum directly ($\phi_i(f+g) = \phi_i(f) + \phi_i(g)$).

### 2.6 TreeSHAP Algorithmic Efficiency
While general Shapley value computation exhibits exponential computational complexity $\mathcal{O}(2^{|F|})$, **TreeSHAP** optimizes attribution calculations specifically for tree-based architectures. By recursively evaluating feature presence through decision paths, TreeSHAP reduces complexity to polynomial time $\mathcal{O}(T L D^2)$, where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum tree depth. This efficiency enables real-time explanation generation in production web environments.

---

## Chapter 3: Dataset Architecture & Exploratory Data Analysis

### 3.1 Cohort Description and Attribute Schema
The study utilizes a structured empirical survey dataset comprising 7,500 adult participants. The feature space captures demographics, quantitative usage hours, notification engagement, and subjective strain.

| Variable Name | Data Type | Value Range / Domain | Role in Modeling |
|---|---|---|---|
| `transaction_id` | String | TXN00001 - TXN07500 | Dropped (Non-informative identifier) |
| `user_id` | String | U00001 - U07500 | Dropped (Non-informative identifier) |
| `age` | Integer | 18 - 35 years | Input Feature |
| `gender` | Categorical | `Female`, `Male`, `Other` | Input Feature |
| `daily_screen_time_hours`| Float | 1.0 - 15.8 hours | Input Feature |
| `social_media_hours` | Float | 0.0 - 9.2 hours | Input Feature |
| `gaming_hours` | Float | 0.0 - 7.5 hours | Input Feature |
| `work_study_hours` | Float | 0.0 - 9.8 hours | Input Feature |
| `sleep_hours` | Float | 3.2 - 9.8 hours | Input Feature |
| `notifications_per_day` | Integer | 15 - 380 notifications | Input Feature |
| `app_opens_per_day` | Integer | 10 - 290 unlocks/launches | Input Feature |
| `weekend_screen_time` | Float | 1.2 - 17.5 hours | Input Feature |
| `stress_level` | Categorical | `Low`, `Medium`, `High` | Input Feature |
| `academic_work_impact` | Binary | `No`, `Yes` | Input Feature |
| `addiction_level` | Categorical | `None`, `Mild`, `Moderate`, `Severe`| Auxiliary (Multiclass reference) |
| `addicted_label` | Integer | `0` (Non-addicted), `1` (Addicted) | Primary Target Variable |

### 3.2 Data Hygiene, Cleaning, and Null Verification
Automated data profiling confirmed that across all 7,500 rows, there were zero missing values in all 12 predictor features as well as the target variable `addicted_label`. In the auxiliary column `addiction_level`, 819 instances showed null values, corresponding precisely to the `None` addiction class (819 instances) which cleanly maps to `addicted_label = 0` (819 None + 1,373 Mild = 2,192 negative instances).

### 3.3 Feature Transformation & Encoding Methodology
To preserve semantic ordering and guarantee compatibility with tree ensembles:
- **Ordinal Feature Encoding:** `stress_level` exhibits intrinsic ranking and was encoded as:
  $$\text{stress\_level} \in \{\text{'Low'}: 0, \text{'Medium'}: 1, \text{'High'}: 2\}$$
- **Binary Feature Encoding:** `academic_work_impact` was mapped as:
  $$\text{academic\_work\_impact} \in \{\text{'No'}: 0, \text{'Yes'}: 1\}$$
- **Nominal Feature Encoding:** `gender` was consistently encoded as:
  $$\text{gender} \in \{\text{'Female'}: 0, \text{'Male'}: 1, \text{'Other'}: 2\}$$

All encoding mappings were serialized to `models/preprocessing_meta.joblib` to ensure identical feature handling during web inference without training data leakage.

### 3.4 Stratified Train-Test Splitting (80:20 Partitioning)
Class prevalence in the dataset indicates:
- Class `1` (Addicted / High Risk): 5,308 samples (70.77%)
- Class `0` (Balanced / Low Risk): 2,192 samples (29.23%)

To ensure class proportions remain identical across training and evaluation subsets, a **Stratified Shuffle Split** was performed using a fixed seed (`random_state=42`):
- **Training Subset ($D_{\text{train}}$):** 6,000 instances (4,246 positive, 1,754 negative)
- **Testing Subset ($D_{\text{test}}$):** 1,500 instances (1,062 positive, 438 negative)

---

## Chapter 4: System Architecture & Workflow Pipeline

### 4.1 End-to-End System Diagram

```text
========================================================================================
                                 END-TO-END PIPELINE
========================================================================================

   [ Raw Survey Data ] ------------------------> [ Data Ingestion & Sanitization ]
 (7,500 Adult Responses)                                        |
                                                                v
                                                [ Feature Mapping & Stratification ]
                                                 - Ordinal / Binary Label Encoding
                                                 - Stratified Split (80% / 20%)
                                                                |
                                       +------------------------+------------------------+
                                       |                                                 |
                                       v                                                 v
                           [ D_train: 6,000 Samples ]                       [ D_test: 1,500 Samples ]
                                       |                                                 |
                   +-------------------+-------------------+                             |
                   |                                       |                             |
                   v                                       v                             |
        [ Random Forest Model ]                 [ XGBoost Boosted Model ]                |
         - 150 Estimators                        - 150 Estimators                        |
         - Max Depth: 12                         - Max Depth: 5, LR: 0.08                |
                   |                                       |                             |
                   +-------------------+-------------------+                             |
                                       |                                                 |
                                       +------------------------+------------------------+
                                                                |
                                                                v
                                                [ Quantitative Model Benchmarking ]
                                                 - Accuracy, F1, Recall, Precision
                                                 - ROC-AUC Curve & Confusion Matrices
                                                                |
                                                                v
                                                [ TreeSHAP Explainability Engine ]
                                                 - Mean |SHAP| Global Feature Importance
                                                 - Summary Beeswarm Distribution
                                                 - Local Waterfall Single-Case Breakdown
                                                                |
                                                                v
                                                [ Streamlit Web Deployment Interface ]
                                                 - Real-time Assessment Form
                                                 - On-the-fly Dynamic Waterfall Chart
========================================================================================
```

### 4.2 Data Pipeline Design
The data ingestion and transformation layer (`src/preprocess.py`) isolates feature engineering logic. Raw data from `data/smartphone_addiction.csv` is sanitized, transformed, and persisted to `data/processed/` as pre-partitioned CSV files (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`), ensuring reproducible experiments across subsequent training executions.

### 4.3 Training & Hyperparameter Tuning Strategy
Hyperparameters were selected to balance model expressive capacity with overfitting constraints:

1. **Random Forest Classifier:**
   - `n_estimators = 150`: Provides sufficient ensemble variance reduction.
   - `max_depth = 12`: Constrains deep memorization of idiosyncratic survey noise.
   - `min_samples_split = 5`: Enforces terminal node generalization.
   - `n_jobs = -1`: Fully parallelizes tree generation across all CPU cores.

2. **XGBoost Classifier:**
   - `n_estimators = 150`: Number of sequential boosting rounds.
   - `max_depth = 5`: Limits maximum tree depth, capturing interaction terms up to order 5 while avoiding tree over-specialization.
   - `learning_rate = 0.08`: Conservative shrinkage parameter dampening each tree's marginal impact.
   - `subsample = 0.85`: Row subsampling fraction per tree.
   - `colsample_bytree = 0.85`: Column subsampling fraction per split.
   - `eval_metric = 'logloss'`: Direct negative log-likelihood objective.

### 4.4 Model Serialization and Artifact Versioning
All fitted estimators and utility structures are serialized to the `models/` directory using `joblib`:
- `random_forest_model.joblib`: Fitted Random Forest estimator.
- `xgboost_model.joblib`: Fitted XGBoost estimator.
- `shap_explainer.joblib`: Pre-compiled `shap.TreeExplainer` instance.
- `preprocessing_meta.joblib`: Feature dictionaries and categorical encoding lookup maps.
- `metrics_comparison.csv`: Persisted performance benchmark table.

---

## Chapter 5: Implementation Modules

### 5.1 Technology Stack & Development Environment
- **Core Runtime:** Python 3.13.2
- **Data Manipulation:** Pandas 3.0.3, NumPy 2.3.5
- **Machine Learning Libraries:** Scikit-Learn 1.7.1, XGBoost 3.4.1
- **Explainability Framework:** SHAP 0.52.0 (TreeSHAP)
- **Visualization Suite:** Matplotlib 3.11.0, Seaborn 0.13.2
- **Web Application Framework:** Streamlit 1.30+
- **Version Control:** Git 2.50.1

### 5.2 Preprocessing Engine (`src/preprocess.py`)
Encapsulates data ingestion, ordinal/binary mapping, target variable extraction, and stratified splitting. Persists processed partitions and exports feature schema metadata.

### 5.3 Model Training and Validation Module (`src/train.py`)
Loads preprocessed training and test partitions, fits the Random Forest and XGBoost estimators, records execution logs, calculates performance metrics, and exports graphical artifacts (`plots/model_comparison.png`, `plots/confusion_matrices.png`, `plots/roc_curves.png`).

### 5.4 SHAP Explainability Engine (`src/explain.py`)
Initializes the `TreeExplainer` on the fitted XGBoost model, computes exact Shapley attributions across the 1,500 test instances, and generates publication-grade visualizations (`plots/shap_feature_importance_bar.png`, `plots/shap_summary_beeswarm.png`, `plots/shap_waterfall_high_risk.png`, `plots/shap_waterfall_low_risk.png`).

### 5.5 Interactive Web Dashboard (`app.py`)
A modular Streamlit application providing four distinct user interfaces:
1. **Live Risk Predictor & SHAP:** Interactive sliders for adult profile inputs; executes instant inference and generates a dynamic SHAP waterfall chart.
2. **Model Benchmarking:** Displays side-by-side performance metrics, confusion matrices, and ROC curves.
3. **Global Explainability:** Renders population-level feature importance and beeswarm plots.
4. **Dataset Overview:** Provides an interactive exploration table and statistical summaries of the underlying dataset.

---

## Chapter 6: Experimental Results & Benchmarking

### 6.1 Evaluation Metrics Definition
To provide a comprehensive evaluation, five standard classification metrics were monitored on unseen test data ($D_{\text{test}}$):

1. **Accuracy:** Overall proportion of correct predictions:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision:** Proportion of predicted addiction cases that were truly positive:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity):** Proportion of true addiction cases correctly flagged:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score:** Harmonic mean of precision and recall:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve, quantifying discrimination capability across all classification thresholds.

### 6.2 Quantitative Benchmark Comparison Table

| Evaluation Metric | Random Forest (Bagging) | XGBoost (Boosting) | Performance Delta ($\Delta$) |
|---|---|---|---|
| **Accuracy** | **93.73%** (1406 / 1500) | 93.53% (1403 / 1500) | +0.20% (RF) |
| **Precision** | **96.45%** | 95.82% | +0.63% (RF) |
| **Recall** | 94.63% | **95.01%** | +0.38% (XGB) |
| **F1-Score** | **95.53%** | 95.41% | +0.12% (RF) |
| **ROC-AUC** | 0.9892 | **0.9893** | +0.0001 (XGB) |

### 6.3 Confusion Matrix & Error Diagnosis
Evaluating the 1,500 test samples (438 Negative, 1,062 Positive):

#### Random Forest Confusion Matrix:
- **True Negatives ($TN$):** 401 instances (91.55% specificity)
- **False Positives ($FP$):** 37 instances (8.45% false alarm rate)
- **False Negatives ($FN$):** 57 instances (5.37% miss rate)
- **True Positives ($TP$):** 1,005 instances (94.63% sensitivity)

#### XGBoost Confusion Matrix:
- **True Negatives ($TN$):** 394 instances (89.95% specificity)
- **False Positives ($FP$):** 44 instances (10.05% false alarm rate)
- **False Negatives ($FN$):** 53 instances (4.99% miss rate)
- **True Positives ($TP$):** 1,009 instances (95.01% sensitivity)

**Clinical Implication:** XGBoost missed only 53 high-risk cases out of 1,062, making it an excellent high-sensitivity candidate for screening tools. Random Forest produced fewer false alarms (37 vs. 44), making it slightly more conservative.

### 6.4 ROC-AUC Curve Analysis
Both models yielded near-perfect discrimination with ROC-AUC scores exceeding **0.989**. The curves demonstrate steep true positive ascent while maintaining false positive rates below 5%, confirming that classification boundaries are sharp and stable across probability decision thresholds.

---

## Chapter 7: Explainable AI (SHAP) Insights & Discussion

### 7.1 Global Feature Importance (Mean |SHAP| Ranking)
The mean absolute Shapley value across all test subjects measures overall feature impact:
$$\bar{\phi}_j = \frac{1}{N} \sum_{k=1}^{N} |\phi_j^{(k)}|$$

**Empirical Ranking of Predictors:**
1. `daily_screen_time_hours`: Mean $|\text{SHAP}| \approx 1.84$ (Strongest overall determinant).
2. `weekend_screen_time`: Mean $|\text{SHAP}| \approx 1.42$ (Prolonged weekend usage acts as a compounding signal).
3. `sleep_hours`: Mean $|\text{SHAP}| \approx 1.18$ (Dominant physiological counter-indicator).
4. `app_opens_per_day`: Mean $|\text{SHAP}| \approx 0.89$ (Compulsive unlocking habits).
5. `notifications_per_day`: Mean $|\text{SHAP}| \approx 0.74$ (Frequency of external interruptions).
6. `stress_level`: Mean $|\text{SHAP}| \approx 0.61$ (Psychological predisposition).
7. `academic_work_impact`: Mean $|\text{SHAP}| \approx 0.48$ (Functional disruption).
8. `social_media_hours`: Mean $|\text{SHAP}| \approx 0.42$ (Application-specific engagement).
9. `gaming_hours`: Mean $|\text{SHAP}| \approx 0.35$ (Specialized leisure use).
10. `work_study_hours`: Mean $|\text{SHAP}| \approx 0.28$ (Productive screen time has minimal addiction attribution).
11. `age`: Mean $|\text{SHAP}| \approx 0.15$ (Low variance within 18-35 cohort).
12. `gender`: Mean $|\text{SHAP}| \approx 0.08$ (Negligible direct predictive attribution).

### 7.2 Directional Impact Analysis (Beeswarm Summary)
The SHAP Summary Beeswarm plot reveals the directionality of each variable:
- **Screen Time Variables:** Red markers (high hours) cluster exclusively on the positive SHAP axis ($\phi > 0$), demonstrating that increased active screen exposure monotonically drives addiction risk.
- **Sleep Duration:** In direct contrast, blue markers (low sleep duration, e.g., < 5.5 hours) concentrate on the positive SHAP axis, while red markers (high sleep duration, e.g., > 7.5 hours) populate the negative SHAP axis ($\phi < 0$). This establishes adequate sleep as an active protective buffer against smartphone addiction.
- **App Opens & Notifications:** High values show a pronounced long right-tail, confirming that frequent context-switching and compulsive phone unlocks significantly push users toward the addicted state.

### 7.3 Local Case Studies (Waterfall Decomposition)

#### Case Study A: High-Risk Subject (Index #1)
- **Base Expected Value $E[f(X)]:$** -0.89 log-odds.
- **Observed Characteristics:** `daily_screen_time_hours = 9.54`, `sleep_hours = 4.92`, `app_opens_per_day = 175`, `stress_level = High`.
- **Attribution Flow:** Screen time contributes $+1.94$, low sleep adds $+1.21$, and high stress contributes $+0.62$, shifting the output to $+2.88$ (corresponding to a predicted probability of 94.7% High Addiction Risk).
- **Counseling Recommendation:** Interventions should target hard daily screen limits and sleep hygiene restoration.

#### Case Study B: Low-Risk Subject (Index #0)
- **Base Expected Value $E[f(X)]:$** -0.89 log-odds.
- **Observed Characteristics:** `daily_screen_time_hours = 3.23`, `sleep_hours = 7.55`, `notifications_per_day = 82`, `stress_level = Low`.
- **Attribution Flow:** Screen time reduces risk by $-1.45$, healthy sleep subtracts $-0.84$, and low stress subtracts $-0.32$, driving final output to $-3.50$ (predicted probability of 2.9% Low Risk).

---

## Chapter 8: Conclusion & Future Scope

### 8.1 Summary of Contributions
1. Constructed a robust, leak-free machine learning pipeline processing 7,500 adult behavioral records.
2. Benchmarked Random Forest (93.73% accuracy) and XGBoost (93.53% accuracy, 0.989 ROC-AUC), establishing superior performance on survey data.
3. Solved the machine learning black-box dilemma by integrating TreeSHAP, producing publication-quality global beeswarm distributions and individualized local waterfall plots.
4. Deployed the complete system into a functional, user-centric Streamlit web application capable of performing live inference and interactive feature attribution.

### 8.2 Practical and Clinical Utility
- **Digital Wellbeing Integration:** Can be embedded into operating-system wellbeing dashboards (e.g., Android Digital Wellbeing, iOS Screen Time) to provide actionable behavioral feedback rather than simple raw usage metrics.
- **Occupational Health Portals:** Applicable in corporate and educational wellness programs to monitor burn-out and compulsive tech engagement.
- **Psychological Screening:** Functions as an objective triaging assistant for counselors evaluating adult behavioral dependencies.

### 8.3 Limitations & Constraints
- Data relies on self-reported survey records, which may carry mild self-reporting biases.
- Cross-sectional observation lacks longitudinal temporal tracking over multi-month intervals.

### 8.4 Roadmap for Future Extensions
1. **Passive Telemetry Integration:** Incorporate passive background sensor logging (accelerometer posture tracking, ambient light sensors) to assess bedtime phone use without manual reporting.
2. **Personalized Intervention Recommender:** Implement a counterfactual explanation module (e.g., using DiCE) that computes the exact minimal behavioral adjustments required to transition from High Risk to Low Risk.
3. **Multi-Class Severity Stratification:** Expand classification from binary risk into granular tiers (`Mild`, `Moderate`, `Severe`) with multi-output SHAP decomposition.

---

## References

1. **Lundberg, S. M., & Lee, S. I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765-4774.
2. **Lundberg, S. M., et al. (2020).** From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.
3. **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
4. **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5-32.
5. **Kwon, M., Kim, D. J., Cho, H., & Yang, S. (2013).** The smartphone addiction scale: development and validation of a short version for adolescents and young adults. *PLoS ONE*, 8(12), e83558.
6. **Yildirim, C., & Correia, A. P. (2015).** Exploring the dimensions of nomophobia: Development and validation of a self-reported questionnaire. *Computers in Human Behavior*, 49, 130-137.
7. **Shapley, L. S. (1953).** A value for n-person games. *Contributions to the Theory of Games*, 2(28), 307-317.
8. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825-2830.

---
*End of Comprehensive Project Report.*
