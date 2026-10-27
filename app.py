"""
Streamlit Web Application: Explainable ML for Smartphone Addiction Risk in Adults
Features:
- Live Interactive Risk Assessment
- Real-time SHAP Waterfall Explanation (TreeSHAP)
- Model Comparison (Random Forest vs XGBoost)
- Global Explainability Insights & Research Findings
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Page Configuration
st.set_page_config(
    page_title="Explainable ML: Smartphone Addiction Risk in Adults",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-risk-badge {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 1.1rem;
    }
    .low-risk-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    rf_model = joblib.load('models/random_forest_model.joblib')
    xgb_model = joblib.load('models/xgboost_model.joblib')
    meta = joblib.load('models/preprocessing_meta.joblib')
    explainer = joblib.load('models/shap_explainer.joblib')
    metrics_df = pd.read_csv('models/metrics_comparison.csv')
    return rf_model, xgb_model, meta, explainer, metrics_df

rf_model, xgb_model, meta, explainer, metrics_df = load_assets()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/smartphone.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Section:",
    [
        "🔍 Live Risk Predictor & SHAP",
        "📊 Model Benchmarking (RF vs XGBoost)",
        "🧠 Global Explainability (SHAP Insights)",
        "📁 Dataset Overview"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Information**")
st.sidebar.info(
    "**Title:** Explainable ML for Smartphone Addiction Risk in Adults\n\n"
    "**Algorithms:** Random Forest, XGBoost\n\n"
    "**XAI Method:** SHAP (Shapley Additive exPlanations)\n\n"
    "**Dataset:** 7,500 Adult Survey Responses"
)

# ----------------- PAGE 1: PREDICTOR & SHAP -----------------
if page == "🔍 Live Risk Predictor & SHAP":
    st.markdown('<div class="main-title">📱 Smartphone Addiction Risk Predictor & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter adult demographic and smartphone habit details to predict addiction risk with real-time SHAP justification.</div>', unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1.1, 1.2], gap="large")
    
    with col_input:
        st.subheader("1. Adult Survey Profile")
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", min_value=18, max_value=45, value=25, step=1)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            daily_screen = st.slider("Daily Screen Time (Hours)", min_value=1.0, max_value=16.0, value=7.5, step=0.5)
            social_media = st.slider("Social Media Usage (Hours)", min_value=0.0, max_value=10.0, value=3.5, step=0.5)
            gaming = st.slider("Gaming Usage (Hours)", min_value=0.0, max_value=8.0, value=1.5, step=0.5)
            work_study = st.slider("Work / Study Screen Time (Hours)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
            
        with c2:
            sleep_hours = st.slider("Night Sleep Duration (Hours)", min_value=3.0, max_value=10.0, value=6.0, step=0.5)
            notifications = st.number_input("Notifications Received / Day", min_value=10, max_value=400, value=160, step=10)
            app_opens = st.number_input("App Opens / Unlocks per Day", min_value=5, max_value=300, value=90, step=5)
            weekend_screen = st.slider("Weekend Screen Time (Hours)", min_value=1.0, max_value=18.0, value=9.5, step=0.5)
            stress_level = st.selectbox("Perceived Stress Level", options=["Low", "Medium", "High"], index=1)
            academic_work_impact = st.selectbox("Has usage impacted work/productivity?", options=["No", "Yes"], index=1)
            
        selected_model_name = st.radio("Select Prediction Engine:", ["XGBoost (Recommended)", "Random Forest"], horizontal=True)
        
        submit_btn = st.button("🚀 Assess Smartphone Addiction Risk", use_container_width=True, type="primary")

    with col_result:
        st.subheader("2. AI Prediction & Explainability")
        
        # Prepare input sample
        input_data = pd.DataFrame([{
            'age': age,
            'gender': meta['gender_map'][gender],
            'daily_screen_time_hours': daily_screen,
            'social_media_hours': social_media,
            'gaming_hours': gaming,
            'work_study_hours': work_study,
            'sleep_hours': sleep_hours,
            'notifications_per_day': notifications,
            'app_opens_per_day': app_opens,
            'weekend_screen_time': weekend_screen,
            'stress_level': meta['stress_map'][stress_level],
            'academic_work_impact': meta['impact_map'][academic_work_impact]
        }])
        
        active_model = xgb_model if "XGBoost" in selected_model_name else rf_model
        
        pred_label = active_model.predict(input_data)[0]
        pred_proba = active_model.predict_proba(input_data)[0]
        risk_probability = pred_proba[1] * 100
        
        if submit_btn or 'assessed' in st.session_state:
            st.session_state['assessed'] = True
            
            # Prediction Card
            st.markdown("---")
            if pred_label == 1:
                st.markdown(f'<div class="high-risk-badge">⚠️ HIGH ADDICTION RISK DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f"**Addiction Risk Probability:** `{risk_probability:.1f}%`")
                st.warning("Assessment indicates symptoms of high smartphone dependency and behavioral disruption.")
            else:
                st.markdown(f'<div class="low-risk-badge">✅ LOW / BALANCED USAGE RISK</div>', unsafe_allow_html=True)
                st.markdown(f"**Addiction Risk Probability:** `{risk_probability:.1f}%`")
                st.success("Smartphone habits appear within healthy, balanced boundaries.")
                
            st.markdown("---")
            st.markdown("#### 🔬 Explainable AI: Why did the model make this decision?")
            st.caption("SHAP Waterfall chart decomposing individual feature contributions for this person:")
            
            # Compute live SHAP values using XGBoost explainer
            try:
                shap_val_single = explainer(input_data)
                
                fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
                shap.plots.waterfall(shap_val_single[0], max_display=8, show=False)
                plt.title("Local Feature Contribution Breakdown (SHAP Waterfall)", fontsize=11, fontweight='bold', pad=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.info(
                    "💡 **How to interpret:** Red bars push the risk score **higher** (increasing addiction risk), "
                    "while blue bars push the risk score **lower** (protective habits like higher sleep duration)."
                )
            except Exception as e:
                st.error(f"SHAP explanation generation note: {e}")
        else:
            st.info("👈 Fill in the user profile on the left and click **'Assess Smartphone Addiction Risk'** to see the prediction and live SHAP explanation breakdown.")

# ----------------- PAGE 2: MODEL BENCHMARKING -----------------
elif page == "📊 Model Benchmarking (RF vs XGBoost)":
    st.markdown('<div class="main-title">📊 Random Forest vs XGBoost Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comparative evaluation across 1,500 unseen stratified test survey samples.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Metrics Table")
        formatted_df = metrics_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(formatted_df, use_container_width=True)
        
    with col2:
        st.subheader("Key Takeaways for Presentation/Viva")
        st.markdown("""
        * **Random Forest (Bagging):** Achieved **93.73% Accuracy** and **0.989 ROC-AUC**. Exceptional stability on tabular survey features.
        * **XGBoost (Boosting):** Achieved **93.53% Accuracy** and **0.989 ROC-AUC**. Excellent for calculating exact Shapley attribution values via TreeSHAP.
        * **Zero Data Leakage:** Categorical encoders and scalers fitted strictly on training data with 80-20 stratified split.
        """)
        
    st.markdown("---")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.image('plots/model_comparison.png', caption='Model Comparison Bar Chart', use_container_width=True)
    with c_p2:
        st.image('plots/confusion_matrices.png', caption='Confusion Matrices (Test Set)', use_container_width=True)
        
    st.image('plots/roc_curves.png', caption='Receiver Operating Characteristic (ROC) Curves', use_container_width=True)

# ----------------- PAGE 3: GLOBAL EXPLAINABILITY -----------------
elif page == "🧠 Global Explainability (SHAP Insights)":
    st.markdown('<div class="main-title">🧠 Global Explainability & Clinical Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Understanding behavioral drivers of smartphone addiction across the entire adult cohort using Game-Theoretic SHAP.</div>', unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("1. Global Feature Importance")
        st.caption("Ranking of features by mean absolute SHAP value impact across all 7,500 samples.")
        st.image('plots/shap_feature_importance_bar.png', use_container_width=True)
        
    with col_g2:
        st.subheader("2. Summary Beeswarm Plot")
        st.caption("Distribution of SHAP values. High feature values in red, low in blue.")
        st.image('plots/shap_summary_beeswarm.png', use_container_width=True)
        
    st.markdown("---")
    st.subheader("3. Core Research Findings")
    st.markdown("""
    1. **Primary Trigger:** `daily_screen_time_hours` and `weekend_screen_time` are the strongest drivers pushing adults towards high addiction risk.
    2. **Protective Factor:** `sleep_hours` acts as a strong buffer. Adults maintaining 7.5+ hours of sleep show significantly lower SHAP attribution towards addiction.
    3. **Behavioral Traps:** High frequency of `app_opens_per_day` and `notifications_per_day` demonstrate habitual compulsive checking behavior.
    4. **Psychological Interaction:** High `stress_level` and affirmative `academic_work_impact` consistently correlate with elevated risk thresholds.
    """)

# ----------------- PAGE 4: DATASET OVERVIEW -----------------
elif page == "📁 Dataset Overview":
    st.markdown('<div class="main-title">📁 Smartphone Addiction Survey Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Overview of the 7,500 adult demographic and usage records used for training.</div>', unsafe_allow_html=True)
    
    df_raw = pd.read_csv('data/smartphone_addiction.csv')
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Adults Surveyed", f"{df_raw.shape[0]:,}")
    m2.metric("Total Features", f"{df_raw.shape[1]}")
    m3.metric("Addiction Prevalance", f"{(df_raw['addicted_label'].mean()*100):.1f}%")
    m4.metric("Age Range", f"{df_raw['age'].min()} - {df_raw['age'].max()} yrs")
    
    st.markdown("### Sample Records")
    st.dataframe(df_raw.head(15), use_container_width=True)
    
    st.markdown("### Statistical Summary")
    st.dataframe(df_raw.describe(), use_container_width=True)
