"""
Explainable AI (XAI) with SHAP for Smartphone Addiction Risk
Generates Global and Local Explanations using TreeExplainer.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

def run_shap_analysis():
    print("Loading test data and trained XGBoost model...")
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    xgb_model = joblib.load('models/xgboost_model.joblib')
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)
    
    # Save explainer object for fast loading in Streamlit
    joblib.dump(explainer, 'models/shap_explainer.joblib')
    print("TreeExplainer saved to models/shap_explainer.joblib")
    
    # 1. Global Feature Importance Bar Plot
    print("Generating [1/4] SHAP Feature Importance Bar Plot...")
    plt.figure(figsize=(10, 6), dpi=300)
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('plots/shap_feature_importance_bar.png', bbox_inches='tight')
    plt.close()
    
    # 2. Global Summary Beeswarm Plot
    print("Generating [2/4] SHAP Beeswarm Summary Plot...")
    plt.figure(figsize=(11, 7), dpi=300)
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Summary Plot: Feature Impact on Smartphone Addiction Risk", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('plots/shap_summary_beeswarm.png', bbox_inches='tight')
    plt.close()
    
    # Find sample indices for Addicted (1) and Healthy (0) cases
    y_pred = xgb_model.predict(X_test)
    addicted_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    healthy_indices = np.where((y_test == 0) & (y_pred == 0))[0]
    
    idx_addicted = addicted_indices[0] if len(addicted_indices) > 0 else 0
    idx_healthy = healthy_indices[0] if len(healthy_indices) > 0 else 0
    
    # 3. Local Explanation: High Risk / Addicted Adult Waterfall Plot
    print(f"Generating [3/4] Local Waterfall Plot for High Risk Adult (Index {idx_addicted})...")
    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(shap_values[idx_addicted], max_display=10, show=False)
    plt.title(f"Individual Case Analysis: High Addiction Risk Adult (Test Sample #{idx_addicted})",
              fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('plots/shap_waterfall_high_risk.png', bbox_inches='tight')
    plt.close()
    
    # 4. Local Explanation: Low Risk Adult Waterfall Plot
    print(f"Generating [4/4] Local Waterfall Plot for Low Risk Adult (Index {idx_healthy})...")
    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(shap_values[idx_healthy], max_display=10, show=False)
    plt.title(f"Individual Case Analysis: Low Addiction Risk Adult (Test Sample #{idx_healthy})",
              fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('plots/shap_waterfall_low_risk.png', bbox_inches='tight')
    plt.close()
    
    print("\nSHAP analysis successfully generated all 4 publication-quality plots in /plots!")

if __name__ == '__main__':
    run_shap_analysis()
