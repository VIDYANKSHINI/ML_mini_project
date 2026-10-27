"""
Model Training and Evaluation Pipeline: Random Forest vs XGBoost
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n{'='*20} {name} Performance {'='*20}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return {
        'name': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'cm': confusion_matrix(y_test, y_pred)
    }

def train_and_evaluate():
    # Load splits
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # 1. Random Forest
    print("\n[1/2] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_res = evaluate_model("Random Forest", rf_model, X_test, y_test)
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    print("Random Forest saved to models/random_forest_model.joblib")
    
    # 2. XGBoost
    print("\n[2/2] Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_res = evaluate_model("XGBoost", xgb_model, X_test, y_test)
    joblib.dump(xgb_model, 'models/xgboost_model.joblib')
    print("XGBoost saved to models/xgboost_model.joblib")
    
    # Plotting Model Comparison
    metrics_df = pd.DataFrame([
        {
            'Model': 'Random Forest',
            'Accuracy': rf_res['accuracy'],
            'Precision': rf_res['precision'],
            'Recall': rf_res['recall'],
            'F1-Score': rf_res['f1'],
            'ROC-AUC': rf_res['roc_auc']
        },
        {
            'Model': 'XGBoost',
            'Accuracy': xgb_res['accuracy'],
            'Precision': xgb_res['precision'],
            'Recall': xgb_res['recall'],
            'F1-Score': xgb_res['f1'],
            'ROC-AUC': xgb_res['roc_auc']
        }
    ])
    metrics_df.to_csv('models/metrics_comparison.csv', index=False)
    
    # Bar Chart Comparison
    plt.figure(figsize=(10, 5), dpi=300)
    df_melt = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Score')
    palette = ['#4f46e5', '#06b6d4']
    ax = sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette=palette)
    plt.title('Model Performance Comparison: Random Forest vs XGBoost', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.1)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points', fontweight='semibold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    # Side-by-side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    labels = ['Low Risk (0)', 'Addicted (1)']
    
    sns.heatmap(rf_res['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels)
    axes[0].set_title(f"Random Forest (Acc: {rf_res['accuracy']*100:.1f}%)", fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(xgb_res['cm'], annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=labels, yticklabels=labels)
    axes[1].set_title(f"XGBoost (Acc: {xgb_res['accuracy']*100:.1f}%)", fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png')
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(8, 6), dpi=300)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_res['y_proba'])
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_res['y_proba'])
    
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_res['roc_auc']:.3f})", color='#4f46e5', lw=2)
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {xgb_res['roc_auc']:.3f})", color='#06b6d4', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png')
    plt.close()
    
    print("\nTraining completed! All models and comparison plots saved in /models and /plots.")

if __name__ == '__main__':
    train_and_evaluate()
