"""
Data Preprocessing Pipeline for Smartphone Addiction Risk in Adults
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

FEATURE_COLS = [
    'age',
    'gender',
    'daily_screen_time_hours',
    'social_media_hours',
    'gaming_hours',
    'work_study_hours',
    'sleep_hours',
    'notifications_per_day',
    'app_opens_per_day',
    'weekend_screen_time',
    'stress_level',
    'academic_work_impact'
]

GENDER_MAP = {'Female': 0, 'Male': 1, 'Other': 2}
STRESS_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
IMPACT_MAP = {'No': 0, 'Yes': 1}

def load_and_preprocess_data(csv_path='data/smartphone_addiction.csv', test_size=0.2, random_state=42):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Handle mappings
    df_clean = df.copy()
    df_clean['gender'] = df_clean['gender'].map(GENDER_MAP)
    df_clean['stress_level'] = df_clean['stress_level'].map(STRESS_MAP)
    df_clean['academic_work_impact'] = df_clean['academic_work_impact'].map(IMPACT_MAP)
    
    # Target
    y = df_clean['addicted_label']
    X = df_clean[FEATURE_COLS]
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Preprocessing completed.")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Features: {list(X.columns)}")
    
    # Save processed splits for reproducibility
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save metadata dictionary for UI
    metadata = {
        'features': FEATURE_COLS,
        'gender_map': GENDER_MAP,
        'stress_map': STRESS_MAP,
        'impact_map': IMPACT_MAP
    }
    joblib.dump(metadata, 'models/preprocessing_meta.joblib')
    print("Preprocessed files and metadata saved successfully!")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    load_and_preprocess_data()
