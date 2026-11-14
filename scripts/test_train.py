import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def main():
    df = pd.read_csv('data/url/processed/url_features_cleaned.csv')
    df['path_ratio'] = df['path_length'] / (df['url_length'] + 1e-9)

    feature_cols = [c for c in df.columns if c not in ['URL', 'is_phishing']]
    X = df[feature_cols]
    y = df['is_phishing']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    print('Training Random Forest Baseline...')
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    print(f'RF Fit Time: {t_rf:.2f}s, ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}')

    print('Training XGBoost Production Model...')
    t0 = time.time()
    xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=4, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    t_xgb = time.time() - t0
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    print(f'XGB Fit Time: {t_xgb:.2f}s, ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}')

if __name__ == '__main__':
    main()
