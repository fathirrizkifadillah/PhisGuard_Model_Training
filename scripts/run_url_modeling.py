import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CLEAN_URL_PATH = Path('data/url/processed/url_features_cleaned.csv')
MODEL_DIR = Path('models/url')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_PLOT_DIR = Path('plots/url')
EVAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading cleaned dataset...")
df_clean = pd.read_csv(CLEAN_URL_PATH)
df_clean['path_ratio'] = df_clean['path_length'] / (df_clean['url_length'] + 1e-9)

feature_cols = [c for c in df_clean.columns if c not in ['URL', 'Class_Name', 'is_phishing']]
X = df_clean[feature_cols]
y = df_clean['is_phishing']

print(f"Total features: {len(feature_cols)}")
print("Features:", feature_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
print(f"Train set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")

# 1. Train Random Forest Baseline
print("\nTraining Random Forest Baseline...")
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=4
)
t0 = time.time()
rf_baseline.fit(X_train, y_train)
rf_fit_time = time.time() - t0
print(f"Random Forest fit time: {rf_fit_time:.2f} s")

# 2. Train XGBoost Production
print("\nTraining XGBoost Production...")
xgb_production = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.08,
    colsample_bytree=0.8,
    subsample=0.8,
    tree_method='hist',
    random_state=42,
    n_jobs=4,
    eval_metric='logloss'
)
t0 = time.time()
xgb_production.fit(X_train, y_train)
xgb_fit_time = time.time() - t0
print(f"XGBoost fit time: {xgb_fit_time:.2f} s")

# Save models
rf_path = MODEL_DIR / 'rf_baseline_url.joblib'
xgb_path = MODEL_DIR / 'xgb_production_url.joblib'
joblib.dump(rf_baseline, rf_path)
joblib.dump(xgb_production, xgb_path)
print(f"Models saved to {rf_path} and {xgb_path}")

# 3. Stratified 5-Fold Cross Validation on 25k sample
print("\nRunning Stratified 5-Fold Cross-Validation on 25k sample...")
sample_cv_idx = X_train.sample(25000, random_state=42).index
X_cv = X_train.loc[sample_cv_idx]
y_cv = y_train.loc[sample_cv_idx]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

# To avoid deadlocks on Windows, use single estimator thread during CV
rf_cv_model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=1)
xgb_cv_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, colsample_bytree=0.8, subsample=0.8, tree_method='hist', random_state=42, n_jobs=1, eval_metric='logloss')

cv_rf = cross_validate(rf_cv_model, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=2)
cv_xgb = cross_validate(xgb_cv_model, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=2)

print("\n--- 5-Fold Cross-Validation Comparison ---")
print(f"Random Forest Accuracy: {cv_rf['test_accuracy'].mean():.4f} +/- {cv_rf['test_accuracy'].std():.4f} | F1: {cv_rf['test_f1'].mean():.4f} | ROC-AUC: {cv_rf['test_roc_auc'].mean():.4f} | Fit Time: {cv_rf['fit_time'].mean():.2f}s")
print(f"XGBoost       Accuracy: {cv_xgb['test_accuracy'].mean():.4f} +/- {cv_xgb['test_accuracy'].std():.4f} | F1: {cv_xgb['test_f1'].mean():.4f} | ROC-AUC: {cv_xgb['test_roc_auc'].mean():.4f} | Fit Time: {cv_xgb['fit_time'].mean():.2f}s")

# 4. Evaluation on Independent Test Set
print("\nEvaluating on Independent Test Set (20,175 rows)...")
y_pred_rf = rf_baseline.predict(X_test)
y_prob_rf = rf_baseline.predict_proba(X_test)[:, 1]

y_pred_xgb = xgb_production.predict(X_test)
y_prob_xgb = xgb_production.predict_proba(X_test)[:, 1]

print("\n=== CLASSIFICATION REPORT: RANDOM FOREST (BASELINE) ===")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate (0)', 'Phishing (1)'], digits=4))

print("=== CLASSIFICATION REPORT: XGBOOST (PRODUCTION) ===")
print(classification_report(y_test, y_pred_xgb, target_names=['Legitimate (0)', 'Phishing (1)'], digits=4))

# 5. Confusion Matrix Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Pred Legit (0)', 'Pred Phish (1)'], yticklabels=['Actual Legit (0)', 'Actual Phish (1)'])
axes[0].set_title(f'Random Forest Baseline (Test Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%)', fontsize=12, fontweight='bold')

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1],
            xticklabels=['Pred Legit (0)', 'Pred Phish (1)'], yticklabels=['Actual Legit (0)', 'Actual Phish (1)'])
axes[1].set_title(f'XGBoost Production (Test Accuracy: {accuracy_score(y_test, y_pred_xgb)*100:.2f}%)', fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison on Independent Testing Set (n=20,175)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
eval_file1 = EVAL_PLOT_DIR / 'eval_01_confusion_matrix.png'
plt.savefig(eval_file1, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {eval_file1}")

# 6. ROC and PR Curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
pr_auc_rf = auc(rec_rf, prec_rf)
pr_auc_xgb = auc(rec_xgb, prec_xgb)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})', color='#3498db', linewidth=2)
axes[0].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.4f})', color='#2ecc71', linewidth=2, linestyle='--')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':')
axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()

axes[1].plot(rec_rf, prec_rf, label=f'Random Forest (PR-AUC = {pr_auc_rf:.4f})', color='#3498db', linewidth=2)
axes[1].plot(rec_xgb, prec_xgb, label=f'XGBoost (PR-AUC = {pr_auc_xgb:.4f})', color='#2ecc71', linewidth=2, linestyle='--')
axes[1].set_title('Precision-Recall Curve (Focus on Minority Target)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend()

plt.suptitle('Performance Curves Comparison (Testing Set)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
eval_file2 = EVAL_PLOT_DIR / 'eval_02_roc_pr_curves.png'
plt.savefig(eval_file2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {eval_file2}")

# 7. Feature Importances
fi_rf = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_baseline.feature_importances_}).sort_values(by='Importance', ascending=True)
fi_xgb = pd.DataFrame({'Feature': feature_cols, 'Importance': xgb_production.feature_importances_}).sort_values(by='Importance', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].barh(fi_rf['Feature'], fi_rf['Importance'], color='#3498db', edgecolor='black', alpha=0.85)
axes[0].set_title('Random Forest Feature Importances (MDI Gini)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance')

axes[1].barh(fi_xgb['Feature'], fi_xgb['Importance'], color='#2ecc71', edgecolor='black', alpha=0.85)
axes[1].set_title('XGBoost Feature Importances (Gain Weight)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance')

plt.suptitle('Model Feature Attribution Comparison: Baseline vs Production', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
eval_file3 = EVAL_PLOT_DIR / 'eval_03_feature_importances.png'
plt.savefig(eval_file3, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {eval_file3}")

# 8. Error Analysis
test_indices = X_test.index
df_error = df_clean.loc[test_indices].copy()
df_error['pred_rf'] = y_pred_rf
df_error['pred_xgb'] = y_pred_xgb
df_error['prob_xgb'] = y_prob_xgb

fp_xgb = df_error[(df_error['is_phishing'] == 0) & (df_error['pred_xgb'] == 1)]
print(f"\nFalse Positive (XGBoost): {len(fp_xgb)} / {sum(y_test==0):,} ({len(fp_xgb)/sum(y_test==0)*100:.3f}%)")

fn_xgb = df_error[(df_error['is_phishing'] == 1) & (df_error['pred_xgb'] == 0)]
print(f"False Negative (XGBoost): {len(fn_xgb)} / {sum(y_test==1):,} ({len(fn_xgb)/sum(y_test==1)*100:.3f}%)")

print("\nURL Modeling complete!")
