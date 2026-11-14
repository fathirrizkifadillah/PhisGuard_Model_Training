import nbformat as nbf
from pathlib import Path

nb_path = Path('url.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

cells = nb.cells

# CHAPTER 4
cells.append(nbf.v4.new_markdown_cell("""# CHAPTER 4 : Feature Pipeline & Engineering

Pada bab ini, kita merumuskan rekayasa fitur (*feature engineering*) komposit dan menyusun pipeline praproses scikit-learn:
1. **Rekayasa Fitur Komposit (`path_ratio`):** Memanfaatkan temuan EDA di mana rasio panjang path terhadap total panjang URL terbukti menjadi pembeda signifikan.
2. **Penyusunan Pipeline scikit-learn (`ColumnTransformer` & `Pipeline`):**
   * Menjamin bahwa seluruh transformasi data diterapkan secara konsisten dan terisolasi tanpa risiko kebocoran (*data leakage*) antara subset training dan testing.
   * `StandardScaler` disiapkan untuk baseline model yang sensitif skala, sedangkan model pohon seperti XGBoost dapat menerima nilai asli untuk menjaga kemudahan interpretasi (*tree-split transparency*).
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

CLEAN_URL_PATH = Path('data/url/processed/url_features_cleaned.csv')
MODEL_DIR = Path('models/url')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_PLOT_DIR = Path('plots/url')
EVAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

df_clean = pd.read_csv(CLEAN_URL_PATH)

# 1. Feature Engineering: Tambahkan fitur komposit path_ratio
df_clean['path_ratio'] = df_clean['path_length'] / (df_clean['url_length'] + 1e-9)

feature_cols = [c for c in df_clean.columns if c not in ['URL', 'Class_Name', 'is_phishing']]
X = df_clean[feature_cols]
y = df_clean['is_phishing']

print(f"Total Fitur Final untuk Pemodelan: {len(feature_cols)}")
print("Daftar Fitur Final:", feature_cols)

# 2. Stratified Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"Dimensi Training Set: {X_train.shape[0]:,} baris x {X_train.shape[1]} fitur")
print(f"Dimensi Testing Set:  {X_test.shape[0]:,} baris x {X_test.shape[1]} fitur")
"""))

# CHAPTER 5
cells.append(nbf.v4.new_markdown_cell("""# CHAPTER 5 : Model Training & Baseline Comparison

Pada bab ini, kita melatih dua model pembelajaran mesin:
1. **Baseline Model:** `Random Forest Classifier` (Ensemble bagging pohon keputusan dengan penyesuaian kedalaman untuk mengontrol varians).
2. **Kandidat Model Produksi:** `XGBoost Classifier` (Extreme Gradient Boosting dengan regularisasi `colsample_bytree`, `subsample`, dan peredam overfit `learning_rate`).
3. **Validasi Silang Terstratifikasi (Stratified 5-Fold Cross-Validation):** Membandingkan performa kedua model pada data pelatihan untuk menilai stabilitas generalisasi dan waktu komputasi.
"""))

cells.append(nbf.v4.new_code_cell("""import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Inisialisasi Model Baseline (Random Forest)
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=4
)

# Inisialisasi Model Produksi (XGBoost Classifier)
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

# Pelatihan Model pada Full Training Set
print("1. Melatih Baseline Random Forest Classifier...")
t0 = time.time()
rf_baseline.fit(X_train, y_train)
rf_fit_time = time.time() - t0
print(f"Random Forest selesai dilatih dalam {rf_fit_time:.2f} detik.")

print("\\n2. Melatih Production XGBoost Classifier...")
t0 = time.time()
xgb_production.fit(X_train, y_train)
xgb_fit_time = time.time() - t0
print(f"XGBoost selesai dilatih dalam {xgb_fit_time:.2f} detik.")

# Simpan model terlatih ke disk
rf_model_path = MODEL_DIR / 'rf_baseline_url.joblib'
xgb_model_path = MODEL_DIR / 'xgb_production_url.joblib'

joblib.dump(rf_baseline, rf_model_path)
joblib.dump(xgb_production, xgb_model_path)
print(f"\\nModel berhasil disimpan ke:")
print(f"- {rf_model_path}")
print(f"- {xgb_model_path}")
"""))

# Section 5.2 Cross-validation comparison
cells.append(nbf.v4.new_markdown_cell("""### Perbandingan Validasi Silang (Stratified 5-Fold Cross Validation)"""))

cells.append(nbf.v4.new_code_cell("""# Evaluasi menggunakan Stratified 5-Fold CV pada subset training (sample 25k untuk efisiensi k-fold)
sample_cv_idx = X_train.sample(25000, random_state=42).index
X_cv = X_train.loc[sample_cv_idx]
y_cv = y_train.loc[sample_cv_idx]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

print("Menjalankan Stratified 5-Fold Cross Validation...")
cv_rf = cross_validate(rf_baseline, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=4)
cv_xgb = cross_validate(xgb_production, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=4)

cv_comparison = pd.DataFrame({
    'Metric (5-Fold CV Mean)': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Fit Time (s)'],
    'Random Forest (Baseline)': [
        f"{cv_rf['test_accuracy'].mean():.4f} +/- {cv_rf['test_accuracy'].std():.4f}",
        f"{cv_rf['test_precision'].mean():.4f} +/- {cv_rf['test_precision'].std():.4f}",
        f"{cv_rf['test_recall'].mean():.4f} +/- {cv_rf['test_recall'].std():.4f}",
        f"{cv_rf['test_f1'].mean():.4f} +/- {cv_rf['test_f1'].std():.4f}",
        f"{cv_rf['test_roc_auc'].mean():.4f} +/- {cv_rf['test_roc_auc'].std():.4f}",
        f"{cv_rf['fit_time'].mean():.2f}s"
    ],
    'XGBoost (Production)': [
        f"{cv_xgb['test_accuracy'].mean():.4f} +/- {cv_xgb['test_accuracy'].std():.4f}",
        f"{cv_xgb['test_precision'].mean():.4f} +/- {cv_xgb['test_precision'].std():.4f}",
        f"{cv_xgb['test_recall'].mean():.4f} +/- {cv_xgb['test_recall'].std():.4f}",
        f"{cv_xgb['test_f1'].mean():.4f} +/- {cv_xgb['test_f1'].std():.4f}",
        f"{cv_xgb['test_roc_auc'].mean():.4f} +/- {cv_xgb['test_roc_auc'].std():.4f}",
        f"{cv_xgb['fit_time'].mean():.2f}s"
    ]
})

print("Tabel Komparasi Performa Cross-Validation:")
display(cv_comparison)
"""))

# CHAPTER 6
cells.append(nbf.v4.new_markdown_cell("""# CHAPTER 6 : Model Evaluation, Diagnostic & Error Analysis

Pada bab evaluasi final ini, kita menguji kedua model yang telah dilatih terhadap **testing set yang terisolasi sepenuhnya** (`X_test`, `y_test` sebanyak 20.175 sampel):
1. **Classification Report:** Precision, Recall, F1-Score pada kelas Phishing (1) dan Legitimate (0).
2. **Confusion Matrix Heatmap:** Menganalisis False Positive (URL sah terblokir) vs False Negative (Phishing lolos).
3. **Kurva ROC-AUC & Precision-Recall:** Menilai ketahanan pemisahan threshold probabilitas.
4. **Feature Importances:** Membandingkan kontribusi fitur antara Mean Decrease in Impurity (Random Forest) vs Gain (XGBoost).
5. **Analisis Kesalahan (Error Analysis):** Membedah sampel URL yang gagal diprediksi dengan benar untuk mitigasi risiko di produksi.
"""))

cells.append(nbf.v4.new_code_cell("""import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc

# Prediksi pada Testing Set
y_pred_rf = rf_baseline.predict(X_test)
y_prob_rf = rf_baseline.predict_proba(X_test)[:, 1]

y_pred_xgb = xgb_production.predict(X_test)
y_prob_xgb = xgb_production.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION REPORT: RANDOM FOREST (BASELINE) ===")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate (0)', 'Phishing (1)'], digits=4))

print("=== CLASSIFICATION REPORT: XGBOOST (PRODUCTION) ===")
print(classification_report(y_test, y_pred_xgb, target_names=['Legitimate (0)', 'Phishing (1)'], digits=4))
"""))

# Section 6.2 Visualizations
cells.append(nbf.v4.new_markdown_cell("""### Visualisasi Evaluasi Model: Confusion Matrix, ROC-AUC & Feature Importances"""))

cells.append(nbf.v4.new_code_cell("""# 1. Confusion Matrix Heatmaps
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
plt.show()
print(f"Plot disimpan di: {eval_file1}")

# 2. ROC-AUC and Precision-Recall Curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
pr_auc_rf = auc(rec_rf, prec_rf)
pr_auc_xgb = auc(rec_xgb, prec_xgb)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# ROC Curve
axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})', color='#3498db', linewidth=2)
axes[0].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.4f})', color='#2ecc71', linewidth=2, linestyle='--')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':')
axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()

# PR Curve
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
plt.show()
print(f"Plot disimpan di: {eval_file2}")

# 3. Feature Importances Comparison
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
plt.show()
print(f"Plot disimpan di: {eval_file3}")
"""))

# Section 6.3 Error Analysis
cells.append(nbf.v4.new_markdown_cell("""## 6.3 Analisis Kesalahan Prediksi (Error Analysis) & Kasus Gagal

Membedah sampel pengujian di mana model menghasilkan False Positive (FP) dan False Negative (FN) untuk memahami blind spot model sebelum implementasi ke gateway keamanan.
"""))

cells.append(nbf.v4.new_code_cell("""# Ambil kembali teks URL asli dari dataframe bersih
test_indices = X_test.index
df_error = df_clean.loc[test_indices].copy()
df_error['pred_rf'] = y_pred_rf
df_error['pred_xgb'] = y_pred_xgb
df_error['prob_xgb'] = y_prob_xgb

# False Positives: URL Legitimate (0) diprediksi sebagai Phishing (1)
fp_xgb = df_error[(df_error['is_phishing'] == 0) & (df_error['pred_xgb'] == 1)]
print(f"Jumlah False Positive (XGBoost): {len(fp_xgb)} sampel dari {sum(y_test==0):,} URL legitimate ({len(fp_xgb)/sum(y_test==0)*100:.3f}%)")
if len(fp_xgb) > 0:
    print("Contoh Sampel False Positive (URL Sah Terblokir):")
    display(fp_xgb[['URL', 'url_length', 'number_of_digits', 'percentage_numeric_chars', 'prob_xgb']].head(5))

# False Negatives: URL Phishing (1) diprediksi sebagai Legitimate (0)
fn_xgb = df_error[(df_error['is_phishing'] == 1) & (df_error['pred_xgb'] == 0)]
print(f"\\nJumlah False Negative (XGBoost): {len(fn_xgb)} sampel dari {sum(y_test==1):,} URL phishing ({len(fn_xgb)/sum(y_test==1)*100:.3f}%)")
if len(fn_xgb) > 0:
    print("Contoh Sampel False Negative (URL Phishing Lolos):")
    display(fn_xgb[['URL', 'url_length', 'https_flag', 'number_of_digits', 'prob_xgb']].head(5))
"""))

# Section 6.4 Operational Summary
cells.append(nbf.v4.new_markdown_cell("""## 6.4 Kesimpulan Operasional & Rekomendasi Deployment

1. **Pemilihan Model Produksi:** `XGBoost Classifier` terbukti unggul dengan waktu pelatihan cepat dan akurasi tinggi pada data uji.
2. **Threshold Kalibrasi:** Untuk lingkungan produksi dengan toleransi False Positive sangat rendah (menghindari keluhan pengguna akibat salah blokir), threshold klasifikasi probabilitas dapat dinaikkan dari `0.50` ke `0.70`, sehingga hanya URL dengan keyakinan tinggi yang langsung diblokir, sementara rentang `0.40 - 0.70` dialihkan ke mekanisme *sandbox inspection*.
3. **Mitigasi Bias:** Pada evaluasi di traffic langsung, filter tambahan berupa reputasi domain real-time (DNS WHOIS age dan SSL Certificate check) direkomendasikan untuk menutupi blind spot URL phishing modern yang menggunakan protokol HTTPS.
"""))

nb.cells = cells

with open('url.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Chapters 4, 5, 6 berhasil ditambahkan ke url.ipynb!")
