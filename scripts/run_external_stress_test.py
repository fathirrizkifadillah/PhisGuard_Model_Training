import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.url_feature_extractor import extract_url_dataframe

PLOT_DIR = Path('plots/external_eval')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Regex cleaner for email
def clean_email_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("=================================================================")
print("PHISGUARD: OUT-OF-DISTRIBUTION (OOD) EXTERNAL STRESS-TEST ENGINE")
print("=================================================================\n")

# 1. URL EXTERNAL TESTING
print(">>> TAHAP 1: OUT-OF-DISTRIBUTION TESTING PADA MODUL URL (XGBOOST)")
xgb_model = joblib.load('models/url/xgb_production_url.joblib')

df_openphish = pd.read_csv('data/external/url/openphish_live_phishing.csv')
df_umbrella = pd.read_csv('data/external/url/umbrella_top_legitimate.csv')
df_curated_url = pd.read_csv('data/external/url/curated_url_edge_cases.csv')

# Gabungkan OpenPhish + Umbrella untuk benchmark OOD skala besar (800 sampel)
df_url_external = pd.concat([df_openphish[['URL', 'is_phishing']], df_umbrella[['URL', 'is_phishing']]], ignore_index=True)
print(f"Total URL Eksternal Benchmark: {len(df_url_external)} (Phishing={sum(df_url_external['is_phishing']==1)}, Safe={sum(df_url_external['is_phishing']==0)})")

print("Mengekstraksi 17 fitur leksikal & struktural dari string URL...")
X_url_ext = extract_url_dataframe(df_url_external['URL'])
y_url_ext = df_url_external['is_phishing']

# Prediksi
y_pred_url = xgb_model.predict(X_url_ext)
y_prob_url = xgb_model.predict_proba(X_url_ext)[:, 1]

print("\n=== CLASSIFICATION REPORT: URL OOD BENCHMARK (OPENPHISH + CISCO UMBRELLA) ===")
print(classification_report(y_url_ext, y_pred_url, target_names=['Legitimate (0)', 'Phishing (1)'], digits=4))

# Analisis Slice Sesuai Skill evaluation-reality-check
print("\n--- AUDIT SUBGROUP SLICE DISPARITY (URL) ---")
# Slice 1: HTTPS vs HTTP pada Phishing OpenPhish
openphish_feats = extract_url_dataframe(df_openphish['URL'])
openphish_preds = xgb_model.predict(openphish_feats)
openphish_probs = xgb_model.predict_proba(openphish_feats)[:, 1]

https_phish_mask = openphish_feats['https_flag'] == 1
http_phish_mask = openphish_feats['https_flag'] == 0

recall_https = np.mean(openphish_preds[https_phish_mask] == 1) if sum(https_phish_mask) > 0 else 0
recall_http = np.mean(openphish_preds[http_phish_mask] == 1) if sum(http_phish_mask) > 0 else 0

print(f"Prevalensi HTTPS pada Phishing OpenPhish: {sum(https_phish_mask)} / {len(df_openphish)} ({sum(https_phish_mask)/len(df_openphish)*100:.1f}%)")
print(f"Recall pada Phishing HTTPS: {recall_https*100:.2f}% ({sum(openphish_preds[https_phish_mask]==1)} / {sum(https_phish_mask)})")
print(f"Recall pada Phishing HTTP:  {recall_http*100:.2f}% ({sum(openphish_preds[http_phish_mask]==1)} / {sum(http_phish_mask)})")
print(f"Slice Disparity Gap: {abs(recall_http - recall_https)*100:.2f}%")

# Uji Kasus Khusus (Curated Edge Cases)
print("\n--- UJI KASUS KHUSUS REAL-WORLD URL (20 SAMPEL) ---")
curated_feats = extract_url_dataframe(df_curated_url['URL'])
curated_preds = xgb_model.predict(curated_feats)
curated_probs = xgb_model.predict_proba(curated_feats)[:, 1]
df_curated_url['pred'] = curated_preds
df_curated_url['prob_phish'] = curated_probs
df_curated_url['correct'] = df_curated_url['is_phishing'] == df_curated_url['pred']

curated_acc = accuracy_score(df_curated_url['is_phishing'], df_curated_url['pred'])
print(f"Akurasi pada Kasus Nyata Terkurasi: {curated_acc*100:.2f}% ({sum(df_curated_url['correct'])} / {len(df_curated_url)})")
for _, r in df_curated_url.iterrows():
    status = "OK" if r['correct'] else "MISSED"
    print(f"[{status}] Label={r['is_phishing']} Pred={r['pred']} (Prob={r['prob_phish']:.3f}) | {r['category']} | {r['URL'][:50]}")

# Visualisasi URL OOD
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cm_url_ext = confusion_matrix(y_url_ext, y_pred_url)
sns.heatmap(cm_url_ext, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Pred Legit (0)', 'Pred Phish (1)'], yticklabels=['Actual Legit (0)', 'Actual Phish (1)'])
axes[0].set_title(f'URL OOD Confusion Matrix (Acc: {accuracy_score(y_url_ext, y_pred_url)*100:.2f}%)', fontsize=12, fontweight='bold')

# Bar plot per slice
slices_df = pd.DataFrame({
    'Subgroup Slice': ['Cisco Umbrella (Safe)', 'OpenPhish HTTP', 'OpenPhish HTTPS', 'Curated Edge Cases'],
    'Accuracy / Recall': [
        np.mean(y_pred_url[y_url_ext==0] == 0) * 100,
        recall_http * 100,
        recall_https * 100,
        curated_acc * 100
    ]
})
sns.barplot(data=slices_df, x='Subgroup Slice', y='Accuracy / Recall', palette='viridis', ax=axes[1])
axes[1].set_ylim(0, 105)
axes[1].set_title('Performance by Slice Disparity', fontsize=12, fontweight='bold')
for i, v in enumerate(slices_df['Accuracy / Recall']):
    axes[1].text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold')

plt.suptitle('URL Threat Detection: Out-of-Distribution Stress Test', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
p1 = PLOT_DIR / '01_external_url_confusion_matrix.png'
plt.savefig(p1, dpi=200, bbox_inches='tight')
plt.close()
print(f"Plot disimpan: {p1}")

# 2. EMAIL EXTERNAL TESTING
print("\n\n>>> TAHAP 2: OUT-OF-DISTRIBUTION TESTING PADA MODUL EMAIL (LINEAR SVM)")
svm_model = joblib.load('models/email/svm_production_email.joblib')
tfidf_model = joblib.load('models/email/tfidf_vectorizer_email.joblib')

df_enron = pd.read_csv('data/external/email/enron_external_benchmark.csv')
df_curated_email = pd.read_csv('data/external/email/curated_real_world_email_lures.csv')

print(f"Total Email Enron Benchmark: {len(df_enron)} (Spam/Phish={sum(df_enron['is_phishing']==1)}, Ham/Safe={sum(df_enron['is_phishing']==0)})")

print("Membersihkan teks email eksternal dan mentransformasikan ke vektor TF-IDF...")
cleaned_enron_texts = df_enron['text'].apply(clean_email_text)
X_enron = tfidf_model.transform(cleaned_enron_texts)
y_enron = df_enron['is_phishing']

y_pred_email = svm_model.predict(X_enron)
y_prob_email = svm_model.predict_proba(X_enron)[:, 1]

print("\n=== CLASSIFICATION REPORT: EMAIL OOD BENCHMARK (ENRON CORPUS) ===")
print(classification_report(y_enron, y_pred_email, target_names=['Safe/Ham (0)', 'Phish/Spam (1)'], digits=4))

# Uji Kasus Khusus Email (Curated Lures)
print("\n--- UJI KASUS KHUSUS REAL-WORLD EMAIL LURES (14 SAMPEL) ---")
cleaned_curated_email = df_curated_email['text'].apply(clean_email_text)
X_curated_email = tfidf_model.transform(cleaned_curated_email)
curated_email_preds = svm_model.predict(X_curated_email)
curated_email_probs = svm_model.predict_proba(X_curated_email)[:, 1]

df_curated_email['pred'] = curated_email_preds
df_curated_email['prob_phish'] = curated_email_probs
df_curated_email['correct'] = df_curated_email['is_phishing'] == df_curated_email['pred']

curated_email_acc = accuracy_score(df_curated_email['is_phishing'], df_curated_email['pred'])
print(f"Akurasi pada Email Serangan Nyata: {curated_email_acc*100:.2f}% ({sum(df_curated_email['correct'])} / {len(df_curated_email)})")
for _, r in df_curated_email.iterrows():
    status = "OK" if r['correct'] else "MISSED"
    print(f"[{status}] Label={r['is_phishing']} Pred={r['pred']} (Prob={r['prob_phish']:.3f}) | {r['category']} | {r['text'][:55]}...")

# Visualisasi Email OOD
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cm_email_ext = confusion_matrix(y_enron, y_pred_email)
sns.heatmap(cm_email_ext, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[0],
            xticklabels=['Pred Safe (0)', 'Pred Phish (1)'], yticklabels=['Actual Safe (0)', 'Actual Phish (1)'])
axes[0].set_title(f'Email Enron Confusion Matrix (Acc: {accuracy_score(y_enron, y_pred_email)*100:.2f}%)', fontsize=12, fontweight='bold')

# Distribusi Skor Probabilitas Prediksi Enron
sns.histplot(y_prob_email[y_enron==0], bins=25, color='#3498db', label='Actual Safe / Ham', ax=axes[1], alpha=0.6, kde=True)
sns.histplot(y_prob_email[y_enron==1], bins=25, color='#e74c3c', label='Actual Phish / Spam', ax=axes[1], alpha=0.6, kde=True)
axes[1].axvline(0.5, color='black', linestyle='--', label='Default Threshold (0.50)')
axes[1].axvline(0.85, color='purple', linestyle=':', label='Strict Quarantine (0.85)')
axes[1].set_title('Probability Calibration Distribution on Enron', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Probability of Phishing')
axes[1].legend()

plt.suptitle('Email Threat Classification: Out-of-Distribution Stress Test', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
p2 = PLOT_DIR / '03_external_email_confusion_matrix.png'
plt.savefig(p2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Plot disimpan: {p2}")

print("\nOOD STRESS-TEST SELESAI DENGAN SUKSES!")
