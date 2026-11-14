import nbformat as nbf
from nbclient import NotebookClient
import os

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.9"
    }
}

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🛡️ PhisGuard: URL Threat Detection Engine
## CHAPTER 1 : Dataset Quality Audit (LegitPhish Dataset)

---

### 📌 Project Metadata & Model Roadmap
* **Domain:** Cybersecurity / Phishing URL Detection
* **Target Model (Production):** `XGBoost Classifier` (Optimized Gradient Boosted Decision Trees)
* **Baseline Model:** `Random Forest Classifier`
* **Dataset Name:** `LegitPhish Dataset`
* **Dataset Source:** Mendeley Data (DOI: [10.17632/hx4m73v2sf.2](https://data.mendeley.com/datasets/hx4m73v2sf/2))
* **Dataset Authors:** Rachana Potpelwar, Uday Kulkarni, Jaishri Waghmare (2025)
* **Underlying Sources:**
  * **Phishing URLs:** URLHaus & PhishTank (feed intelijen ancaman terverifikasi)
  * **Legitimate URLs:** Google Search API, Wikipedia, dan Stack Overflow (high-authority domains)

---
### 🎯 Audit Objectives (/data-question-framing, /statistical-analysis, /scikit-learn)
1. **Integritas Skema & Anomali:** Memeriksa missing values, tipe data, sentinel values (`-1`), dan artefak data kotor (misal string status server).
2. **Keseimbangan Kelas & Polarisasi Label:** Memverifikasi proporsi target `ClassLabel` dan mendokumentasikan konvensi label (0 vs 1).
3. **Statistik Deskriptif Mendalam:** Menganalisis tendensi sentral (Mean vs Median gap), skewness, IQR, serta persentil $p1 \dots p99$.
4. **Uji Hipotesis & Daya Diskriminasi Fitur:** Melakukan Mann-Whitney U Test dan perhitungan Effect Size (Cohen's $d$) untuk mengetahui fitur pembeda utama.
5. **Multikolinearitas & Resiko Data Leakage:** Mengidentifikasi redundansi fitur dan bias sumber data (domain authority bias).
6. **Rencana Preprocessing Chapter 2:** Menyusun langkah *data cleanroom* sebelum modeling.
"""))

# Section 1.1
cells.append(nbf.v4.new_markdown_cell("""## 1.1 Problem Framing & Decision Canvas (/data-question-framing)

Sebelum mengeksekusi model machine learning, kita mendefinisikan *decision canvas*:
* **Unit of Analysis:** Satu baris URL tunggal beserta 16 fitur leksikal, struktural, dan statistik yang diekstrak.
* **Target Definition:** `ClassLabel`
  * ⚠️ **Temuan Penting Polarisasi Label:** Pada dataset LegitPhish, `0.0 = Phishing` dan `1.0 = Legitimate`. Ini berlawanan dengan konvensi biner deteksi intrusi umum (di mana 1 biasanya adalah ancaman). Keputusan: Kita dokumentasikan polaritas ini secara eksplisit dan merencanakan opsi transformasi pada tahap preprocessing.
* **Baseline vs Candidate Model:**
  * Baseline: `Random Forest Classifier` (evaluasi varians rendah, pohon independen).
  * Candidate: `XGBoost Classifier` (gradient boosting efisien menangani interaksi non-linear dan fitur leksikal bernilai ekstrem).
* **Guardrail Metric:** False Positive Rate (FPR) pada URL legitimasi harus ditekan sangat rendah agar pengguna tidak terganggu pemblokiran situs aman (false alarms), sedangkan Recall pada kelas Phishing harus dioptimalkan.
"""))

# Section 1.2
cells.append(nbf.v4.new_markdown_cell("""## 1.2 Data Ingestion & Structural Inspection"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic styling
sns.set_theme(style='whitegrid', font='sans-serif')
palette = {'Phishing (0)': '#e74c3c', 'Legitimate (1)': '#2ecc71'}

# Path dataset & output plots
DATA_PATH = Path('data/url/url_features_extracted1.csv')
PLOT_DIR = Path('plots/url')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Ingestion
df_raw = pd.read_csv(DATA_PATH)
print(f"Dataset berhasil dimuat!")
print(f"Total Baris: {df_raw.shape[0]:,}")
print(f"Total Kolom: {df_raw.shape[1]}")
df_raw.head(3)
"""))

# Section 1.3
cells.append(nbf.v4.new_markdown_cell("""## 1.3 Schema & Data Type Verification"""))

cells.append(nbf.v4.new_code_cell("""# Informasi tipe data dan pemakaian memori
schema_info = pd.DataFrame({
    'Column': df_raw.columns,
    'Non-Null Count': df_raw.notnull().sum().values,
    'Null Count': df_raw.isnull().sum().values,
    'Dtype': df_raw.dtypes.values
})
print("=== RINGKASAN SKEMA & TIPE DATA ===")
display(schema_info)

mem_mb = df_raw.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"Estimasi Penggunaan Memori: {mem_mb:.2f} MB")
"""))

# Section 1.4
cells.append(nbf.v4.new_markdown_cell("""## 1.4 Missing Values & Data Corruption Anomaly Audit

Kita memeriksa apakah terdapat nilai kosong (`NaN`), nilai tak hingga (`Inf`), atau nilai sentinel seperti `-1` pada penghitungan subdomain yang mengindikasikan anomali parser.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Cek Missing Value
null_series = df_raw.isnull().sum()
null_cols = null_series[null_series > 0]
print("=== AUDIT MISSING VALUES ===")
if len(null_cols) == 0:
    print("Tidak ditemukan missing value.")
else:
    for col, val in null_cols.items():
        print(f"Kolom '{col}': {val} missing value ({val/len(df_raw)*100:.4f}%)")

# Tampilkan baris yang memiliki NaN pada ClassLabel
null_row = df_raw[df_raw['ClassLabel'].isnull()]
print("\\nDetail baris dengan ClassLabel NaN:")
display(null_row[['URL', 'url_length', 'ClassLabel']])

# 2. Cek Nilai Sentinel / Anomali Negatif pada subdomain_count
negative_subdomains = (df_raw['subdomain_count'] < 0).sum()
print(f"\\nJumlah baris dengan subdomain_count < 0 (nilai sentinel -1): {negative_subdomains:,} baris")

# 3. Cek Artefak Teks Non-URL pada kolom 'URL'
status_artifacts = df_raw[df_raw['URL'].str.lower().isin(['offline', 'online', 'error', '404', 'nan'])]
print(f"Jumlah artefak string status crawler pada kolom URL: {len(status_artifacts)}")
display(status_artifacts[['URL', 'url_length', 'subdomain_count', 'ClassLabel']])
"""))

# Section 1.5
cells.append(nbf.v4.new_markdown_cell("""## 1.5 Deduplication & Integrity Audit

Mengecek duplikasi baris, duplikasi string URL, dan apakah ada inkonsistensi label (*conflicting labels*) di mana URL yang sama memiliki dua label berbeda.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Duplikasi Baris Lengkap
exact_duplicates = df_raw.duplicated().sum()
print(f"Duplikasi Baris Persis (Identik di semua kolom): {exact_duplicates:,}")

# 2. Duplikasi pada string URL
url_duplicates = df_raw['URL'].duplicated().sum()
print(f"Duplikasi pada Kolom URL: {url_duplicates:,}")

# 3. Uji Konsistensi Label pada URL yang duplikat
url_label_conflict = df_raw.groupby('URL')['ClassLabel'].nunique()
conflicting_urls = url_label_conflict[url_label_conflict > 1]
print(f"Jumlah URL dengan Label Bertentangan (Conflicting Labels): {len(conflicting_urls)}")

# Bersihkan 1 baris NaN untuk audit distribusi selanjutnya
df = df_raw.dropna(subset=['ClassLabel']).copy()
df['ClassLabel'] = df['ClassLabel'].astype(int)
df['Class_Name'] = df['ClassLabel'].map({0: 'Phishing (0)', 1: 'Legitimate (1)'})
"""))

# Section 1.6
cells.append(nbf.v4.new_markdown_cell("""## 1.6 Class Distribution & Imbalance Audit

Mengevaluasi distribusi target `ClassLabel`. Dataset yang memiliki rasio kelas seimbang memudahkan penentuan *threshold* klasifikasi standar ($0.5$).
"""))

cells.append(nbf.v4.new_code_cell("""class_counts = df['Class_Name'].value_counts()
class_pcts = df['Class_Name'].value_counts(normalize=True) * 100
imbalance_ratio = class_counts['Phishing (0)'] / class_counts['Legitimate (1)']

print("=== DISTRIBUSI KELAS TARGET ===")
for cname in class_counts.index:
    print(f"{cname}: {class_counts[cname]:,} sampel ({class_pcts[cname]:.2f}%)")
print(f"Imbalance Ratio (Phishing : Legitimate): {imbalance_ratio:.2f} : 1")

# Visualisasi Distribusi Kelas
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(class_counts.index, class_counts.values, color=['#e74c3c', '#2ecc71'], edgecolor='black', alpha=0.85)
for i, v in enumerate(class_counts.values):
    pct = v / len(df) * 100
    ax[0].text(i, v + 1000, f'{v:,}\\n({pct:.1f}%)', ha='center', fontweight='bold', fontsize=11)
ax[0].set_title('URL Class Distribution (Sample Count)', fontsize=13, fontweight='bold', pad=12)
ax[0].set_ylabel('Total Samples')
ax[0].set_ylim(0, max(class_counts.values) * 1.15)

ax[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'],
          startangle=140, explode=(0.04, 0), wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
          textprops={'fontsize': 11, 'fontweight': 'bold'})
ax[1].set_title(f'URL Class Proportion (Ratio {imbalance_ratio:.2f}:1)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plot_file1 = PLOT_DIR / '01_url_class_distribution.png'
plt.savefig(plot_file1, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {plot_file1}")
"""))

# Section 1.7
cells.append(nbf.v4.new_markdown_cell("""## 1.7 Statistical Profiling & Central Tendency (/statistical-analysis)

Sesuai metodologi `/statistical-analysis`:
* Untuk setiap variabel numerik, kita wajib membandingkan **Mean dan Median**. Jika terdapat selisih (*gap*) yang lebar, distribusi mengalami kemiringan (*skewness*) yang dipengaruhi oleh nilai ekstrim/outlier.
* Menghitung **Spread & Variabilitas**: Standar Deviasi ($Std$), $IQR = Q3 - Q1$, serta persentil detail ($p1, p5, p25, p50, p75, p90, p95, p99$).
"""))

cells.append(nbf.v4.new_code_cell("""num_cols = df.select_dtypes(include=[np.number]).columns.drop('ClassLabel').tolist()

stat_summary = []
for col in num_cols:
    s = df[col]
    mean_val = s.mean()
    median_val = s.median()
    iqr_val = s.quantile(0.75) - s.quantile(0.25)
    skew_val = s.skew()
    stat_summary.append({
        'Feature': col,
        'Mean': round(mean_val, 2),
        'Median': round(median_val, 2),
        'Std': round(s.std(), 2),
        'IQR': round(iqr_val, 2),
        'Skewness': round(skew_val, 2),
        'Min': round(s.min(), 2),
        'p1': round(s.quantile(0.01), 2),
        'p25': round(s.quantile(0.25), 2),
        'p50': round(median_val, 2),
        'p75': round(s.quantile(0.75), 2),
        'p99': round(s.quantile(0.99), 2),
        'Max': round(s.max(), 2)
    })

stat_df = pd.DataFrame(stat_summary)
print("=== STATISTICAL PROFILING 16 FITUR NUMERIK ===")
display(stat_df)
"""))

# Section 1.8
cells.append(nbf.v4.new_markdown_cell("""## 1.8 Bivariate Hypothesis Testing & Feature Discriminative Power

Untuk menentukan signifikansi perbedaan antara URL Phishing vs Legitimate:
1. **Uji Hipotesis Non-Parametrik (Mann-Whitney U Test):** Memeriksa apakah distribusi dua kelompok berbeda secara signifikan tanpa mengasumsikan normalitas data ($H_0$: distribusi identik).
2. **Effect Size (Cohen's $d$):** Mengukur besarnya efek praktis (*practical significance*), bukan hanya nilai $p$-value.
   $$\\text{Cohen's } d = \\frac{\\bar{X}_{\\text{phish}} - \\bar{X}_{\\text{legit}}}{S_{\\text{pooled}}}$$
"""))

cells.append(nbf.v4.new_code_cell("""phish_data = df[df['ClassLabel'] == 0]
legit_data = df[df['ClassLabel'] == 1]

tests = []
for col in num_cols:
    p_s = phish_data[col]
    l_s = legit_data[col]
    
    # Mann-Whitney U Test
    u_stat, p_val = stats.mannwhitneyu(p_s, l_s, alternative='two-sided')
    
    # Cohen's d
    n1, n2 = len(p_s), len(l_s)
    s1, s2 = p_s.std(), l_s.std()
    pooled_s = np.sqrt(((n1 - 1)*(s1**2) + (n2 - 1)*(s2**2)) / (n1 + n2 - 2))
    cohen_d = (p_s.mean() - l_s.mean()) / (pooled_s + 1e-9)
    
    tests.append({
        'Feature': col,
        'Phish_Mean': round(p_s.mean(), 3),
        'Phish_Median': round(p_s.median(), 3),
        'Legit_Mean': round(l_s.mean(), 3),
        'Legit_Median': round(l_s.median(), 3),
        'Cohen_d': round(cohen_d, 3),
        'Abs_Cohen_d': round(abs(cohen_d), 3),
        'p_value': p_val
    })

hypo_df = pd.DataFrame(tests).sort_values(by='Abs_Cohen_d', ascending=False)
print("=== HASIL UJI HIPOTESIS & PERINGKAT EFFECT SIZE FITUR ===")
display(hypo_df[['Feature', 'Phish_Mean', 'Phish_Median', 'Legit_Mean', 'Legit_Median', 'Cohen_d', 'p_value']])

# Visualisasi Ranking Cohen's d
plt.figure(figsize=(12, 7))
eff_sorted = hypo_df.sort_values(by='Abs_Cohen_d', ascending=True)
bar_colors = ['#e74c3c' if x > 0 else '#3498db' for x in eff_sorted['Cohen_d']]
bars = plt.barh(eff_sorted['Feature'], eff_sorted['Cohen_d'], color=bar_colors, edgecolor='black', alpha=0.85)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title("Feature Discriminative Power (Cohen's d Effect Size)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Cohen's d (>0 Lebih Tinggi di Phishing, <0 Lebih Tinggi di Legitimate)")
for bar in bars:
    w = bar.get_width()
    ha = 'left' if w > 0 else 'right'
    offset = 0.15 if w > 0 else -0.15
    plt.text(w + offset, bar.get_y() + bar.get_height()/2, f'{w:.2f}', va='center', ha=ha, fontsize=10, fontweight='bold')
plt.tight_layout()
plot_file4 = PLOT_DIR / '04_url_effect_size_ranking.png'
plt.savefig(plot_file4, dpi=200, bbox_inches='tight')
plt.show()
"""))

# Section 1.9 Visual Boxplots
cells.append(nbf.v4.new_markdown_cell("""### Visualisasi Perbandingan Fitur Utama (Boxplot)"""))

cells.append(nbf.v4.new_code_cell("""top_cols = ['percentage_numeric_chars', 'number_of_digits', 'token_count', 'dot_count', 'url_length', 'tld_length']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, feat in enumerate(top_cols):
    sns.boxplot(data=df, x='Class_Name', y=feat, hue='Class_Name', ax=axes[idx], palette=palette, legend=False, showmeans=True,
                meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'6'},
                fliersize=2)
    axes[idx].set_title(f'Distribution of {feat}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('')
plt.suptitle('Lexical & Structural Feature Comparison: Phishing vs Legitimate URLs', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plot_file2 = PLOT_DIR / '02_url_top_features_comparison.png'
plt.savefig(plot_file2, dpi=200, bbox_inches='tight')
plt.show()
"""))

# Section 1.10 Collinearity
cells.append(nbf.v4.new_markdown_cell("""## 1.9 Multicollinearity & Feature Correlation Analysis

Menganalisis korelasi antar fitur untuk mendeteksi redundansi (*multicollinearity*) yang dapat mempengaruhi interpretasi model dan feature importances.
"""))

cells.append(nbf.v4.new_code_cell("""# Correlation Matrix Heatmap
corr = df[num_cols + ['ClassLabel']].corr()

plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of URL Features & ClassLabel', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plot_file3 = PLOT_DIR / '03_url_correlation_matrix.png'
plt.savefig(plot_file3, dpi=200, bbox_inches='tight')
plt.show()

# Identifikasi pasangan dengan korelasi tinggi (|r| > 0.70)
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        c1, c2 = corr.columns[i], corr.columns[j]
        val = corr.iloc[i, j]
        if abs(val) >= 0.70 and c1 != 'ClassLabel' and c2 != 'ClassLabel':
            high_corr.append({'Feature 1': c1, 'Feature 2': c2, 'Correlation': round(val, 3)})

print("=== PASANGAN FITUR DENGAN MULTIKOLINEARITAS TINGGI (|r| >= 0.70) ===")
display(pd.DataFrame(high_corr))

# Scatter URL Entropy vs Length
plt.figure(figsize=(11, 5.5))
sample_plot = df.sample(10000, random_state=42)
sns.scatterplot(data=sample_plot, x='url_length', y='url_entropy', hue='Class_Name', palette=palette, alpha=0.4, s=20)
plt.title('Shannon Entropy vs URL Length by Class (Sample n=10,000)', fontsize=13, fontweight='bold')
plt.xlabel('URL Length')
plt.ylabel('URL Entropy (Shannon)')
plt.xlim(0, 200)
plt.tight_layout()
plot_file5 = PLOT_DIR / '05_url_entropy_vs_length.png'
plt.savefig(plot_file5, dpi=200, bbox_inches='tight')
plt.show()
"""))

# Section 1.11 Data Leakage & Biases
cells.append(nbf.v4.new_markdown_cell("""## 1.10 Threat Validity & Data Source Bias Audit

### 🚨 Temuan Bias Sumber Data:
1. **Bias Protokol HTTPS (`https_flag`):**
   * Pada kelas legitimate, $100\%$ URL menggunakan protokol HTTPS.
   * Pada kelas phishing, hanya $5.5\\%$ URL yang menggunakan HTTPS.
   * *Critical Threat Assessment:* Di dunia nyata saat ini (2025/2026), sebagian besar penyerang phishing telah menggunakan SSL/TLS gratis (misal Let's Encrypt). Mempercayai `https_flag` sebagai fitur utama dapat menyebabkan *false negatives* tinggi saat dideploy ke traffic langsung.
2. **Ketergantungan Sumber Legitimasi:**
   * URL kelas legitimate didominasi oleh Stack Overflow dan Wikipedia dengan path yang seragam (misal `/questions/tagged/`).
   * Hal ini menyebabkan metrik seperti `path_length` dan `domain_name_length` memiliki distribusi yang terikat erat dengan domain-domain tertentu (*corpus sampling bias*).
"""))

# Section 1.12 Action Checklist & Plan
cells.append(nbf.v4.new_markdown_cell("""## 1.11 Audit Summary & Cleanroom Preprocessing Plan for Chapter 2 (/data-question-framing)

### 📋 Ringkasan Keputusan Audit (Audit Decision: `fix-first`)

| Item Audit | Temuan Empiris | Tindakan Cleanroom untuk Chapter 2 |
| :--- | :--- | :--- |
| **Missing Values** | 1 baris NaN pada `ClassLabel` (URL: `chocolatleroux.com`) | Drop baris tersebut secara permanen. |
| **Baris Duplikat** | 346 baris duplikat persis | Lakukan deduplikasi (`drop_duplicates()`) sebelum pembagian split. |
| **Sentinel Values** | 1,308 baris memiliki `subdomain_count == -1` | Imputasi atau ganti sentinel $-1$ menjadi $0$ atau representasi missing terpisah. |
| **Artefak Teks Non-URL** | Baris 599 berisi teks literal `"Offline"` | Drop baris karena bukan URL valid. |
| **Polaritas Label** | `0.0 = Phishing`, `1.0 = Legitimate` | Pertimbangkan inverter label (1 = Phishing, 0 = Legitimate) agar metrik Precision/Recall terarah ke deteksi ancaman. |
| **Multikolinearitas** | `number_of_digits` vs `percentage_numeric_chars` ($r = 0.88$) | Evaluasi eliminasi atau regularisasi pada model tree. |
| **Strategi Split Data** | Rasio 63k : 37k (Imbalance 1.7:1) | Gunakan `StratifiedKFold` atau `Stratified train_test_split` ($80:20$). |
| **Model Candidate Readiness** | Baseline `Random Forest` & Target `XGBoost` | Lanjutkan ke Chapter 2 untuk implementasi data pipeline scikit-learn & pelatihan model. |
"""))

nb.cells = cells

# Save notebook
with open('url.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("File url.ipynb berhasil dibuat!")
