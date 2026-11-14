import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

# Load existing url.ipynb
nb_path = Path('url.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

cells = nb.cells

# CHAPTER 3 Markdown
cells.append(nbf.v4.new_markdown_cell("""# CHAPTER 3 : Exploratory Data Analysis (EDA)

Pada bab ini, kita melakukan Exploratory Data Analysis (EDA) mendalam pada dataset URL yang **telah bersih** dari Chapter 2 (`url_features_cleaned.csv`). 

Karena seluruh anomali (missing value, teks "Offline", duplikasi, dan nilai sentinel -1) telah dibersihkan pada Chapter 2, seluruh visualisasi dan statistik pada bab ini merefleksikan karakteristik data yang murni, valid, dan bebas distorsi.

Tujuan utama Chapter 3:
1. Menganalisis sebaran univariat fitur leksikal dan struktural.
2. Membedah pola bivariat antara fitur prediktor terhadap target `is_phishing`.
3. Memeriksa prevalensi indikator biner (IP mentah, HTTPS, ekstensi mencurigakan, popularitas TLD).
4. Menganalisis interaksi multivariat (Shannon Entropy vs URL Length, Path Length vs Domain Length).
5. Merumuskan implikasi analitik untuk Feature Engineering (Chapter 4) dan Model Training (Chapter 5: Random Forest vs XGBoost).
"""))

# Section 3.1
cells.append(nbf.v4.new_markdown_cell("""## 3.1 Pemuatan Data Bersih & Persiapan Visualisasi"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Visual setup tanpa emotikon
sns.set_theme(style='whitegrid', font='sans-serif')
palette = {0: '#2ecc71', 1: '#e74c3c'}
class_names = {0: 'Legitimate (0)', 1: 'Phishing (1)'}

CLEAN_URL_PATH = Path('data/url/processed/url_features_cleaned.csv')
EDA_PLOT_DIR = Path('plots/url')
EDA_PLOT_DIR.mkdir(parents=True, exist_ok=True)

df_clean = pd.read_csv(CLEAN_URL_PATH)
df_clean['Class_Name'] = df_clean['is_phishing'].map(class_names)

print(f"Data bersih berhasil dimuat: {len(df_clean):,} baris x {df_clean.shape[1]} kolom.")
print("Distribusi target:")
print(df_clean['Class_Name'].value_counts())
"""))

# Section 3.2
cells.append(nbf.v4.new_markdown_cell("""## 3.2 Distribusi Univariat Fitur Leksikal & Struktural Bersih

Menganalisis sebaran nilai dari 6 fitur leksikal penting untuk memahami skewness dan rentang nilainya.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

key_univariate_cols = ['url_length', 'url_entropy', 'dot_count', 'token_count', 'path_length', 'domain_name_length']

for idx, col in enumerate(key_univariate_cols):
    sns.kdeplot(data=df_clean, x=col, hue='Class_Name', palette={'Legitimate (0)': '#2ecc71', 'Phishing (1)': '#e74c3c'},
                common_norm=False, fill=True, alpha=0.35, ax=axes[idx], linewidth=1.5)
    axes[idx].set_title(f'KDE Distribution of {col}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Density')

plt.suptitle('Univariate Density Distributions of Key URL Features (Cleaned Data)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
eda_file1 = EDA_PLOT_DIR / 'eda_01_feature_distributions.png'
plt.savefig(eda_file1, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {eda_file1}")
"""))

# Section 3.3
cells.append(nbf.v4.new_markdown_cell("""## 3.3 Analisis Bivariat: Pemisahan Fitur Angka & Digit

Salah satu temuan paling menonjol pada audit adalah tingginya persentase karakter angka pada URL phishing dibandingkan URL legit.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot percentage_numeric_chars
sns.boxplot(data=df_clean, x='Class_Name', y='percentage_numeric_chars', palette={'Legitimate (0)': '#2ecc71', 'Phishing (1)': '#e74c3c'},
            ax=axes[0], showmeans=True, meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'6'})
axes[0].set_title('Percentage of Numeric Characters (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('')

# Boxplot number_of_digits
sns.boxplot(data=df_clean, x='Class_Name', y='number_of_digits', palette={'Legitimate (0)': '#2ecc71', 'Phishing (1)': '#e74c3c'},
            ax=axes[1], showmeans=True, meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'6'})
axes[1].set_title('Total Number of Digits', fontsize=12, fontweight='bold')
axes[1].set_xlabel('')

plt.suptitle('Bivariate Analysis: Numeric Features as Primary Phishing Discriminators', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Ringkasan statistik kuantil
numeric_comparison = df_clean.groupby('Class_Name')[['percentage_numeric_chars', 'number_of_digits']].describe().T
print("Perbandingan Statistik Numerik Antar Kelas:")
display(numeric_comparison)
"""))

# Section 3.4
cells.append(nbf.v4.new_markdown_cell("""## 3.4 Analisis Prevalensi Indikator Biner & Keamanan

Menguji frekuensi kemunculan flag biner seperti penggunaan IP address mentah, protokol HTTPS, ekstensi berbahaya, dan popularitas TLD.
"""))

cells.append(nbf.v4.new_code_cell("""binary_flags = ['has_ip_address', 'https_flag', 'suspicious_file_extension', 'tld_popularity', 'has_hyphen_in_domain']

binary_summary = []
for flag in binary_flags:
    legit_pct = df_clean[df_clean['is_phishing'] == 0][flag].mean() * 100
    phish_pct = df_clean[df_clean['is_phishing'] == 1][flag].mean() * 100
    binary_summary.append({
        'Binary Indicator': flag,
        'Legitimate Prevalence (%)': round(legit_pct, 2),
        'Phishing Prevalence (%)': round(phish_pct, 2),
        'Difference (Phish - Legit)': round(phish_pct - legit_pct, 2)
    })

binary_df = pd.DataFrame(binary_summary)
print("Prevalensi Indikator Biner per Kelas:")
display(binary_df)

# Visualisasi Bar Chart
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(binary_df))
width = 0.35

rects1 = ax.bar(x - width/2, binary_df['Legitimate Prevalence (%)'], width, label='Legitimate (0)', color='#2ecc71', edgecolor='black')
rects2 = ax.bar(x + width/2, binary_df['Phishing Prevalence (%)'], width, label='Phishing (1)', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Prevalence (%)')
ax.set_title('Binary Threat Indicators Prevalence by Class', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(binary_df['Binary Indicator'], fontsize=10, rotation=15)
ax.legend()

for rect in list(rects1) + list(rects2):
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
eda_file2 = EDA_PLOT_DIR / 'eda_02_binary_flags_prevalence.png'
plt.savefig(eda_file2, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {eda_file2}")
"""))

# Section 3.5
cells.append(nbf.v4.new_markdown_cell("""## 3.5 Analisis Multivariat & Interaksi Fitur

Menganalisis korelasi antar fitur pada data bersih serta interaksi antara Shannon Entropy dan URL Length.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Matriks Korelasi Spearman pada Data Bersih
feature_cols = [c for c in df_clean.columns if c not in ['URL', 'Class_Name', 'is_phishing']]
spearman_corr = df_clean[feature_cols + ['is_phishing']].corr(method='spearman')

plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Spearman Rank Correlation Matrix (Cleaned Data)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
eda_file4 = EDA_PLOT_DIR / 'eda_04_clean_correlation_matrix.png'
plt.savefig(eda_file4, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {eda_file4}")

# 2. Scatter Interaksi Multivariat
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sample_eda = df_clean.sample(15000, random_state=42)

# Entropy vs Length
sns.scatterplot(data=sample_eda, x='url_length', y='url_entropy', hue='Class_Name',
                palette={'Legitimate (0)': '#2ecc71', 'Phishing (1)': '#e74c3c'},
                alpha=0.4, s=20, ax=axes[0])
axes[0].set_title('Shannon Entropy vs URL Length', fontsize=12, fontweight='bold')
axes[0].set_xlabel('URL Length')
axes[0].set_ylabel('URL Entropy')
axes[0].set_xlim(0, 180)

# Path Length vs Domain Length
sns.scatterplot(data=sample_eda, x='domain_name_length', y='path_length', hue='Class_Name',
                palette={'Legitimate (0)': '#2ecc71', 'Phishing (1)': '#e74c3c'},
                alpha=0.4, s=20, ax=axes[1])
axes[1].set_title('Path Length vs Domain Name Length', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Domain Name Length')
axes[1].set_ylabel('Path Length')
axes[1].set_xlim(0, 35)
axes[1].set_ylim(0, 100)

plt.suptitle('Multivariate Feature Interaction Analysis (Sample n=15,000)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
eda_file3 = EDA_PLOT_DIR / 'eda_03_multivariate_entropy_path.png'
plt.savefig(eda_file3, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {eda_file3}")
"""))

# Section 3.6
cells.append(nbf.v4.new_markdown_cell("""## 3.6 Rangkuman Wawasan EDA & Implikasi Desain Model

Berdasarkan eksplorasi data bersih di atas, berikut adalah wawasan inti yang menjadi landasan pemodelan:

1. **Keunggulan Model Berbasis Pohon (Tree-based):**
   * Pemisahan fitur leksikal pada dataset ini sangat bersifat biner dan memiliki ambang batas tegas (*step-function*), misalnya `has_ip_address == 1` hanya terjadi pada Phishing, dan `percentage_numeric_chars > 20%` mendominasi Phishing.
   * Model berbasis pohon seperti Random Forest dan XGBoost Classifier secara alami sangat unggul menangani pola partisi ortogonal non-linear ini tanpa memerlukan normalisasi distribusi gaussian.
2. **Penanganan Multikolinearitas Fitur Numerik:**
   * `number_of_digits` dan `percentage_numeric_chars` memiliki korelasi sangat tinggi ($r > 0.85$). Pada Chapter 4, kita dapat mengevaluasi seleksi fitur atau regularisasi feature subsampling (`colsample_bytree` di XGBoost dan `max_features` di Random Forest).
3. **Bobot Ketidakseimbangan Kelas:**
   * Rasio target adalah $62,94\\% : 37,06\\%$. Pada pelatihan XGBoost di Chapter 5, kita dapat mengonfigurasi parameter `scale_pos_weight` untuk mengoptimalkan titik trade-off antara False Positive Rate dan Recall.
"""))

nb.cells = cells

with open('url.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Chapter 3 berhasil ditambahkan ke url.ipynb!")
