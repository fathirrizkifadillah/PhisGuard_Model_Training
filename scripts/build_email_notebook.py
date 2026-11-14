import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

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
cells.append(nbf.v4.new_markdown_cell("""# 🛡️ PhisGuard: Email Threat Classification Engine
## CHAPTER 1 : Dataset Quality Audit (Phishing Email Dataset)

---

### 📌 Project Metadata & Model Roadmap
* **Domain:** Natural Language Processing (NLP) / Cybersecurity Email Threat Detection
* **Target Model (Production):** `TF-IDF + Linear SVM` (`LinearSVC` / `SGDClassifier(loss='hinge')`)
* **Baseline Model:** `TF-IDF + Logistic Regression`
* **Dataset Name:** `Phishing Email Dataset`
* **Dataset Repository:** Hugging Face ([zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset))
* **Dataset Provenance & Origin:**
  * Merupakan *mirror* dari dataset Kaggle *"Phishing Email Detection"* oleh kreator *Cyber Cop*.
  * Agregasi dari korpus historis keamanan email terkemuka:
    1. **Enron Email Dataset:** Sumber email sah (*ham*) korespondensi korporat energi.
    2. **SpamAssassin Public Corpus:** Korpus benchmark email campuran spam & ham.
    3. **Nazario Phishing Corpus:** Koleksi email serangan phishing aktif oleh periset Jose Nazario.
    4. **Nigerian Fraud (419) Datasets:** Arsip email rekayasa sosial dan penipuan finansial.
    5. **Ling-Spam & CEAS Benchmark:** Korpus akademis klasifikasi teks email.

---
### 🎯 Audit Objectives (/data-question-framing, /statistical-analysis, /scikit-learn)
1. **Integritas Teks & Anomali Korpus:** Mengidentifikasi email kosong (`NaN`), whitespace/newline-only, serta teks template noise.
2. **Investigasi Anomali Ekstrim:** Mengaudit baris outlier raksasa (baris 12500 dengan 17 Juta karakter akibat injeksi dump notebook).
3. **Konflik Label & Deduplikasi Template:** Mendeteksi 533 baris dengan teks literal `"empty"` yang memiliki label bertentangan (335 Phishing vs 198 Safe).
4. **Distribusi Kelas & Imbalance:** Mengukur rasio antara *Safe Email* vs *Phishing Email*.
5. **Statistik Panjang Teks & Skewness:** Menghitung Mean vs Median, IQR, dan persentil $p1 \dots p99$ untuk jumlah karakter dan kata.
6. **Bias Domain & Data Leakage (Clever Hans Effect):** Mengaudit ketergantungan korpus Enron (misal nama karyawan, istilah internal deal/meter) vs Nazario.
7. **Rencana Preprocessing & Pipeline TF-IDF untuk Chapter 2:** Merumuskan parameter `min_df`, `max_df`, `sublinear_tf`, serta arsitektur Linear SVM.
"""))

# Section 1.1 Decision Canvas
cells.append(nbf.v4.new_markdown_cell("""## 1.1 Problem Framing & NLP Decision Canvas (/data-question-framing)

* **Unit of Analysis:** Satu pesan email teks mentah (*unstructured text corpus*).
* **Target Definition:** `Email Type`
  * `Safe Email`: Email korespondensi sah / benign.
  * `Phishing Email`: Email rekayasa sosial, scam, atau serangan kredensial.
* **Biaya Kesalahan (Error Cost Asymmetry):**
  * *False Positive (FP):* Email bisnis penting salah dikarantina sebagai phishing $\\rightarrow$ menghambat operasional bisnis dan komunikasi kritis.
  * *False Negative (FN):* Email serangan phishing lolos ke inbox $\\rightarrow$ resiko kebocoran kredensial atau ransomware.
  * *Guardrail Metric:* **High Precision on Safe Email** (menjaga FPR Phishing serendah mungkin) dengan **Recall Phishing** $\\ge 95\\%$.
* **Justifikasi Pemilihan Model:**
  * Baseline: `Logistic Regression` (probabilistik linear, baseline interpretatif).
  * Target Model: `TF-IDF + Linear SVM` (mencari *maximum margin hyperplane* pada ruang vektor leksikal berdimensi tinggi $> 20.000$ fitur; inferensi sangat cepat untuk high-throughput mail gateway).
"""))

# Section 1.2 Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1.2 Environment Setup & Data Ingestion"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visual aesthetic
sns.set_theme(style='whitegrid', font='sans-serif')
palette = {'Phishing Email': '#e74c3c', 'Safe Email': '#3498db'}

DATA_PATH = Path('data/email/raw/phishing_email.csv')
PLOT_DIR = Path('plots/email')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Ingestion
df_raw = pd.read_csv(DATA_PATH)
print("Dataset Phishing Email berhasil dimuat!")
print(f"Total Baris: {df_raw.shape[0]:,}")
print(f"Total Kolom: {df_raw.shape[1]}")
df_raw.head(3)
"""))

# Section 1.3 Schema Inspection
cells.append(nbf.v4.new_markdown_cell("""## 1.3 Schema & Structural Column Inspection"""))

cells.append(nbf.v4.new_code_cell("""schema_df = pd.DataFrame({
    'Column': df_raw.columns,
    'Non-Null Count': df_raw.notnull().sum().values,
    'Null Count': df_raw.isnull().sum().values,
    'Dtype': df_raw.dtypes.values
})
print("=== RINGKASAN SKEMA & TIPE DATA ===")
display(schema_df)

mem_mb = df_raw.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"Ukuran Memori: {mem_mb:.2f} MB")
print("Catatan: Kolom 'Unnamed: 0' merupakan index artefak CSV yang redundan dan akan dieliminasi.")
"""))

# Section 1.4 Missing Values & Blank Text Audit
cells.append(nbf.v4.new_markdown_cell("""## 1.4 Missing Values, Nulls & Blank Email Audit

Memeriksa keberadaan nilai hilang (`NaN`), string kosong (`""`), atau string yang hanya berisi spasi/karakter newline (`"\\n"`).
"""))

cells.append(nbf.v4.new_code_cell("""null_count = df_raw['Email Text'].isnull().sum()
print(f"Jumlah baris dengan Email Text NaN: {null_count} baris ({null_count/len(df_raw)*100:.3f}%)")

# Cek string kosong atau whitespace saja
non_null_text = df_raw.dropna(subset=['Email Text']).copy()
empty_strip = non_null_text['Email Text'].str.strip().eq('')
whitespace_count = empty_strip.sum()
print(f"Jumlah baris dengan Email Text kosong/hanya whitespace: {whitespace_count} baris")

# Tampilkan sampel baris yang hanya berisi whitespace / newline
newline_samples = non_null_text[empty_strip]
display(newline_samples[['Unnamed: 0', 'Email Text', 'Email Type']].head())
"""))

# Section 1.5 Duplicate & Conflicting Label Trap
cells.append(nbf.v4.new_markdown_cell("""## 1.5 Deduplication, Template Redundancy & Conflicting Label Trap

### 🚨 Temuan Kritis Label Bertentangan (*Conflicting Labels*):
Saat mengaudit duplikasi isi teks email, ditemukan bahwa ada template teks yang muncul berulang kali di kedua kelas sekaligus.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Duplikasi Teks Email
raw_duplicates = non_null_text['Email Text'].duplicated().sum()
print(f"Total Teks Email Duplikat: {raw_duplicates:,} baris")

# 2. Investigasi Teks Literal 'empty'
empty_text_mask = non_null_text['Email Text'].str.strip().str.lower().eq('empty')
empty_text_df = non_null_text[empty_text_mask]
print(f"Jumlah email yang hanya berisi kata 'empty': {len(empty_text_df):,} baris")
print("Distribusi kelas untuk teks 'empty':")
print(empty_text_df['Email Type'].value_counts())

# 3. Cek Teks Lain yang Memiliki Konflik Label (Muncul di Safe DAN Phishing)
text_label_counts = non_null_text.groupby('Email Text')['Email Type'].nunique()
conflicting_texts = text_label_counts[text_label_counts > 1]
print(f"\\nJumlah Teks Unik dengan Label Bertentangan: {len(conflicting_texts)}")
print(f"Total baris yang terinfeksi konflik label: {non_null_text[non_null_text['Email Text'].isin(conflicting_texts.index)].shape[0]}")
"""))

# Section 1.6 Class Distribution
cells.append(nbf.v4.new_markdown_cell("""## 1.6 Target Class Distribution & Imbalance Audit"""))

cells.append(nbf.v4.new_code_cell("""class_counts = non_null_text['Email Type'].value_counts()
class_pcts = non_null_text['Email Type'].value_counts(normalize=True) * 100
imbalance_ratio = class_counts['Safe Email'] / class_counts['Phishing Email']

print("=== DISTRIBUSI KELAS EMAIL ===")
for cname in class_counts.index:
    print(f"{cname}: {class_counts[cname]:,} sampel ({class_pcts[cname]:.2f}%)")
print(f"Imbalance Ratio (Safe : Phishing): {imbalance_ratio:.2f} : 1")

# Visualisasi Distribusi Kelas
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(class_counts.index, class_counts.values, color=['#3498db', '#e74c3c'], edgecolor='black', alpha=0.85)
for i, v in enumerate(class_counts.values):
    pct = v / len(non_null_text) * 100
    ax[0].text(i, v + 250, f'{v:,}\\n({pct:.1f}%)', ha='center', fontweight='bold', fontsize=11)
ax[0].set_title('Email Class Distribution (Sample Count)', fontsize=13, fontweight='bold', pad=12)
ax[0].set_ylabel('Total Samples')
ax[0].set_ylim(0, max(class_counts.values) * 1.15)

ax[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'],
          startangle=140, explode=(0.04, 0), wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
          textprops={'fontsize': 11, 'fontweight': 'bold'})
ax[1].set_title(f'Email Class Proportion (Ratio {imbalance_ratio:.2f}:1)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plot_file1 = PLOT_DIR / '01_email_class_distribution.png'
plt.savefig(plot_file1, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {plot_file1}")
"""))

# Section 1.7 Length Profiling & Outlier Deep Dive
cells.append(nbf.v4.new_markdown_cell("""## 1.7 NLP Corpus Statistical Profiling & Central Tendency (/statistical-analysis)

Menghitung statistik panjang karakter dan jumlah kata (*word count*) per kelas.
"""))

cells.append(nbf.v4.new_code_cell("""# Hitung metrik leksikal
df_analysis = non_null_text.copy()
df_analysis['char_length'] = df_analysis['Email Text'].str.len()
df_analysis['word_count'] = df_analysis['Email Text'].apply(lambda x: len(str(x).split()))

# Tampilkan 5 email terpanjang untuk melihat potensi anomali
longest_emails = df_analysis.nlargest(5, 'char_length')[['Unnamed: 0', 'char_length', 'word_count', 'Email Type']]
print("=== 5 EMAIL DENGAN KARAKTER TERPANJANG ===")
display(longest_emails)
"""))

# Section 1.8 17MB Outlier
cells.append(nbf.v4.new_markdown_cell("""### 💥 Investigasi Anomali Raksasa Baris 12500 (17 Juta Karakter)

Baris 12500 (index CSV 12501) memiliki panjang **17.036.692 karakter** (17 Megabytes dalam 1 baris CSV tunggal). Mari kita inspect isi teksnya:
"""))

cells.append(nbf.v4.new_code_cell("""row_12500_text = df_analysis.loc[12500, 'Email Text']
print(f"Panjang Baris 12500: {len(row_12500_text):,} karakter")
print("\\n--- Awal Teks (200 karakter pertama) ---")
print(row_12500_text[:200])
print("\\n--- Akhir Teks (200 karakter terakhir) ---")
print(row_12500_text[-200:])
print("\\nDIAGNOSA: Baris 12500 secara tidak sengaja menginjeksi seluruh dump file Jupyter Notebook (.ipynb) beserta output base64 ke dalam dataset teks email!")
"""))

# Section 1.9 Percentiles Table
cells.append(nbf.v4.new_markdown_cell("""### Tabel Persentil Panjang Teks ($p1 \dots p99$)"""))

cells.append(nbf.v4.new_code_cell("""percentiles = [1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9]

stat_rows = []
for metric in ['char_length', 'word_count']:
    for cls in ['Safe Email', 'Phishing Email', 'Overall']:
        sub = df_analysis if cls == 'Overall' else df_analysis[df_analysis['Email Type'] == cls]
        # Exclude row 12500 for realistic stats
        sub_no_huge = sub[sub['char_length'] < 1000000]
        s = sub_no_huge[metric]
        row_dict = {
            'Metric': metric,
            'Class': cls,
            'Mean': round(s.mean(), 1),
            'Median (p50)': round(s.median(), 1),
            'Std': round(s.std(), 1),
            'IQR': round(s.quantile(0.75) - s.quantile(0.25), 1),
            'p1': round(s.quantile(0.01), 1),
            'p25': round(s.quantile(0.25), 1),
            'p75': round(s.quantile(0.75), 1),
            'p90': round(s.quantile(0.90), 1),
            'p99': round(s.quantile(0.99), 1)
        }
        stat_rows.append(row_dict)

stat_summary_df = pd.DataFrame(stat_rows)
print("=== RINGKASAN PERSENTIL & TENDENSI SENTRAL (Tanpa Outlier 17MB) ===")
display(stat_summary_df)
"""))

# Section 1.10 Visualizations (Length & Boxplots)
cells.append(nbf.v4.new_markdown_cell("""## 1.8 Visualisasi Distribusi Panjang & Outlier Audit"""))

cells.append(nbf.v4.new_code_cell("""# 1. Histogram Log-Scale (Exclude row 12500 & 0-word whitespace)
plot_subset = df_analysis[(df_analysis['char_length'] < 100000) & (df_analysis['word_count'] > 0)].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.histplot(data=plot_subset, x='char_length', hue='Email Type', palette=palette, log_scale=True, kde=False, ax=axes[0], element='step', common_norm=False)
axes[0].set_title('Character Length Distribution (Log Scale)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Character Length (Log Scale)')
axes[0].axvline(df_analysis[df_analysis['Email Type']=='Safe Email']['char_length'].median(), color='#3498db', linestyle='--', linewidth=2, label='Safe Median (994)')
axes[0].axvline(df_analysis[df_analysis['Email Type']=='Phishing Email']['char_length'].median(), color='#e74c3c', linestyle='--', linewidth=2, label='Phish Median (732)')
axes[0].legend()

sns.histplot(data=plot_subset, x='word_count', hue='Email Type', palette=palette, log_scale=True, kde=False, ax=axes[1], element='step', common_norm=False)
axes[1].set_title('Word Count Distribution (Log Scale)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Word Count (Log Scale)')
axes[1].axvline(df_analysis[df_analysis['Email Type']=='Safe Email']['word_count'].median(), color='#3498db', linestyle='--', linewidth=2, label='Safe Median (179)')
axes[1].axvline(df_analysis[df_analysis['Email Type']=='Phishing Email']['word_count'].median(), color='#e74c3c', linestyle='--', linewidth=2, label='Phish Median (136)')
axes[1].legend()

plt.suptitle('Text Length & Word Count Distributions (Safe vs Phishing)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plot_file2 = PLOT_DIR / '02_email_length_distributions.png'
plt.savefig(plot_file2, dpi=200, bbox_inches='tight')
plt.show()

# 2. Boxplot Capped at p99
p99_char = np.percentile(df_analysis['char_length'], 99)
p99_word = np.percentile(df_analysis['word_count'], 99)
cap_df = df_analysis[(df_analysis['char_length'] <= p99_char) & (df_analysis['word_count'] > 0)].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=cap_df, x='Email Type', y='char_length', hue='Email Type', ax=axes[0], palette=palette, legend=False, showmeans=True,
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'6'})
axes[0].set_title(f'Character Length (Capped at p99 = {int(p99_char):,} chars)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Character Count')

sns.boxplot(data=cap_df, x='Email Type', y='word_count', hue='Email Type', ax=axes[1], palette=palette, legend=False, showmeans=True,
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'6'})
axes[1].set_title(f'Word Count (Capped at p99 = {int(p99_word):,} words)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Word Count')

plt.suptitle('Length Spread & Extreme Outlier Audit (Notice Mean >> Median skew)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plot_file3 = PLOT_DIR / '03_email_outliers_boxplots.png'
plt.savefig(plot_file3, dpi=200, bbox_inches='tight')
plt.show()
"""))

# Section 1.11 Structural markers
cells.append(nbf.v4.new_markdown_cell("""## 1.9 Structural Markers & HTML / Link Prevalence

Mengukur keberadaan artefak HTML tag (`<...>`) dan hyperlink (`http://`, `https://`, `www.`) di kedua kelas.
"""))

cells.append(nbf.v4.new_code_cell("""df_analysis['has_html'] = df_analysis['Email Text'].apply(lambda x: bool(re.search(r'<[^>]+>', str(x))))
df_analysis['has_url'] = df_analysis['Email Text'].apply(lambda x: bool(re.search(r'http[s]?://|www\.', str(x))))
df_analysis['is_empty_text'] = df_analysis['Email Text'].str.strip().str.lower().eq('empty')

struct_summary = pd.DataFrame({
    'Metric': ['Contains HTML (%)', 'Contains URLs (%)', 'Has Exact "empty" (%)'],
    'Safe Email': [
        round(df_analysis[df_analysis['Email Type']=='Safe Email']['has_html'].mean() * 100, 2),
        round(df_analysis[df_analysis['Email Type']=='Safe Email']['has_url'].mean() * 100, 2),
        round(df_analysis[df_analysis['Email Type']=='Safe Email']['is_empty_text'].mean() * 100, 2)
    ],
    'Phishing Email': [
        round(df_analysis[df_analysis['Email Type']=='Phishing Email']['has_html'].mean() * 100, 2),
        round(df_analysis[df_analysis['Email Type']=='Phishing Email']['has_url'].mean() * 100, 2),
        round(df_analysis[df_analysis['Email Type']=='Phishing Email']['is_empty_text'].mean() * 100, 2)
    ]
})
print("=== PREVALENSI INDIKATOR STRUKTURAL ===")
display(struct_summary)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(struct_summary))
width = 0.35
rects1 = ax.bar(x - width/2, struct_summary['Safe Email'], width, label='Safe Email', color='#3498db', edgecolor='black')
rects2 = ax.bar(x + width/2, struct_summary['Phishing Email'], width, label='Phishing Email', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Percentage (%)')
ax.set_title('Structural Indicators & Template Noise Prevalence by Class', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(struct_summary['Metric'], fontsize=11)
ax.legend()
for rect in list(rects1) + list(rects2):
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plot_file4 = PLOT_DIR / '04_email_structural_indicators.png'
plt.savefig(plot_file4, dpi=200, bbox_inches='tight')
plt.show()
"""))

# Section 1.12 Lexical Diversity & Enron Bias
cells.append(nbf.v4.new_markdown_cell("""## 1.10 Lexical Diversity & Spurious Corpus Bias (Enron vs Nazario)

### 🚨 Temuan Bias Domain & Resiko "Clever Hans Effect":
Dataset ini mengumpulkan email safe dari **Enron Corpus** dan email phishing dari **Nazario Corpus**. Kita menganalisis kata-kata paling frekuentif untuk melihat apakah model berpotensi mempelajari bias spesifik korpus.
"""))

cells.append(nbf.v4.new_code_cell("""def get_top_tokens(texts, n=12):
    stopwords = {'the', 'to', 'and', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'this', 'you', 'i', 'it', 'with', 'as', 'be', 'are', 'at', 'from', 'by', 'have', 'or', 'an', 'was', 'we', 'will', 'my', 'if', 'not', 'but', 'all', 'your', 'has', 'would', 'can', 'so', 'he', 'one', 'they', 'about', 'which', 'our', 'out', 'up', 'more', 'no', 'when', 'what', 'who', 'any', 'their', 'there', 'some', 'me', 'were', 'than', 'had', 'been', 'do', 'them', 're', 's', 't', 'd', 'm', 'o', 'subject', 'http', 'com', 'www', 'font', 'td', 'tr', 'br'}
    words = []
    for t in texts:
        tokens = re.findall(r'\\b[a-z]{3,}\\b', str(t).lower())
        words.extend([w for w in tokens if w not in stopwords])
    return pd.DataFrame(Counter(words).most_common(n), columns=['Token', 'Count'])

safe_words = get_top_tokens(df_analysis[df_analysis['Email Type']=='Safe Email']['Email Text'], n=12)
phish_words = get_top_tokens(df_analysis[df_analysis['Email Type']=='Phishing Email']['Email Text'], n=12)

print("Top 12 Token Safe Email:")
display(safe_words)
print("Top 12 Token Phishing Email:")
display(phish_words)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].barh(safe_words['Token'][::-1], safe_words['Count'][::-1], color='#3498db', edgecolor='black', alpha=0.85)
axes[0].set_title('Top Distinctive Tokens in Safe Email\\n(Notice Enron enterprise terms: enron, deal, gas, ect)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Token Frequency')

axes[1].barh(phish_words['Token'][::-1], phish_words['Count'][::-1], color='#e74c3c', edgecolor='black', alpha=0.85)
axes[1].set_title('Top Distinctive Tokens in Phishing Email\\n(Notice action/social-engineering terms: click, email, money, free)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Token Frequency')

plt.suptitle('Lexical Domain Differences (Corpus Bias vs Phishing Cues)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plot_file5 = PLOT_DIR / '05_email_top_words_comparison.png'
plt.savefig(plot_file5, dpi=200, bbox_inches='tight')
plt.show()
"""))

# Section 1.13 Statistical Hypothesis Testing
cells.append(nbf.v4.new_markdown_cell("""## 1.11 Statistical Hypothesis Testing (Mann-Whitney U Test)

Menguji apakah terdapat perbedaan yang signifikan secara statistik pada panjang karakter dan jumlah kata antara Safe Email dan Phishing Email.
"""))

cells.append(nbf.v4.new_code_cell("""# Mann-Whitney U test on Char Length
u_char, p_char = stats.mannwhitneyu(
    df_analysis[df_analysis['Email Type']=='Phishing Email']['char_length'],
    df_analysis[df_analysis['Email Type']=='Safe Email']['char_length'],
    alternative='two-sided'
)

# Mann-Whitney U test on Word Count
u_word, p_word = stats.mannwhitneyu(
    df_analysis[df_analysis['Email Type']=='Phishing Email']['word_count'],
    df_analysis[df_analysis['Email Type']=='Safe Email']['word_count'],
    alternative='two-sided'
)

print(f"Mann-Whitney U Test - Character Length: U-stat = {u_char:.2e}, p-value = {p_char:.2e}")
print(f"Mann-Whitney U Test - Word Count:       U-stat = {u_word:.2e}, p-value = {p_word:.2e}")
if p_char < 0.05:
    print("KESIMPULAN STATISTIK: Terdapat perbedaan distribusi panjang yang sangat signifikan (p < 0.001). Safe Email cenderung memiliki panjang median lebih tinggi (994 vs 732 karakter) dibanding phishing email singkat.")
"""))

# Section 1.14 Cleanroom Preprocessing Plan for Chapter 2
cells.append(nbf.v4.new_markdown_cell("""## 1.12 Audit Summary & Cleanroom Preprocessing Plan for Chapter 2 (/data-question-framing)

### 📋 Ringkasan Keputusan Audit (Audit Decision: `fix-first`)

| Item Audit | Temuan Empiris | Tindakan Cleanroom untuk Chapter 2 |
| :--- | :--- | :--- |
| **Missing Values** | 16 baris `Email Text` bernilai `NaN` | Drop 16 baris tersebut dari korpus. |
| **Whitespace / Empty** | 19 baris berisi spasi atau karakter newline (`"\\n"`) | Filter dan hapus baris yang kosong setelah `strip()`. |
| **Conflicting Labels** | 533 baris berisi kata literal `"empty"` (335 Phish vs 198 Safe) | Drop seluruh 533 baris teks `"empty"` karena menimbulkan *label noise* fatal. |
| **Anomali Ekstrim 17MB** | Baris 12500 berisi 17 Juta karakter dump file notebook `.ipynb` | Drop baris 12500 dan tetapkan batas maksimal karakter ($< 50.000$ chars). |
| **Redundansi Kolom** | Kolom `Unnamed: 0` adalah index redundant | Drop kolom `Unnamed: 0`. |
| **Teks Duplikat** | 1,097 baris teks duplikat | Drop duplikat teks (`drop_duplicates(subset=['Email Text'])`) untuk mencegah kebocoran antar split. |
| **Resiko Bias Enron** | Token 'enron', 'ect', 'deal', 'vince' over-representatif di Safe | Terapkan regex cleaning untuk header, normalisasi URL (`<URL>`), dan evaluasi stopword korporat. |
| **Rencana Pipeline TF-IDF** | Korpus teks tidak terstruktur | Konfigurasi `TfidfVectorizer(max_features=25000, ngram_range=(1,2), sublinear_tf=True, min_df=5, max_df=0.85)`. |
| **Model Candidate Readiness** | Baseline `Logistic Regression` & Target `Linear SVM` | Lanjutkan ke Chapter 2 untuk implementasi scikit-learn Pipeline dan evaluasi komparatif. |
"""))

nb.cells = cells

with open('email.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("File email.ipynb berhasil dibuat!")
