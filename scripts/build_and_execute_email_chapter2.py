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
cells.append(nbf.v4.new_markdown_cell("""# PhisGuard: Email Threat Classification Engine
Notebook Eksperimen dan Pipeline Klasifikasi Email Phishing

---

### Informasi Proyek dan Roadmap Model
* Domain: Natural Language Processing (NLP) / Email Threat Detection
* Target Model (Production): TF-IDF + Linear SVM (LinearSVC / SGDClassifier loss hinge)
* Baseline Model: TF-IDF + Logistic Regression
* Nama Dataset: Phishing Email Dataset
* Sumber Repository: Hugging Face ([zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset))
* Asal Usul Korpus: Mirror dataset Kaggle "Phishing Email Detection" oleh Cyber Cop (Enron Corpus, Nazario Corpus, SpamAssassin, Nigerian Fraud 419, Ling-Spam, CEAS)

---
### Struktur Notebook
1. CHAPTER 1 : Dataset Quality Audit
2. CHAPTER 2 : Data Cleaning & Text Preprocessing
3. CHAPTER 3 : Model Training & Evaluation (Tahap Selanjutnya)
"""))

# CHAPTER 1
cells.append(nbf.v4.new_markdown_cell("""# CHAPTER 1 : Dataset Quality Audit

Bab ini berfokus pada audit komprehensif korpus teks Phishing Email Dataset:
1. Integritas korpus teks mentah, missing values, dan entri whitespace kosong.
2. Identifikasi baris anomali raksasa (dump notebook baris 12500 sebesar 17 MB).
3. Deteksi konflik label pada template teks literal "empty" (335 Phish vs 198 Safe).
4. Evaluasi distribusi target dan ketidakseimbangan kelas.
5. Analisis statistik tendensi sentral panjang karakter dan jumlah kata (Mean vs Median gap).
6. Prevalensi tag HTML, URL, dan artefak struktural.
7. Analisis risiko bias domain Enron vs Nazario (Clever Hans Effect).
"""))

# Section 1.1 Decision Canvas
cells.append(nbf.v4.new_markdown_cell("""## 1.1 Problem Framing & NLP Decision Canvas (/data-question-framing)

* Unit of Analysis: Satu pesan email teks mentah.
* Target Definition: Kolom Email Type (Safe Email vs Phishing Email).
* Biaya Kesalahan: False Positive (email bisnis sah dikarantina) sangat merugikan operasional, sehingga Precision pada Safe Email harus dijaga sangat tinggi tanpa mengorbankan Recall deteksi Phishing.
* Justifikasi Model: Fitur TF-IDF berdimensi tinggi (> 20.000 token n-gram) optimal dipisahkan oleh Linear SVM (maximum margin hyperplane) dengan kecepatan inferensi tinggi untuk mail gateway.
"""))

# Section 1.2 Ingestion
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

sns.set_theme(style='whitegrid', font='sans-serif')
palette = {'Phishing Email': '#e74c3c', 'Safe Email': '#3498db'}

DATA_PATH = Path('data/email/raw/phishing_email.csv')
PLOT_DIR = Path('plots/email')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv(DATA_PATH)
print("Dataset Phishing Email mentah berhasil dimuat.")
print(f"Dimensi Data: {df_raw.shape[0]:,} baris, {df_raw.shape[1]} kolom.")
df_raw.head(3)
"""))

# Section 1.3 Schema
cells.append(nbf.v4.new_markdown_cell("""## 1.3 Schema & Structural Column Inspection"""))

cells.append(nbf.v4.new_code_cell("""schema_df = pd.DataFrame({
    'Column': df_raw.columns,
    'Non-Null Count': df_raw.notnull().sum().values,
    'Null Count': df_raw.isnull().sum().values,
    'Dtype': df_raw.dtypes.values
})
print("Ringkasan Skema Kolom:")
display(schema_df)

mem_mb = df_raw.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"Ukuran Memori: {mem_mb:.2f} MB")
print("Catatan: Kolom 'Unnamed: 0' merupakan index redundan yang akan dieliminasi.")
"""))

# Section 1.4 Missing & Blank
cells.append(nbf.v4.new_markdown_cell("""## 1.4 Missing Values, Nulls & Blank Email Audit"""))

cells.append(nbf.v4.new_code_cell("""null_count = df_raw['Email Text'].isnull().sum()
print(f"Jumlah baris dengan Email Text NaN: {null_count} baris ({null_count/len(df_raw)*100:.3f}%)")

non_null_text = df_raw.dropna(subset=['Email Text']).copy()
empty_strip = non_null_text['Email Text'].str.strip().eq('')
whitespace_count = empty_strip.sum()
print(f"Jumlah baris dengan Email Text kosong atau hanya whitespace: {whitespace_count} baris")

newline_samples = non_null_text[empty_strip]
display(newline_samples[['Unnamed: 0', 'Email Text', 'Email Type']].head())
"""))

# Section 1.5 Duplicate & Conflicting Labels
cells.append(nbf.v4.new_markdown_cell("""## 1.5 Deduplication, Template Redundancy & Conflicting Label Trap

Temuan Konflik Label:
Ditemukan teks template identik yang muncul berulang kali di kedua kelas sekaligus.
"""))

cells.append(nbf.v4.new_code_cell("""raw_duplicates = non_null_text['Email Text'].duplicated().sum()
print(f"Total Teks Email Duplikat: {raw_duplicates:,} baris")

empty_text_mask = non_null_text['Email Text'].str.strip().str.lower().eq('empty')
empty_text_df = non_null_text[empty_text_mask]
print(f"Jumlah email yang hanya berisi kata 'empty': {len(empty_text_df):,} baris")
print("Distribusi kelas untuk teks 'empty':")
print(empty_text_df['Email Type'].value_counts())

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

print("Distribusi Kelas Email:")
for cname in class_counts.index:
    print(f"- {cname}: {class_counts[cname]:,} sampel ({class_pcts[cname]:.2f}%)")
print(f"Rasio Ketidakseimbangan (Safe : Phishing): {imbalance_ratio:.2f} : 1")

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

# Section 1.7 Length Profiling & Outlier 12500
cells.append(nbf.v4.new_markdown_cell("""## 1.7 NLP Corpus Statistical Profiling & Central Tendency (/statistical-analysis)"""))

cells.append(nbf.v4.new_code_cell("""df_analysis = non_null_text.copy()
df_analysis['char_length'] = df_analysis['Email Text'].str.len()
df_analysis['word_count'] = df_analysis['Email Text'].apply(lambda x: len(str(x).split()))

longest_emails = df_analysis.nlargest(5, 'char_length')[['Unnamed: 0', 'char_length', 'word_count', 'Email Type']]
print("5 Email dengan Karakter Terpanjang:")
display(longest_emails)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Investigasi Anomali Raksasa Baris 12500 (17 Juta Karakter)

Baris 12500 memiliki panjang 17.036.692 karakter. Berikut adalah cuplikan teks awal dan akhir:
"""))

cells.append(nbf.v4.new_code_cell("""row_12500_text = df_analysis.loc[12500, 'Email Text']
print(f"Panjang Baris 12500: {len(row_12500_text):,} karakter")
print("\\nAwal Teks (200 karakter pertama):")
print(row_12500_text[:200])
print("\\nAkhir Teks (200 karakter terakhir):")
print(row_12500_text[-200:])
print("\\nDiagnosa: Baris 12500 secara tidak sengaja memuat dump file Jupyter Notebook (.ipynb) beserta output base64.")
"""))

# Section 1.8 Percentiles
cells.append(nbf.v4.new_markdown_cell("""### Tabel Persentil Panjang Teks (p1 hingga p99)"""))

cells.append(nbf.v4.new_code_cell("""stat_rows = []
for metric in ['char_length', 'word_count']:
    for cls in ['Safe Email', 'Phishing Email', 'Overall']:
        sub = df_analysis if cls == 'Overall' else df_analysis[df_analysis['Email Type'] == cls]
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
print("Ringkasan Persentil dan Tendensi Sentral (Tanpa Outlier 17MB):")
display(stat_summary_df)
"""))

# Section 1.9 Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 1.8 Visualisasi Distribusi Panjang & Outlier Audit"""))

cells.append(nbf.v4.new_code_cell("""plot_subset = df_analysis[(df_analysis['char_length'] < 100000) & (df_analysis['word_count'] > 0)].copy()

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

# Boxplot Capped at p99
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

# Section 1.10 Structural markers
cells.append(nbf.v4.new_markdown_cell("""## 1.9 Structural Markers & HTML / Link Prevalence"""))

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
print("Prevalensi Indikator Struktural:")
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

# Section 1.11 Lexical Diversity & Enron Bias
cells.append(nbf.v4.new_markdown_cell("""## 1.10 Lexical Diversity & Spurious Corpus Bias (Enron vs Nazario)"""))

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

# Section 1.12 Hypothesis Testing
cells.append(nbf.v4.new_markdown_cell("""## 1.11 Statistical Hypothesis Testing (Mann-Whitney U Test)"""))

cells.append(nbf.v4.new_code_cell("""u_char, p_char = stats.mannwhitneyu(
    df_analysis[df_analysis['Email Type']=='Phishing Email']['char_length'],
    df_analysis[df_analysis['Email Type']=='Safe Email']['char_length'],
    alternative='two-sided'
)

u_word, p_word = stats.mannwhitneyu(
    df_analysis[df_analysis['Email Type']=='Phishing Email']['word_count'],
    df_analysis[df_analysis['Email Type']=='Safe Email']['word_count'],
    alternative='two-sided'
)

print(f"Mann-Whitney U Test - Character Length: U-stat = {u_char:.2e}, p-value = {p_char:.2e}")
print(f"Mann-Whitney U Test - Word Count:       U-stat = {u_word:.2e}, p-value = {p_word:.2e}")
"""))

# Section 1.13 Summary
cells.append(nbf.v4.new_markdown_cell("""## 1.12 Audit Summary & Handoff ke Cleanroom Preprocessing

Ringkasan Hasil Audit (Keputusan: fix-first):
* Missing Values: 16 baris Email Text bernilai NaN.
* Whitespace Only: 19 baris berisi whitespace/newline tanpa teks.
* Label Conflicts: 533 baris teks literal "empty" dengan label bertentangan (335 Phish vs 198 Safe).
* Outlier Ekstrim: Baris 12500 memuat 17 Juta karakter dump notebook.
* Duplikasi: 1.097 baris teks duplikat.
* Redundansi Skema: Kolom 'Unnamed: 0' merupakan index bawaan CSV.
"""))

# ==========================================
# CHAPTER 2
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""# CHAPTER 2 : Data Cleaning & Text Preprocessing

Pada bab ini, kita mengeksekusi pipeline *data cleanroom* dan rekayasa teks NLP untuk menghasilkan korpus bersih yang siap diproses oleh TF-IDF Vectorizer:
1. **Eliminasi Missing Value & Whitespace:** Menghapus 16 baris `NaN` dan baris yang hanya berisi whitespace/newline.
2. **Eliminasi Konflik Label Template:** Menghapus 533 baris teks literal `"empty"` yang menimbulkan noise label fatal.
3. **Filtering Outlier Ekstrim:** Menghapus baris 12500 (dump notebook 17 MB) dan membatasi teks maksimal $\\le 50.000$ karakter.
4. **Deduplikasi Korpus:** Menghapus teks duplikat untuk mencegah kebocoran informasi (*data leakage*) antara train dan test set.
5. **Pembersihan Kolom Redundan:** Mengeliminasi kolom index CSV `Unnamed: 0`.
6. **Standardisasi Label Target:** Melakukan encoding label biner menjadi `is_phishing` ($1 = \\text{Phishing Email}$, $0 = \\text{Safe Email}$).
7. **Rekayasa Pembersihan Teks NLP:** Membangun fungsi modular pembersihan teks:
   * Stripping tag HTML (`<...>` $\\rightarrow$ spasi).
   * Normalisasi tautan hyperlink menjadi token khusus `url_token`.
   * Normalisasi alamat email menjadi token khusus `email_token`.
   * Lowercasing dan normalisasi spasi.
8. **Validasi Integritas Sebelum vs Sesudah:** Memverifikasi kualitas korpus akhir.
9. **Penyimpanan Data Bersih:** Mengekspor hasil pembersihan ke `data/email/processed/email_cleaned.csv`.
10. **Pembagian Data Bebas Leakage (Train-Test Split):** Melakukan stratified train-test split ($80:20$) dengan `random_state=42` sesuai panduan scikit-learn.
"""))

# Section 2.1 Pipeline Execution
cells.append(nbf.v4.new_markdown_cell("""## 2.1 Eksekusi Pipeline Data Cleanroom Korpus Email"""))

cells.append(nbf.v4.new_code_cell("""email_audit_tracker = []
email_audit_tracker.append({'Tahap': '1. Dataset Mentah (Raw Ingestion)', 'Jumlah Baris': len(df_raw), 'Baris Dihapus': 0})

# 1. Hapus kolom index redundan 'Unnamed: 0'
df_cleaned_email = df_raw.copy()
if 'Unnamed: 0' in df_cleaned_email.columns:
    df_cleaned_email = df_cleaned_email.drop(columns=['Unnamed: 0'])

# 2. Hapus baris dengan Email Text NaN
n_before = len(df_cleaned_email)
df_cleaned_email = df_cleaned_email.dropna(subset=['Email Text']).copy()
dropped_null = n_before - len(df_cleaned_email)
email_audit_tracker.append({'Tahap': '2. Eliminasi Email Text NaN', 'Jumlah Baris': len(df_cleaned_email), 'Baris Dihapus': dropped_null})

# 3. Hapus baris teks kosong atau whitespace-only
n_before = len(df_cleaned_email)
df_cleaned_email = df_cleaned_email[df_cleaned_email['Email Text'].str.strip() != ''].copy()
dropped_whitespace = n_before - len(df_cleaned_email)
email_audit_tracker.append({'Tahap': '3. Eliminasi Teks Whitespace / Newline', 'Jumlah Baris': len(df_cleaned_email), 'Baris Dihapus': dropped_whitespace})

# 4. Hapus 533 baris teks literal 'empty' yang memicu konflik label
n_before = len(df_cleaned_email)
df_cleaned_email = df_cleaned_email[df_cleaned_email['Email Text'].str.strip().str.lower() != 'empty'].copy()
dropped_empty = n_before - len(df_cleaned_email)
email_audit_tracker.append({'Tahap': '4. Eliminasi Konflik Label (Teks Literal empty)', 'Jumlah Baris': len(df_cleaned_email), 'Baris Dihapus': dropped_empty})

# 5. Filter outlier ekstrim (> 50,000 karakter, termasuk dump notebook 17MB)
df_cleaned_email['char_length'] = df_cleaned_email['Email Text'].str.len()
n_before = len(df_cleaned_email)
df_cleaned_email = df_cleaned_email[df_cleaned_email['char_length'] <= 50000].copy()
dropped_outlier = n_before - len(df_cleaned_email)
email_audit_tracker.append({'Tahap': '5. Eliminasi Outlier Ekstrim (> 50k chars)', 'Jumlah Baris': len(df_cleaned_email), 'Baris Dihapus': dropped_outlier})

# 6. Deduplikasi teks email identik
n_before = len(df_cleaned_email)
df_cleaned_email = df_cleaned_email.drop_duplicates(subset=['Email Text']).copy()
dropped_dedup = n_before - len(df_cleaned_email)
email_audit_tracker.append({'Tahap': '6. Deduplikasi Teks Email', 'Jumlah Baris': len(df_cleaned_email), 'Baris Dihapus': dropped_dedup})

# 7. Standardisasi label target (1 = Phishing Email, 0 = Safe Email)
df_cleaned_email['is_phishing'] = (df_cleaned_email['Email Type'] == 'Phishing Email').astype(int)

# Tampilkan log ringkasan tahapan pembersihan
email_tracker_df = pd.DataFrame(email_audit_tracker)
email_tracker_df['Total Berkurang Kumulatif'] = len(df_raw) - email_tracker_df['Jumlah Baris']
print("Log Eksekusi Cleanroom Data Email:")
display(email_tracker_df)
"""))

# Section 2.2 NLP Text Cleaning Function
cells.append(nbf.v4.new_markdown_cell("""## 2.2 Rekayasa Pembersihan Teks NLP (Text Preprocessing)

Membangun fungsi preprocessing teks modular untuk membersihkan tag markup dan menormalkan entitas leksikal:
* Menghapus tag HTML (`<.*?>`).
* Menormalkan hyperlink menjadi token generik `url_token`.
* Menormalkan alamat email menjadi token generik `email_token`.
* Mengubah ke huruf kecil (*lowercasing*).
* Menghapus karakter spasi dan newline berlebih.
"""))

cells.append(nbf.v4.new_code_cell("""def clean_email_text(text):
    text = str(text)
    # 1. Hapus tag HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Normalisasi URL
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' url_token ', text)
    # 3. Normalisasi Email
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_token ', text)
    # 4. Lowercase
    text = text.lower()
    # 5. Normalisasi spasi dan karakter whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Terapkan fungsi pembersihan ke seluruh korpus
print("Menjalankan pembersihan teks pada seluruh baris...")
df_cleaned_email['cleaned_text'] = df_cleaned_email['Email Text'].apply(clean_email_text)

# Hitung ulang metrik panjang karakter dan kata pasca-pembersihan
df_cleaned_email['cleaned_char_length'] = df_cleaned_email['cleaned_text'].str.len()
df_cleaned_email['cleaned_word_count'] = df_cleaned_email['cleaned_text'].apply(lambda x: len(x.split()))

# Tampilkan 3 sampel teks sebelum vs sesudah
sample_comparison = df_cleaned_email[['Email Text', 'cleaned_text', 'is_phishing']].head(3)
for idx, row in sample_comparison.iterrows():
    label_name = 'Phishing' if row['is_phishing'] == 1 else 'Safe'
    print(f"--- Sampel Baris {idx} ({label_name}) ---")
    print("Sebelum (Raw):", repr(row['Email Text'][:100]))
    print("Sesudah (Cleaned):", repr(row['cleaned_text'][:100]))
"""))

# Section 2.3 Validation
cells.append(nbf.v4.new_markdown_cell("""## 2.3 Validasi Integritas Pasca-Pembersihan

Memverifikasi perbandingan kualitas korpus sebelum vs sesudah data cleanroom.
"""))

cells.append(nbf.v4.new_code_cell("""val_email = {
    'Metrik Kualitas Korpus': [
        'Total Baris Sampel',
        'Jumlah Email Text Missing (NaN)',
        'Jumlah Baris Teks Duplikat',
        'Sampel Teks Literal empty (Label Conflict)',
        'Maksimum Panjang Karakter',
        'Median Panjang Karakter',
        'Sampel Safe Email (Label 0)',
        'Sampel Phishing Email (Label 1)',
        'Proporsi Safe Email (%)',
        'Proporsi Phishing Email (%)'
    ],
    'Sebelum Pembersihan (Raw)': [
        f"{len(df_raw):,}",
        f"{df_raw['Email Text'].isnull().sum():,}",
        f"{df_raw['Email Text'].duplicated().sum():,}",
        f"{(df_raw['Email Text'].str.strip().str.lower() == 'empty').sum():,}",
        f"{df_analysis['char_length'].max():,} chars (17MB dump)",
        f"{df_analysis['char_length'].median():.1f} chars",
        f"{(df_raw['Email Type'] == 'Safe Email').sum():,}",
        f"{(df_raw['Email Type'] == 'Phishing Email').sum():,}",
        f"{(df_raw['Email Type'] == 'Safe Email').mean()*100:.2f}%",
        f"{(df_raw['Email Type'] == 'Phishing Email').mean()*100:.2f}%"
    ],
    'Sesudah Pembersihan (Clean)': [
        f"{len(df_cleaned_email):,}",
        f"{df_cleaned_email['cleaned_text'].isnull().sum():,}",
        f"{df_cleaned_email['cleaned_text'].duplicated().sum():,}",
        f"{(df_cleaned_email['cleaned_text'].str.strip().str.lower() == 'empty').sum():,}",
        f"{df_cleaned_email['cleaned_char_length'].max():,} chars",
        f"{df_cleaned_email['cleaned_char_length'].median():.1f} chars",
        f"{(df_cleaned_email['is_phishing'] == 0).sum():,}",
        f"{(df_cleaned_email['is_phishing'] == 1).sum():,}",
        f"{(df_cleaned_email['is_phishing'] == 0).mean()*100:.2f}%",
        f"{(df_cleaned_email['is_phishing'] == 1).mean()*100:.2f}%"
    ]
}

val_email_df = pd.DataFrame(val_email)
print("Tabel Komparasi Kualitas Korpus Email:")
display(val_email_df)
"""))

# Section 2.4 Saving Cleaned Data
cells.append(nbf.v4.new_markdown_cell("""## 2.4 Penyimpanan Dataset Bersih (Processed Data)"""))

cells.append(nbf.v4.new_code_cell("""OUTPUT_EMAIL_CLEAN_DIR = Path('data/email/processed')
OUTPUT_EMAIL_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_EMAIL_PATH = OUTPUT_EMAIL_CLEAN_DIR / 'email_cleaned.csv'
# Simpan kolom esensial yang siap pakai
df_cleaned_email[['cleaned_text', 'is_phishing', 'cleaned_char_length', 'cleaned_word_count']].to_csv(CLEAN_EMAIL_PATH, index=False)

print(f"Dataset email bersih berhasil disimpan ke: {CLEAN_EMAIL_PATH}")
print(f"Ukuran file: {CLEAN_EMAIL_PATH.stat().st_size / (1024 ** 2):.2f} MB")
print(f"Dimensi akhir: {df_cleaned_email.shape[0]:,} baris")
"""))

# Section 2.5 Train-Test Split Free of Leakage
cells.append(nbf.v4.new_markdown_cell("""## 2.5 Pembagian Data Bebas Kebocoran (Train-Test Split) (/scikit-learn)

Sesuai metodologi `/scikit-learn`:
* Pemisahan korpus prediktor $X$ (`cleaned_text`) dan target $y$ (`is_phishing`).
* Pembagian data menggunakan `train_test_split` dengan rasio $80\\%$ training set dan $20\\%$ testing set.
* Parameter `stratify=y` digunakan agar proporsi kelas $62,58\\% : 37,42\\%$ terjaga konsisten di subset pelatihan dan pengujian.
* Parameter `random_state=42` menjamin eksperimen dapat direplikasi sepenuhnya.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split

X_email = df_cleaned_email['cleaned_text']
y_email = df_cleaned_email['is_phishing']

X_train_email, X_test_email, y_train_email, y_test_email = train_test_split(
    X_email, y_email,
    test_size=0.20,
    stratify=y_email,
    random_state=42
)

email_split_summary = pd.DataFrame({
    'Subset': ['Training Set (80%)', 'Testing Set (20%)', 'Total Korpus'],
    'Total Sampel': [len(X_train_email), len(X_test_email), len(X_email)],
    'Sampel Safe Email (0)': [sum(y_train_email == 0), sum(y_test_email == 0), sum(y_email == 0)],
    'Sampel Phishing Email (1)': [sum(y_train_email == 1), sum(y_test_email == 1), sum(y_email == 1)],
    'Proporsi Phishing (%)': [
        f"{sum(y_train_email == 1)/len(y_train_email)*100:.2f}%",
        f"{sum(y_test_email == 1)/len(y_test_email)*100:.2f}%",
        f"{sum(y_email == 1)/len(y_email)*100:.2f}%"
    ]
})

print("Ringkasan Pembagian Data Korpus Email (Train-Test Split):")
display(email_split_summary)
"""))

# Section 2.6 Handoff to Chapter 3
cells.append(nbf.v4.new_markdown_cell("""## 2.6 Handoff ke Chapter 3 (Model Training & Evaluation)

Korpus teks telah berada dalam kondisi siap vektorisasi (*vectorization-ready*):
* `X_train_email` dan `y_train_email` siap diproses oleh pipeline `TfidfVectorizer` untuk melatih baseline Logistic Regression dan model produksi Linear SVM.
* `X_test_email` dan `y_test_email` diisolasi untuk pengujian akhir guna mencegah kebocoran informasi (*data leakage*).
* Vektorizer TF-IDF akan di-fit HANYA pada data `X_train_email` dan sekadar men-transform `X_test_email` pada tahap pemodelan di Chapter 3.
"""))

nb.cells = cells

with open('email.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("File email.ipynb dengan Chapter 1 dan Chapter 2 berhasil dibuat.")
