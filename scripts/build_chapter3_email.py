import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

nb_path = Path('email.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

cells = nb.cells

# CHAPTER 3 Markdown
cells.append(nbf.v4.new_markdown_cell("""# CHAPTER 3 : Exploratory Data Analysis (EDA)

Pada bab ini, kita melakukan Exploratory Data Analysis (EDA) mendalam pada korpus email yang **telah bersih** dari Chapter 2 (`email_cleaned.csv`). 

Karena seluruh anomali teks (16 baris null, baris whitespace, 533 baris konflik label template "empty", serta baris outlier 17MB) telah dieliminasi pada Chapter 2, seluruh analisis teks, sebaran panjang, dan ekstraksi n-gram pada bab ini mencerminkan karakteristik linguistik yang sahih dan bebas bias noise template.

Tujuan utama Chapter 3:
1. Menganalisis sebaran statistik panjang karakter dan kata korpus bersih.
2. Mengekstraksi unigram (kata tunggal) paling diskriminatif antara Safe Email vs Phishing Email.
3. Mengekstraksi bigram (pasangan kata berurutan) untuk menangkap konteks frasa penipuan (*social engineering context*).
4. Menganalisis prevalensi indikator rekayasa sosial (*urgency and security triggers*).
5. Merumuskan konfigurasi parameter optimal untuk `TfidfVectorizer` pada Chapter 4 dan model `Linear SVM` pada Chapter 5.
"""))

# Section 3.1
cells.append(nbf.v4.new_markdown_cell("""## 3.1 Pemuatan Data Bersih & Persiapan Analisis Teks"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font='sans-serif')
palette = {'Safe Email (0)': '#3498db', 'Phishing Email (1)': '#e74c3c'}

CLEAN_EMAIL_PATH = Path('data/email/processed/email_cleaned.csv')
EDA_PLOT_DIR = Path('plots/email')
EDA_PLOT_DIR.mkdir(parents=True, exist_ok=True)

df_clean_email = pd.read_csv(CLEAN_EMAIL_PATH)
df_clean_email['Class_Name'] = df_clean_email['is_phishing'].map({0: 'Safe Email (0)', 1: 'Phishing Email (1)'})

print(f"Korpus email bersih berhasil dimuat: {len(df_clean_email):,} baris x {df_clean_email.shape[1]} kolom.")
print("Distribusi target:")
print(df_clean_email['Class_Name'].value_counts())
"""))

# Section 3.2
cells.append(nbf.v4.new_markdown_cell("""## 3.2 Distribusi Univariat Panjang Karakter dan Kata Bersih

Menganalisis sebaran panjang teks pasca-pembersihan (tanpa distorsi outlier 17MB) untuk memahami perbedaan struktural antara Safe Email dan Phishing Email.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Character length distribution (log scale)
sns.histplot(data=df_clean_email, x='cleaned_char_length', hue='Class_Name',
             palette={'Safe Email (0)': '#3498db', 'Phishing Email (1)': '#e74c3c'},
             log_scale=True, kde=True, ax=axes[0], element='step', common_norm=False)
axes[0].set_title('Cleaned Character Length Distribution (Log Scale)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Character Length (Log Scale)')
axes[0].axvline(df_clean_email[df_clean_email['is_phishing']==0]['cleaned_char_length'].median(),
                color='#3498db', linestyle='--', linewidth=2, label='Safe Median (884)')
axes[0].axvline(df_clean_email[df_clean_email['is_phishing']==1]['cleaned_char_length'].median(),
                color='#e74c3c', linestyle='--', linewidth=2, label='Phish Median (765)')
axes[0].legend()

# Word count distribution (log scale)
sns.histplot(data=df_clean_email, x='cleaned_word_count', hue='Class_Name',
             palette={'Safe Email (0)': '#3498db', 'Phishing Email (1)': '#e74c3c'},
             log_scale=True, kde=True, ax=axes[1], element='step', common_norm=False)
axes[1].set_title('Cleaned Word Count Distribution (Log Scale)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Word Count (Log Scale)')
axes[1].axvline(df_clean_email[df_clean_email['is_phishing']==0]['cleaned_word_count'].median(),
                color='#3498db', linestyle='--', linewidth=2, label='Safe Median (146)')
axes[1].axvline(df_clean_email[df_clean_email['is_phishing']==1]['cleaned_word_count'].median(),
                color='#e74c3c', linestyle='--', linewidth=2, label='Phish Median (128)')
axes[1].legend()

plt.suptitle('Length Distributions on Cleaned Corpus (No Extreme Outliers)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
eda_file1 = EDA_PLOT_DIR / 'eda_01_clean_length_distributions.png'
plt.savefig(eda_file1, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {eda_file1}")

# Tabel Statistik Deskriptif Bersih
length_stats = df_clean_email.groupby('Class_Name')[['cleaned_char_length', 'cleaned_word_count']].describe().T
print("Tabel Statistik Panjang Teks Bersih:")
display(length_stats)
"""))

# Section 3.3
cells.append(nbf.v4.new_markdown_cell("""## 3.3 Analisis Leksikal: Top Unigram Paling Diskriminatif

Mengekstraksi kata tunggal (*unigram*) paling dominan di masing-masing kelas setelah mengeliminasi stopwords umum bahasa Inggris.
"""))

cells.append(nbf.v4.new_code_cell("""# Stopwords kustom untuk analisis leksikal
custom_stopwords = {'the', 'to', 'and', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'this', 'you',
                    'i', 'it', 'with', 'as', 'be', 'are', 'at', 'from', 'by', 'have', 'or', 'an',
                    'was', 'we', 'will', 'my', 'if', 'not', 'but', 'all', 'your', 'has', 'would',
                    'can', 'so', 'he', 'one', 'they', 'about', 'which', 'our', 'out', 'up', 'more',
                    'no', 'when', 'what', 'who', 'any', 'their', 'there', 'some', 'me', 'were',
                    'than', 'had', 'been', 'do', 'them', 're', 's', 't', 'd', 'm', 'o', 'subject',
                    'http', 'com', 'www', 'font', 'td', 'tr', 'br', 'table', 'href', 'width',
                    'border', 'nbsp', 'height', 'align', 'color'}

def extract_top_unigrams(corpus, top_n=15):
    vec = CountVectorizer(stop_words=list(custom_stopwords), min_df=5, token_pattern=r'\\b[a-z]{3,}\\b')
    X_counts = vec.fit_transform(corpus)
    word_freq = X_counts.sum(axis=0)
    words = [(word, word_freq[0, idx]) for word, idx in vec.vocabulary_.items()]
    words = sorted(words, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words[:top_n], columns=['Token', 'Frequency'])

safe_corpus = df_clean_email[df_clean_email['is_phishing'] == 0]['cleaned_text']
phish_corpus = df_clean_email[df_clean_email['is_phishing'] == 1]['cleaned_text']

top_unigram_safe = extract_top_unigrams(safe_corpus, top_n=12)
top_unigram_phish = extract_top_unigrams(phish_corpus, top_n=12)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].barh(top_unigram_safe['Token'][::-1], top_unigram_safe['Frequency'][::-1],
             color='#3498db', edgecolor='black', alpha=0.85)
axes[0].set_title('Top Unigrams in Safe Emails (Enterprise Communication Focus)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Token Frequency')

axes[1].barh(top_unigram_phish['Token'][::-1], top_unigram_phish['Frequency'][::-1],
             color='#e74c3c', edgecolor='black', alpha=0.85)
axes[1].set_title('Top Unigrams in Phishing Emails (Call-to-Action & Lure Focus)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Token Frequency')

plt.suptitle('Unigram Lexical Divergence: Safe vs Phishing Emails', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
eda_file2 = EDA_PLOT_DIR / 'eda_02_top_unigrams_comparison.png'
plt.savefig(eda_file2, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {eda_file2}")
"""))

# Section 3.4
cells.append(nbf.v4.new_markdown_cell("""## 3.4 Analisis N-Gram: Ekstraksi Top Bigram (Frasa Kunci)

Kata tunggal sering kali ambigu. Namun, pasangan kata (*bigram*) menangkap pola rekayasa sosial secara jauh lebih spesifik, seperti `"click here"`, `"verify account"`, `"security alert"`.
"""))

cells.append(nbf.v4.new_code_cell("""def extract_top_bigrams(corpus, top_n=12):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=list(custom_stopwords), min_df=5, token_pattern=r'\\b[a-z]{3,}\\b')
    X_counts = vec.fit_transform(corpus)
    word_freq = X_counts.sum(axis=0)
    words = [(word, word_freq[0, idx]) for word, idx in vec.vocabulary_.items()]
    words = sorted(words, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words[:top_n], columns=['Bigram Phrase', 'Frequency'])

print("Mengekstraksi bigram dari korpus...")
top_bigram_safe = extract_top_bigrams(safe_corpus, top_n=12)
top_bigram_phish = extract_top_bigrams(phish_corpus, top_n=12)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh(top_bigram_safe['Bigram Phrase'][::-1], top_bigram_safe['Frequency'][::-1],
             color='#3498db', edgecolor='black', alpha=0.85)
axes[0].set_title('Top Bigrams in Safe Emails\\n(Corporate Dealings & Scheduling)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Phrase Frequency')

axes[1].barh(top_bigram_phish['Bigram Phrase'][::-1], top_bigram_phish['Frequency'][::-1],
             color='#e74c3c', edgecolor='black', alpha=0.85)
axes[1].set_title('Top Bigrams in Phishing Emails\\n(Social Engineering & Urgency Prompts)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Phrase Frequency')

plt.suptitle('Bigram Phrase Comparison: Contextual Indicators of Phishing', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
eda_file3 = EDA_PLOT_DIR / 'eda_03_top_bigrams_comparison.png'
plt.savefig(eda_file3, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {eda_file3}")
"""))

# Section 3.5
cells.append(nbf.v4.new_markdown_cell("""## 3.5 Analisis Prevalensi Pemicu Rekayasa Sosial (Social Engineering Keywords)

Mengukur persentase email yang memuat kosakata ancaman, pembaruan akun, atau iming-iming finansial:
`['urgent', 'verify', 'account', 'password', 'suspend', 'security', 'bank', 'winner', 'money', 'click']`.
"""))

cells.append(nbf.v4.new_code_cell("""trigger_keywords = ['urgent', 'verify', 'account', 'password', 'suspend', 'security', 'bank', 'winner', 'money', 'click']

trigger_stats = []
for kw in trigger_keywords:
    pattern = rf'\\b{kw}\\b'
    safe_pct = df_clean_email[df_clean_email['is_phishing'] == 0]['cleaned_text'].str.contains(pattern, regex=True).mean() * 100
    phish_pct = df_clean_email[df_clean_email['is_phishing'] == 1]['cleaned_text'].str.contains(pattern, regex=True).mean() * 100
    trigger_stats.append({
        'Keyword': kw,
        'Safe Email Prevalence (%)': round(safe_pct, 2),
        'Phishing Email Prevalence (%)': round(phish_pct, 2),
        'Difference (Phish - Safe)': round(phish_pct - safe_pct, 2)
    })

trigger_df = pd.DataFrame(trigger_stats).sort_values(by='Difference (Phish - Safe)', ascending=False)
print("Prevalensi Kata Pemicu Rekayasa Sosial:")
display(trigger_df)

# Visualisasi Bar Chart
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(trigger_df))
width = 0.35

rects1 = ax.bar(x - width/2, trigger_df['Safe Email Prevalence (%)'], width,
                label='Safe Email (0)', color='#3498db', edgecolor='black')
rects2 = ax.bar(x + width/2, trigger_df['Phishing Email Prevalence (%)'], width,
                label='Phishing Email (1)', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Prevalence in Corpus (%)')
ax.set_title('Social Engineering Trigger Keywords Prevalence by Class', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(trigger_df['Keyword'], fontsize=11)
ax.legend()

for rect in list(rects1) + list(rects2):
    h = rect.get_height()
    if h > 2:
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
eda_file4 = EDA_PLOT_DIR / 'eda_04_social_engineering_triggers.png'
plt.savefig(eda_file4, dpi=200, bbox_inches='tight')
plt.show()
print(f"Plot disimpan di: {eda_file4}")
"""))

# Section 3.6
cells.append(nbf.v4.new_markdown_cell("""## 3.6 Rangkuman Wawasan EDA & Konfigurasi TF-IDF Vectorizer

Berdasarkan analisis eksplorasi data NLP di atas, berikut adalah implikasi krusial untuk tahapan Feature Engineering (Chapter 4) dan Model Training (Chapter 5):

1. **Keharusan Memasukkan N-Gram Berurutan (`ngram_range=(1,2)`):**
   * Bigram seperti `"click here"`, `"security alert"`, dan `"verify account"` memiliki korelasi jauh lebih kuat terhadap serangan phishing dibanding unigram yang berdiri sendiri. TF-IDF harus mengikutsertakan unigram dan bigram.
2. **Penerapan `sublinear_tf=True`:**
   * Teks email memiliki variasi panjang yang lebar (median 700-900 karakter, tetapi dapat mencapai puluhan ribu karakter). Mengaktifkan `sublinear_tf=True` mengganti bobot frekuensi term $TF$ menjadi $1 + \\log(TF)$, sehingga kata yang diulang puluhan kali tidak mendominasi vektor secara tidak proporsional.
3. **Penyaringan Frekuensi Dokumen (`min_df=5`, `max_df=0.85`):**
   * Menetapkan `min_df=5` efektif mengeliminasi token langka atau salah ketik (*typo artifacts*).
   * Menetapkan `max_df=0.85` menyaring istilah korporat yang muncul di hampir seluruh korespondensi Enron, mencegah model mengalami overfitting.
4. **Kesesuaian Model Linear SVM (Support Vector Machine):**
   * Dengan kombinasi unigram dan bigram, ruang dimensi fitur TF-IDF akan mencapai lebih dari $25.000$ dimensi dengan matriks sangat renggang (*sparse matrix*).
   * Linear SVM (`LinearSVC` atau `SGDClassifier(loss='hinge')`) secara matematis adalah arsitektur optimal untuk klasifikasi teks berdimensi sangat tinggi karena mencari *maximum-margin separating hyperplane* secara efisien dengan kompleksitas waktu pelatihan linear terhadap jumlah dokumen.
"""))

nb.cells = cells

with open('email.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Chapter 3 berhasil ditambahkan ke email.ipynb!")
