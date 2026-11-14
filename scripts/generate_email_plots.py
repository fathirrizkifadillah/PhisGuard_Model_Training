import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter

output_dir = Path('plots/email')
output_dir.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', font='sans-serif')
palette = {'Phishing Email': '#e74c3c', 'Safe Email': '#3498db'}

df = pd.read_csv('data/email/raw/phishing_email.csv')
clean_df = df.dropna(subset=['Email Text']).copy()

# 1. Class Distribution
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
counts = clean_df['Email Type'].value_counts()
ax[0].bar(counts.index, counts.values, color=['#3498db', '#e74c3c'], edgecolor='black', alpha=0.85)
for i, v in enumerate(counts.values):
    pct = v / len(clean_df) * 100
    ax[0].text(i, v + 250, f'{v:,}\n({pct:.1f}%)', ha='center', fontweight='bold', fontsize=11)
ax[0].set_title('Email Class Distribution (Sample Count)', fontsize=13, fontweight='bold', pad=12)
ax[0].set_ylabel('Total Samples')
ax[0].set_ylim(0, max(counts.values) * 1.15)

ax[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'],
          startangle=140, explode=(0.05, 0), wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
          textprops={'fontsize': 11, 'fontweight': 'bold'})
ax[1].set_title('Email Class Proportion (Imbalance Ratio: ~1.55:1)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(output_dir / '01_email_class_distribution.png', dpi=200, bbox_inches='tight')
plt.close()

# 2. Text Length and Word Count Distributions (Exclude extreme 17MB outlier row 12500 and 0-word whitespace rows)
clean_df['char_length'] = clean_df['Email Text'].str.len()
clean_df['word_count'] = clean_df['Email Text'].apply(lambda x: len(x.split()))
plot_subset = clean_df[(clean_df['char_length'] < 100000) & (clean_df['word_count'] > 0)].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.histplot(data=plot_subset, x='char_length', hue='Email Type', palette=palette, log_scale=True, kde=False, ax=axes[0], element='step', common_norm=False)
axes[0].set_title('Character Length Distribution (Log Scale)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Character Length (Log Scale)')
axes[0].axvline(clean_df[clean_df['Email Type']=='Safe Email']['char_length'].median(), color='#3498db', linestyle='--', linewidth=2, label='Safe Median (994)')
axes[0].axvline(clean_df[clean_df['Email Type']=='Phishing Email']['char_length'].median(), color='#e74c3c', linestyle='--', linewidth=2, label='Phish Median (732)')
axes[0].legend()

sns.histplot(data=plot_subset, x='word_count', hue='Email Type', palette=palette, log_scale=True, kde=False, ax=axes[1], element='step', common_norm=False)
axes[1].set_title('Word Count Distribution (Log Scale)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Word Count (Log Scale)')
axes[1].axvline(clean_df[clean_df['Email Type']=='Safe Email']['word_count'].median(), color='#3498db', linestyle='--', linewidth=2, label='Safe Median (179)')
axes[1].axvline(clean_df[clean_df['Email Type']=='Phishing Email']['word_count'].median(), color='#e74c3c', linestyle='--', linewidth=2, label='Phish Median (136)')
axes[1].legend()

plt.suptitle('Text Length & Word Count Distributions (Safe vs Phishing)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '02_email_length_distributions.png', dpi=200, bbox_inches='tight')
plt.close()

# 3. Boxplots showing percentiles (Capped at 99th percentile for visual clarity)
p99_char = np.percentile(clean_df['char_length'], 99)
p99_word = np.percentile(clean_df['word_count'], 99)
cap_df = clean_df[(clean_df['char_length'] <= p99_char) & (clean_df['word_count'] > 0)].copy()

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
plt.savefig(output_dir / '03_email_outliers_boxplots.png', dpi=200, bbox_inches='tight')
plt.close()

# 4. Structural indicators
clean_df['has_html'] = clean_df['Email Text'].apply(lambda x: bool(re.search(r'<[^>]+>', str(x))))
clean_df['has_url'] = clean_df['Email Text'].apply(lambda x: bool(re.search(r'http[s]?://|www\.', str(x))))
clean_df['is_empty_text'] = clean_df['Email Text'].str.strip().str.lower().eq('empty')

struct_summary = pd.DataFrame({
    'Metric': ['Contains HTML (%)', 'Contains URLs (%)', 'Has Exact "empty" (%)'],
    'Safe Email': [
        clean_df[clean_df['Email Type']=='Safe Email']['has_html'].mean() * 100,
        clean_df[clean_df['Email Type']=='Safe Email']['has_url'].mean() * 100,
        clean_df[clean_df['Email Type']=='Safe Email']['is_empty_text'].mean() * 100
    ],
    'Phishing Email': [
        clean_df[clean_df['Email Type']=='Phishing Email']['has_html'].mean() * 100,
        clean_df[clean_df['Email Type']=='Phishing Email']['has_url'].mean() * 100,
        clean_df[clean_df['Email Type']=='Phishing Email']['is_empty_text'].mean() * 100
    ]
})

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
plt.savefig(output_dir / '04_email_structural_indicators.png', dpi=200, bbox_inches='tight')
plt.close()

# 5. Top Words / Distinctive Tokens Comparison
def get_top_words(texts, n=12):
    stopwords = {'the', 'to', 'and', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'this', 'you', 'i', 'it', 'with', 'as', 'be', 'are', 'at', 'from', 'by', 'have', 'or', 'an', 'was', 'we', 'will', 'my', 'if', 'not', 'but', 'all', 'your', 'has', 'would', 'can', 'so', 'he', 'one', 'they', 'about', 'which', 'our', 'out', 'up', 'more', 'no', 'when', 'what', 'who', 'any', 'their', 'there', 'some', 'me', 'were', 'than', 'had', 'been', 'do', 'them', 're', 's', 't', 'd', 'm', 'o', 'subject', 'http', 'com', 'www', 'font', 'td', 'tr', 'br'}
    words = []
    for t in texts:
        tokens = re.findall(r'\b[a-z]{3,}\b', str(t).lower())
        words.extend([w for w in tokens if w not in stopwords])
    return pd.DataFrame(Counter(words).most_common(n), columns=['word', 'count'])

safe_words = get_top_words(clean_df[clean_df['Email Type']=='Safe Email']['Email Text'], n=12)
phish_words = get_top_words(clean_df[clean_df['Email Type']=='Phishing Email']['Email Text'], n=12)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].barh(safe_words['word'][::-1], safe_words['count'][::-1], color='#3498db', edgecolor='black', alpha=0.85)
axes[0].set_title('Top Distinctive Tokens in Safe Email\n(Notice Enron enterprise terms: enron, deal, gas, ect)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Token Frequency')

axes[1].barh(phish_words['word'][::-1], phish_words['count'][::-1], color='#e74c3c', edgecolor='black', alpha=0.85)
axes[1].set_title('Top Distinctive Tokens in Phishing Email\n(Notice urgent/call-to-action terms: click, email, money, free)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Token Frequency')

plt.suptitle('Lexical Domain Differences (Corpus Bias vs Phishing Cues)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '05_email_top_words_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

print('Email plots generated successfully!')
