# PhisGuard: Dual-Engine Threat Detection System

PhisGuard adalah sistem deteksi ancaman siber berbasis machine learning yang mengintegrasikan dua mesin pendeteksi:
1. **URL Threat Detection**: Menggunakan algoritma XGBoost Classifier yang dilatih pada 17 fitur leksikal dan struktural, dibandingkan terhadap baseline Random Forest.
2. **Email Phishing Classification**: Menggunakan algoritma Linear Support Vector Machine (LinearSVC) dengan kalibrasi probabilitas Platt Scaling (sigmoid) dan vektorisasi TF-IDF unigram + bigram (25.000 n-gram), dibandingkan terhadap baseline Logistic Regression.

Repository ini mencakup siklus lengkap pengembangan machine learning: audit kualitas data, pembersihan korpus, analisis data eksploratif (EDA), rekayasa fitur, pelatihan model, evaluasi diagnostik, analisis kesalahan prediksi (*error analysis*), serta validasi eksternal (*out-of-distribution stress-testing*).

> **Integrasi Proyek Utama**: Seluruh model yang dilatih dan dievaluasi di repository ini (`PhisGuad_training`) dikembangkan secara khusus untuk digunakan sebagai mesin inferensi inti pada proyek aplikasi utama: [https://github.com/fathirrizkifadillah/PhisGuard](https://github.com/fathirrizkifadillah/PhisGuard).

---

## 1. Rangkuman Kinerja Model

### A. Evaluasi Internal (In-Distribution 80:20 Split)

| Modul | Model | Accuracy | Precision | Recall | F1-Score | Waktu Latih |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **URL** | Random Forest (Baseline) | 99.96% | 0.9994 | 1.0000 | 0.9995 | 2.37s |
| **URL** | **XGBoost (Produksi)** | **99.95%** | **0.9994** | **0.9998** | **0.9996** | 4.95s |
| **Email** | Logistic Regression (Baseline) | 98.40% | 0.9686 | 0.9893 | 0.9789 | 0.38s |
| **Email** | **Linear SVM (Produksi)** | **98.97%** | **0.9863** | **0.9863** | **0.9863** | 0.67s |

### B. Validasi Eksternal & Stress-Test (Out-of-Distribution / OOD)

Pengujian OOD dilakukan tanpa melatih ulang model untuk menguji ketahanan terhadap data dunia nyata:
- **Email pada Korpus Enron (600 email kantor unseen)**: Akurasi mencapai **99.00%** dengan False Positive hanya 1.39% (4 dari 288 email kantor).
- **Email pada Serangan Nyata (14 skenario kurasi)**: Akurasi **85.71%**. Sukses mendeteksi CEO Fraud / BEC (prob=0.814), Microsoft 365 Password Expiry (prob=0.982), DHL Delivery Scam (prob=0.928), dan Binance Crypto Alert (prob=0.970).
- **URL pada OpenPhish Live Feed (300 URL aktif)**: Recall deteksi phishing aktif mencapai **93.00%** (279 dari 300 terdeteksi).
  - Phishing HTTP biasa: Recall **100.00%** (138 dari 138).
  - Phishing HTTPS modern: Recall **87.04%** (141 dari 162). Terdapat gap disparitas sebesar **12.96%** yang memvalidasi hipotesis bias protokol SSL pada dataset akademis.

---

## 2. Struktur Repository

```text
PhisGuad_training/
├── data/
│   ├── email/
│   │   ├── raw/
│   │   │   └── phishing_email.csv            <- Dataset asal Hugging Face
│   │   └── processed/
│   │       └── email_cleaned.csv             <- Dataset bersih hasil pembersihan Ch 2
│   ├── url/
│   │   ├── url_features_extracted1.csv       <- Dataset asal Mendeley LegitPhish
│   │   └── processed/
│   │       └── url_features_cleaned.csv      <- Dataset bersih hasil sanitasi Ch 2
│   └── external/
│       ├── email/
│       │   ├── enron_external_benchmark.csv  <- Korpus Enron (Hugging Face)
│       │   └── curated_real_world_email_lures.csv <- Kasus serangan nyata (BEC, M365, DHL)
│       └── url/
│           ├── openphish_live_phishing.csv   <- Feed langsung OpenPhish
│           ├── umbrella_top_legitimate.csv   <- Domain teratas Cisco Umbrella 1M
│           └── curated_url_edge_cases.csv    <- Kasus batas (typosquatting, IP, cloud)
├── experiments/
│   ├── journal.md                            <- Log eksperimen terstruktur (EXP-001 & EXP-002)
│   └── lessons.md                            <- Kaidah teknis & pelajaran empiris
├── models/
│   ├── email/
│   │   ├── lr_baseline_email.joblib          <- Model baseline Logistic Regression
│   │   ├── svm_production_email.joblib       <- Model produksi Calibrated Linear SVM (Inggris)
│   │   ├── tfidf_vectorizer_email.joblib     <- Objek vektorizer TF-IDF (25.000 n-gram)
│   │   ├── bilingual_svm_email.joblib        <- Model produksi Calibrated Linear SVM Dwibahasa (ID + EN)
│   │   └── bilingual_tfidf_vectorizer_email.joblib <- Vektorizer TF-IDF Dwibahasa (35.000 n-gram)
│   └── url/
│       ├── rf_baseline_url.joblib            <- Model baseline Random Forest
│       └── xgb_production_url.joblib         <- Model produksi XGBoost Classifier
├── notebook/
│   ├── email.ipynb                           <- Notebook end-to-end modul Email (Ch 1 - 6 Inggris & Ch 7 Adaptasi Dwibahasa Indonesia)
│   ├── external_testing.ipynb                <- Notebook stress-test OOD & threat modeling (Ch 1 - 7)
│   └── url.ipynb                             <- Notebook end-to-end modul URL (Ch 1 - 6)
├── plots/
│   ├── email/                                <- 15 visualisasi audit, EDA, dan evaluasi email
│   ├── url/                                  <- 15 visualisasi audit, EDA, dan evaluasi URL
│   └── external_eval/                        <- Visualisasi evaluasi OOD & slice disparity
├── .gitignore
├── requirements.txt                          <- Daftar dependensi pustaka Python
└── README.md
```

---

## 3. Sumber Dataset & Sitasi

1. **URL Legitimate vs Phishing (LegitPhish Dataset)**:
   - Sumber: Mendeley Data
   - Tautan: https://data.mendeley.com/datasets/hx4m73v2sf/2
   - Volume: 100.871 baris setelah sanitasi (57.1% Phishing, 42.9% Legitimate).
2. **Email Phishing Dataset**:
   - Sumber: Hugging Face (`zefang-liu/phishing-email-dataset`)
   - Tautan: https://huggingface.co/datasets/zefang-liu/phishing-email-dataset
   - Volume: 17.510 baris setelah pembersihan (37.4% Phishing, 62.6% Safe).
3. **OpenPhish Community Feed**:
   - Sumber: OpenPhish (`https://openphish.com/feed.txt`)
   - Kegunaan: Validasi independen URL phishing aktif bertranspor HTTPS.
4. **Cisco Umbrella Top 1M**:
   - Sumber: Cisco Umbrella Public Static S3
   - Kegunaan: Validasi domain resmi global terverifikasi.
5. **Enron Spam/Ham Corpus**:
   - Sumber: Hugging Face (`SetFit/enron_spam`)
   - Kegunaan: Uji False Positive pada email lalu lintas kantor nyata.

---

## 4. Panduan Eksekusi Notebook

Seluruh notebook tersimpan di folder `notebook/` dan dirancang independen (*self-contained*) dengan path relatif terhadap aset data:
- **`notebook/url.ipynb`**:
  * Chapter 1 : Dataset Quality Audit
  * Chapter 2 : Data Cleaning & Preprocessing
  * Chapter 3 : Exploratory Data Analysis (EDA)
  * Chapter 4 : Feature Pipeline & Engineering (`path_ratio` composite)
  * Chapter 5 : Model Training & Baseline Comparison (Random Forest vs XGBoost)
  * Chapter 6 : Model Evaluation, Diagnostic & Error Analysis
- **`notebook/email.ipynb`**:
  * Chapter 1 : Dataset Quality Audit
  * Chapter 2 : Data Cleaning & Text Preprocessing
  * Chapter 3 : Exploratory Data Analysis & Stylometrics
  * Chapter 4 : Text Vectorization & NLP Feature Pipeline (TF-IDF N-grams)
  * Chapter 5 : Model Training & Baseline Comparison (Logistic Regression vs Calibrated Linear SVM)
  * Chapter 6 : Model Evaluation, Diagnostic & Error Analysis
  * Chapter 7 : Ekstensi Model Dwibahasa (Bilingual Indonesian-English Adaptation & OOV Mitigation)
- **`notebook/external_testing.ipynb`**:
  * Chapter 1 : Threat Modeling & OOD Evaluation Protocol
  * Chapter 2 : Ingestion of External Datasets
  * Chapter 3 : Feature Extraction & Pipeline
  * Chapter 4 : URL Threat Detection Stress-Test
  * Chapter 5 : Email Classification Stress-Test
  * Chapter 6 : Edge Cases & Failure Slice Dissection
  * Chapter 7 : Evaluation Reality Check & Production Hardening

---

## 5. Cara Instalasi & Menjalankan

1. **Clone repository**:
   ```bash
   git clone <URL_REPOSITORY>
   cd PhisGuad_training
   ```

2. **Buat dan aktifkan virtual environment**:
   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Buka dan jalankan Jupyter Notebook**:
   ```bash
   jupyter notebook notebook/
   ```

---

## 6. Rekomendasi Deployment Produksi

Berdasarkan framework `evaluation-reality-check`, model direkomendasikan untuk implementasi bertahap (*Proceed-with-Risk*) pada [PhisGuard](https://github.com/fathirrizkifadillah/PhisGuard) dengan mitigasi berikut:
1. **Pre-filter Domain Allowlist**: Terapkan pengecekan awal terhadap 10.000 domain teratas (Tranco/Umbrella) sebelum evaluasi leksikal URL untuk mencegah False Positive pada domain resmi berparameter panjang (misal: AWS S3 signed link, GitHub repo).
2. **Tri-State Decisioning**:
   * Skor > 0.85: Pemblokiran keras otomatis (*Hard Block / Quarantine*).
   * Skor 0.45 - 0.85: Sisipkan *Security Warning Banner* dan kirim tautan ke sandbox inspeksi sekunder.
   * Skor < 0.45: Diteruskan ke kotak masuk (*Pass*).
3. **Korelasi Ansambel Email + URL**: Email dengan kata-kata netral yang lolos dari klasifikasi teks (seperti notifikasi tanda tangan dokumen atau undangan rapat) wajib diekstraksi tautan URL-nya untuk diuji menggunakan mesin XGBoost URL.
