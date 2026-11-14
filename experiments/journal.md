# Machine Learning Experiment Journal: PhisGuard

Protokol pencatatan eksperimen (*experiment journal*) mengikuti kaidah ilmiah `ml-experiment` untuk melacak hipotesis, perubahan konfigurasi, metrik baseline, hasil aktual, dan pembelajaran sistemik.

---

### 2026-11-14 21:15 — EXP-001: In-Distribution Baseline vs Production Modeling

**Status**: COMPLETED

**Hypothesis**:
1. URL: Penerapan extreme gradient boosting (XGBoost) dengan regularisasi `colsample_bytree=0.8` dan histogram split akan mengungguli Random Forest dalam waktu latih dan stabilitas 5-fold cross-validation pada dataset tabular leksikal LegitPhish.
2. Email: Ruang fitur teks berdimensi tinggi (sparse matrix 25.000 n-gram) sangat cocok untuk Linear Support Vector Machine (LinearSVC). Kalibrasi probabilitas menggunakan sigmoid/Platt scaling memungkinkan output probabilitas kontinu tanpa mengorbankan ketajaman hyperplane margin pemisah.

**Change**:
- Inisiasi pemodelan dari data mentah yang telah diaudit (Chapter 1) dan dibersihkan (Chapter 2).
- URL: Random Forest (100 trees, max_depth=12) vs XGBoost (150 trees, max_depth=6, lr=0.08).
- Email: Logistic Regression (L2, C=1.0) vs Calibrated LinearSVC (C=1.0, 3-fold sigmoid calibration).

**Config**:
- URL feature count: 17 fitur (termasuk komposit `path_ratio`)
- Email feature count: 25.000 n-gram (unigram + bigram, sublinear TF)
- Split: 80% train, 20% test stratified

**Baseline**:
- URL Target Metric: Accuracy > 98%, Phishing Recall > 99%
- Email Target Metric: Accuracy > 95%, Phishing F1-Score > 95%

**Outcome**:
- URL Random Forest: Test Acc 99.96%, Phishing F1 0.9997, Fit Time 2.37s
- URL XGBoost: Test Acc 99.95%, Phishing F1 0.9996, Fit Time 4.95s, CV Stability superior
- Email Logistic Regression: Test Acc 98.40%, Phishing F1 0.9789, Fit Time 0.38s
- Email Linear SVM: Test Acc 98.97%, Phishing F1 0.9863, Fit Time 0.67s, False Positive berkurang 57%

---

### 2026-11-14 21:20 — EXP-002: Out-of-Distribution (OOD) & In-The-Wild Threat Stress-Testing

**Status**: COMPLETED

**Hypothesis**:
1. **URL Domain Shift (The HTTPS Trap)**: Model XGBoost dilatih pada dataset LegitPhish yang memiliki bias artefak kuat di mana 100% sampel sah menggunakan HTTPS dan 94.5% sampel phishing menggunakan HTTP polos. Di dunia nyata modern (OpenPhish feed 2026), mayoritas serangan phishing (>70%) telah mengadopsi sertifikat SSL/HTTPS gratis. Diperkirakan akurasi in-sample (99.95%) akan mengalami penurunan degradasi pada URL phishing HTTPS OpenPhish, namun kombinasi fitur non-linear (`percentage_numeric_chars`, `url_entropy`, `subdomain_count`) akan mempertahankan recall di atas 80%.
2. **Email Corporate Ham vs Phishing Lures**: Vektorizer TF-IDF yang dilatih pada korpus Hugging Face akan diuji terhadap korpus independen Enron (email korporat nyata) dan 14 template serangan spear-phishing nyata (CEO Fraud/BEC, Microsoft 365 credential harvesting, DHL delivery scare). Diperkirakan model Linear SVM akan mempertahankan False Positive Rate rendah (< 3%) pada email korporat Enron karena tidak adanya token pemicu rekayasa sosial, namun variasi spear-phishing bertarget dengan kata-kata sopan membutuhkan penyesuaian threshold mitigasi.

**Change**:
- Menjalankan inferensi murni tanpa refitting pada data eksternal yang belum pernah dilihat:
  * URL Phishing: OpenPhish Live Community Feed (300 URL aktif)
  * URL Legitimate: Cisco Umbrella Top 1M (500 domain terverifikasi)
  * URL Edge Cases: 20 kasus terkurasi (Typosquatting, IP-based, Cloud storage abuse, Deep legitimate paths)
  * Email Benchmark: Enron Email Corpus (600 email: ham & spam)
  * Email Curated Lures: 14 kasus nyata (BEC, M365, Crypto theft, DevOps alerts)

**Config**:
- Models: `xgb_production_url.joblib`, `svm_production_email.joblib`, `tfidf_vectorizer_email.joblib`
- URL Features: 17 fitur diekstraksi via `url_feature_extractor.py`
- Text Cleaning: regex cleaner modular Chapter 2

**Expected Outcome**:
- Mengkuantifikasi kesenjangan generalisasi (*generalization gap*) antara performa laboratorium in-sample vs dunia nyata.
- Menemukan irisan kegagalan (*failure slices*) spesifik untuk panduan mitigasi gateway keamanan.

**Outcome**:
- **Email Linear SVM OOD (Enron Corpus)**: Akurasi **99.00%**, Precision Safe/Ham 99.30%, Recall Phish/Spam 99.36%, F1-Score 0.9904. Model sangat tangguh dan minim False Positive pada email kantor.
- **Email Curated Real-World Lures**: Akurasi **85.71%** (12/14). Sukses mendeteksi CEO Fraud/BEC (prob=0.814), M365 credential harvesting (prob=0.982), DHL scam (prob=0.928), dan Binance crypto theft (prob=0.970). Gagal pada teks pendek netral (DocuSign, Zoom).
- **URL XGBoost OOD (OpenPhish + Cisco Umbrella)**: Recall deteksi phishing OpenPhish mencapai **93.00%** (279 dari 300 terdeteksi).
- **Subgroup Slice Disparity Terbukti**: Phishing HTTP terdeteksi 100.00% (138/138), sedangkan phishing HTTPS terdeteksi 87.04% (141/162). Terdapat gap disparitas sebesar **12.96%** yang membuktikan hipotesis bias protokol HTTPS.
- **URL Curated Edge Cases**: Akurasi **70.00%** (14/20). Sukses mendeteksi typosquatting (`paypa1`), IP-based attacks, cloud bucket abuse (`storage.googleapis.com`), dan reverse subdomain spoof. Terjadi False Positive pada URL sah berparameter query panjang (Google search, AWS S3 signed asset, ArXiv paper).
- **Artefak Dihasilkan**: `external_testing.ipynb`, folder data `data/external/`, plot visualisasi di `plots/external_eval/`.
