"""
PhisGuard Bilingual Email Model Retraining & Calibration Pipeline v2.0
Mengimplementasikan prinsip skill:
- baseline-ablation-lab (Naive Bayes, Logistic Regression, Linear SVM + Calibrated CV)
- evaluation-reality-check (Subgroup Slice Testing ID vs EN, OOD Stress Testing)
- scikit-learn (Sublinear TF-IDF, Platt Scaling Calibrated Classifier)
"""

import sys
import os
import re
import json
import time
import shutil
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "email" / "processed" / "bilingual_email_dataset_v2.csv"
MODEL_DIR = BASE_DIR / "models" / "email"
APP_MODEL_DIR = Path(r"C:\CODING\PhisLess\models\email")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' url_token ', text)
    text = re.sub(r'\S+@\S+', ' email_token ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("================================================================================")
    print("      PHISGUARD BILINGUAL MODEL RETRAINING & MULTI-METRIC EVALUATION v2.0       ")
    print("================================================================================")

    # 1. Muat Dataset Dwibahasa Diperkaya
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset dwibahasa tidak ditemukan di: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\n[1] Dataset Dimuat: {len(df):,} baris")
    print(f"    - Sampel Bahasa Inggris  : {sum(df['language']=='en'):,}")
    print(f"    - Sampel Bahasa Indonesia: {sum(df['language']=='id'):,}")
    print(f"    - Distribusi Target: Legitimate={sum(df['is_phishing']==0):,} | Phishing={sum(df['is_phishing']==1):,}")

    # 2. Pembagian Data Latih dan Evaluasi (Stratified 80/20)
    print("\n[2] Membagi Dataset (Stratified Train 80% / Test 20%, random_state=42)...")
    X_train_raw, X_test_raw, y_train, y_test, lang_train, lang_test = train_test_split(
        df['cleaned_text'].fillna(''),
        df['is_phishing'].astype(int),
        df['language'],
        test_size=0.20,
        stratify=df['is_phishing'],
        random_state=42
    )

    print(f"    - Data Latih : {len(X_train_raw):,} sampel")
    print(f"    - Data Uji   : {len(X_test_raw):,} sampel (EN={sum(lang_test=='en')}, ID={sum(lang_test=='id')})")

    # 3. Fitting Vektorizer Dwibahasa
    print("\n[3] Fitting TfidfVectorizer Dwibahasa (ngram_range=(1,2), sublinear_tf=True, max_features=40000)...")
    start_vec = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.90,
        max_features=40000
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    vec_time = time.time() - start_vec

    vocab_size = len(vectorizer.vocabulary_)
    sparsity = 100 * (1 - X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]))
    print(f"    - Ukuran Kosakata     : {vocab_size:,} fitur unigram & bigram dwibahasa")
    print(f"    - Sparsitas Matriks   : {sparsity:.2f}%")
    print(f"    - Waktu Vektorisasi   : {vec_time:.2f} detik")

    # 4. Pelatihan Baseline & Model Produksi (Baseline Ablation Lab)
    print("\n[4] Melatih Model: Baseline Naive Bayes, Baseline Logistic Regression, & Calibrated Linear SVM...")
    
    # Baseline 1: Naive Bayes
    t0 = time.time()
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_vec, y_train)
    t_nb = time.time() - t0

    # Baseline 2: Logistic Regression
    t0 = time.time()
    lr_model = LogisticRegression(C=1.0, max_iter=1500, random_state=42)
    lr_model.fit(X_train_vec, y_train)
    t_lr = time.time() - t0

    # Model Produksi: Calibrated Linear SVM
    t0 = time.time()
    base_svc = LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=3000)
    svm_calibrated = CalibratedClassifierCV(estimator=base_svc, method='sigmoid', cv=5)
    svm_calibrated.fit(X_train_vec, y_train)
    t_svm = time.time() - t0

    print(f"    - MultinomialNB          : Selesai ({t_nb:.2f}s)")
    print(f"    - LogisticRegression     : Selesai ({t_lr:.2f}s)")
    print(f"    - Calibrated Linear SVM  : Selesai ({t_svm:.2f}s)")

    # 5. Evaluasi Komparatif (Baseline vs Produksi)
    print("\n[5] Evaluasi Metrik Kuantitatif pada Test Set Holdout (3,589 sampel):")
    models = {
        "Multinomial Naive Bayes": nb_model,
        "Logistic Regression": lr_model,
        "Calibrated Linear SVM (Prod)": svm_calibrated
    }

    eval_results = []
    for name, m in models.items():
        y_pred = m.predict(X_test_vec)
        y_prob = m.predict_proba(X_test_vec)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        eval_results.append({
            "Model": name,
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
            "ROC-AUC": f"{auc:.4f}"
        })

    df_eval = pd.DataFrame(eval_results)
    print(df_eval.to_string(index=False))

    # 6. Analisis Irisan Subkelompok (Subgroup Slice Testing: ID vs EN) - Evaluation Reality Check
    print("\n[6] Subgroup Slice Analysis (Disparitas Bahasa Inggris vs Indonesia):")
    mask_en = (lang_test == 'en').values
    mask_id = (lang_test == 'id').values

    for lang_name, mask in [("Bahasa Inggris (EN)", mask_en), ("Bahasa Indonesia (ID)", mask_id)]:
        if sum(mask) == 0:
            continue
        y_sub_true = y_test.values[mask]
        X_sub_vec = X_test_vec[mask]
        
        y_sub_pred = svm_calibrated.predict(X_sub_vec)
        y_sub_prob = svm_calibrated.predict_proba(X_sub_vec)[:, 1]

        sub_acc = accuracy_score(y_sub_true, y_sub_pred)
        sub_prec = precision_score(y_sub_true, y_sub_pred, zero_division=0)
        sub_rec = recall_score(y_sub_true, y_sub_pred, zero_division=0)
        sub_f1 = f1_score(y_sub_true, y_sub_pred, zero_division=0)
        
        cm = confusion_matrix(y_sub_true, y_sub_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0)

        print(f"\n  ▶ Slice: {lang_name} ({sum(mask):,} sampel)")
        print(f"    - Akurasi  : {sub_acc*100:.2f}%")
        print(f"    - Presisi  : {sub_prec*100:.2f}%")
        print(f"    - Recall   : {sub_rec*100:.2f}%")
        print(f"    - F1-Score : {sub_f1*100:.2f}%")
        print(f"    - Matriks  : TN={tn}, FP={fp}, FN={fn}, TP={tp} (False Positive Rate: {fp/(tn+fp+1e-9)*100:.2f}%)")

    # 7. Uji Stres Kasus Nyata di Luar Distribusi (Hard Out-Of-Distribution Benchmarks)
    print("\n[7] Uji Stres Out-Of-Distribution (Kasus Kritis Nyata yang Diuji User):")
    ood_cases = [
        # Kasus Kritis 1: Email Google Play ToS Update (Kasus awal False Positive)
        (
            "Pada 17 Desember 2026, kami akan melakukan beberapa perubahan pada Persyaratan Layanan Google Play kami. "
            "Perubahan ini tidak akan memengaruhi cara Anda menggunakan Google Play, tetapi perubahan ini dapat membantu Anda agar lebih mudah memahami apa yang dapat diharapkan dari Google Play. "
            "Anda dapat meninjau persyaratan baru di sini. Persyaratan yang berlaku: Kami memperbarui definisi Konten dalam persyaratan untuk memperjelas bahwa Konten mencakup layanan sistem. "
            "Biaya Pihak Ketiga: Anda bertanggung jawab atas biaya apa pun yang dikenakan oleh pihak ketiga kepada Anda. "
            "Dengan terus menggunakan Google Play setelah perubahan ini, Anda menyetujui persyaratan baru tersebut. Terima kasih telah menggunakan Google Play.",
            0,
            "Google Play ToS Update (ID)"
        ),
        # Kasus Kritis 2: Surat Pengisian KRS Sintesys Unsil
        (
            "Pengumuman Akademik: Seluruh mahasiswa Universitas Siliwangi diwajibkan melakukan bimbingan akademik dan pengisian KRS semester genap secara mandiri melalui portal resmi sintesys.unsil.ac.id sebelum penutupan sistem.",
            0,
            "Akademik Unsil KRS (ID)"
        ),
        # Kasus Kritis 3: Slip Gaji Elektronik Kantor
        (
            "Yth Bapak/Ibu staf, slip gaji elektronik periode bulan Oktober telah tersedia di intranet korporat perusahaan. Silakan login ke sistem HRIS untuk mengunduh dokumen laporan gaji bulanan.",
            0,
            "Slip Gaji HRIS Kantor (ID)"
        ),
        # Kasus Kritis 4: Phishing BCA Mobile
        (
            "Peringatan Resmi Bank BCA: Akun m-BCA Anda telah diblokir sementara karena aktivitas mencurigakan. Segera verifikasi nomor rekening dan PIN Anda melalui link berikut dalam waktu 1x24 jam untuk membuka blokir rekening tabungan.",
            1,
            "Phishing BCA Mobile (ID)"
        ),
        # Kasus Kritis 5: Penipuan Resi Kurir SiCepat APK
        (
            "Paket SiCepat Express gagal dikirimkan kurir karena alamat penerima tidak jelas. Unduh file resi konfirmasi alamat paket apk untuk jadwal pengantaran ulang kurir segera.",
            1,
            "Kurir SiCepat APK (ID)"
        ),
        # Kasus Kritis 6: Tilang Elektronik ETLE Palsu
        (
            "Surat Tilang Elektronik ETLE: Kendaraan Anda tertangkap kamera melakukan pelanggaran batas kecepatan. Unduh surat bukti tilang format apk untuk melunasi denda sebelum STNK diblokir.",
            1,
            "ETLE Tilang APK (ID)"
        ),
        # Kasus Kritis 7: Enron / Work Email (EN Safe)
        (
            "Hi Team, please find attached the weekly progress report for the product rollout. Let me know if you have any questions before our standup tomorrow morning.",
            0,
            "Corporate Work Progress (EN)"
        ),
        # Kasus Kritis 8: Microsoft 365 Phishing (EN Phish)
        (
            "URGENT: Your Microsoft 365 account password will expire in 2 hours! Click the link below immediately to keep your current password or your mailbox will be permanently suspended.",
            1,
            "Microsoft 365 Expiry (EN)"
        )
    ]

    all_ood_passed = True
    for text, true_lbl, desc in ood_cases:
        cln = clean_text(text)
        vec = vectorizer.transform([cln])
        prob = svm_calibrated.predict_proba(vec)[0, 1]
        pred = 1 if prob >= 0.50 else 0
        is_ok = (pred == true_lbl)
        if not is_ok:
            all_ood_passed = False
        status_icon = "✅ LULUS" if is_ok else "❌ GAGAL"
        print(f"  {status_icon} | {desc:28s} | Benar={true_lbl} | Pred={pred} (Prob Phish={prob:.4f}) | Fitur Cocok: {vec.nnz} kata")

    # 8. Simpan Model Baru ke Folder Training & Sinkronkan ke PhisLess
    print("\n[8] Menyimpan Artefak Model Produksi Baru...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    APP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    vec_path_train = MODEL_DIR / "bilingual_tfidf_vectorizer_email.joblib"
    svm_path_train = MODEL_DIR / "bilingual_svm_email.joblib"
    
    vec_path_app = APP_MODEL_DIR / "bilingual_tfidf_vectorizer_email.joblib"
    svm_path_app = APP_MODEL_DIR / "bilingual_svm_email.joblib"

    joblib.dump(vectorizer, vec_path_train)
    joblib.dump(svm_calibrated, svm_path_train)
    print(f"    - Disimpan di Training Repo : {vec_path_train.name}, {svm_path_train.name}")

    # Salin ke PhisLess
    shutil.copy2(vec_path_train, vec_path_app)
    shutil.copy2(svm_path_train, svm_path_app)
    print(f"    - Disinkronkan ke PhisLess  : {vec_path_app.name}, {svm_path_app.name}")

    print("\n================================================================================")
    if all_ood_passed:
        print(">>> SEMUA EVALUASI & STRESS TEST SELESAI DENGAN STATUS LULUS 100%! <<<")
    else:
        print(">>> PELATIHAN SELESAI DENGAN CATATAN PADA SEBAGIAN KASUS UJI <<<")
    print("================================================================================")

if __name__ == "__main__":
    main()
