# PhisGuard Engineering Lessons & Distilled Rules

Dokumen ini mendokumentasikan prinsip-prinsip desain, kegagalan umum (*failure modes*), dan aturan empiris yang terbukti valid selama pengembangan sistem deteksi ancaman PhisGuard.

---

### Aturan 1: Bias Artefak Protokol (The HTTPS Fallacy)
- **Observasi**: Dataset akademis sering kali membekukan kondisi web pada tahun pengumpulan data, di mana situs resmi menggunakan HTTPS dan phishing menggunakan HTTP.
- **Dampak**: Model pembelajaran mesin dapat mengambil "jalan pintas" (*shortcut learning*) dengan hanya memeriksa flag HTTPS, sehingga gagal mendeteksi serangan phishing modern yang menggunakan sertifikat SSL/TLS gratis (Let's Encrypt / Cloudflare).
- **Aturan**: Fitur struktural independen (entropi URL, rasio digit numerik, kedalaman path, dan rasio token) harus diperkuat dan diaudit secara terpisah (*slice testing*) untuk memastikan deteksi phishing tetap andal meskipun protokolnya HTTPS.

### Aturan 2: Dimensi Tinggi & Sparsitas pada Klasifikasi Teks
- **Observasi**: Klasifikasi teks berbasis TF-IDF menghasilkan matriks fitur sangat renggang (*sparse matrix* > 99.8% nol) dengan puluhan ribu n-gram unigram dan bigram.
- **Dampak**: Pohon keputusan atau ansambel boosting sering kali lambat dan kurang optimal pada ruang berdimensi tinggi dibandingkan model linier.
- **Aturan**: Linear SVM (didukung kalibrasi probabilitas Platt Scaling sigmoid) memberikan rasio presisi-kecepatan terbaik untuk throughput tinggi pada Email Security Gateway (SEG), dengan waktu inferensi sub-milidetik per email.

### Aturan 3: Kalibrasi Threshold Adaptif (Toleransi False Positive)
- **Observasi**: Penggunaan threshold default 0.50 tidak mencerminkan kebutuhan operasional nyata di mana biaya False Positive (email sah dari eksekutif/klien terblokir) jauh lebih mahal daripada menunda pengiriman email mencurigakan untuk inspeksi tambahan.
- **Aturan**: Terapkan arsitektur pertahanan berlapis dengan tri-state decisioning:
  * Probabilitas > 0.85: Pemblokiran keras (*Hard Block / Quarantine*).
  * Probabilitas 0.45 - 0.85: Peringatan keamanan (*Inject Warning Banner*) dan ekstraksi URL untuk inspeksi sekunder.
  * Probabilitas < 0.45: Lolos ke kotak masuk (*Pass to Inbox*).
