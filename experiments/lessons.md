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

### Aturan 4: Disparitas Multi-Part TLD & Sanitasi Ekstraksi Fitur (Kasus ccSLD `.ac.id` / `.go.id`)
- **Observasi**: Korpus pelatihan tabular publik (seperti Mendeley LegitPhish) hampir seluruhnya didominasi oleh domain generik tingkat satu (*single-part TLD* seperti `.com`, `.org`, `.net`). Ketika berhadapan dengan nama domain nasional bertingkat (*country-code Second-Level Domain* / ccSLD seperti `.ac.id`, `.go.id`, `.sch.id`), pemisahan titik naif (`host.split('.')`) secara keliru mengekstrak nama institusi sebagai *subdomain* dan menganggap nama domain hanya berpanjang 2 karakter (misal: `ac`).
- **Dampak**: Model tabular (XGBoost) mengalami bias artefak leksikal dan menandai portal resmi akademik/pemerintah (seperti `sintesys.unsil.ac.id`) sebagai *False Positive* akibat metrik `domain_name_length` yang sangat pendek (2 huruf) dan `subdomain_count` yang terhitung ganda.
- **Aturan**:
  * Tahap pra-pemrosesan ekstraksi fitur inferensi wajib menggunakan dekomposisi berbasis hirarki registrasi domain (*multi-part ccTLD mapping*) agar nama institusi (`unsil`) diekstrak sebagai domain utama yang sah (panjang = 5) dan jumlah subdomain terhitung akurat (`sintesys` = 1).
  * Validasi regulasi legalitas registrar (seperti pengawasan PANDI pada domain institusional `.ac.id` dan `.go.id`) harus diintegrasikan pada lapisan rekonsiliasi pertahanan hibrida (*Defensive AI Architecture*) guna mencegah terjadinya *false positive* pada portal layanan publik/akademik terdaftar.

### Aturan 5: Subdomain Brand Spoofing vs Root Domain Ownership
- **Observasi**: Penyerang siber modern sering kali mendaftarkan domain asing berbiaya murah (misal: `myonlineservicecanada.com`), lalu memasang nama entitas resmi sebagai *subdomain* (misal: `optus.myonlineservicecanada.com`) serta melengkapinya dengan sertifikat SSL/HTTPS gratis.
- **Dampak**: Penilai manusia maupun model bahasa besar (LLM) dapat tertipu oleh keberadaan nama merek di subdomain dan protokol HTTPS jika tidak memisahkan kepemilikan domain utama (*Root Domain*) dari subdomain.
- **Aturan**:
  * Analisis leksikal wajib memisahkan dan membandingkan *subdomain* terhadap *root domain* untuk mendeteksi anomali peniruan entitas (*Brand Impersonation*).
  * Sistem inferensi wajib menyertakan inspeksi konten web langsung (*Live Web Content Inspection*) untuk mendeteksi respons pemblokiran WAF (*Web Application Firewall*, misal: `Suspected Phishing | Cloudflare`) dan indikasi formulir pengumpulan kredensial secara independen.

### Aturan 6: Mitigasi Sparse Vector Collapse & Bias Intercept pada Domain-Shift Bahasa Alami (Dwibahasa)
- **Observasi**: Model Linear SVM yang dilatih dengan 99.7% korpus bahasa Inggris dan hanya 0.3% sampel bahasa Indonesia mengalami keruntuhan vektor (*sparse vector collapse*) ketika memproses teks bahasa Indonesia formal di luar kosakata (*Out-Of-Vocabulary* / OOV).
- **Dampak**: Vektor fitur TF-IDF runtuh menjadi $\vec{x} \approx \vec{0}$. Akibat nilai bias/intercept positif model hasil pelatihan ($b > 0$), fungsi sigmoid Platt Scaling (`CalibratedClassifierCV`) secara keliru memetakan input non-bahasa Inggris ke probabilitas phishing tinggi (~78%), menimbulkan *False Positive* fatal pada email pengumuman resmi seperti Google Play Terms of Service update.
- **Aturan**:
  1. Penyeimbangan representasi korpus dwibahasa (Inggris & Indonesia) dalam matriks fitur TF-IDF dengan ekspansi kosakata formal (kebijakan layanan, pengumuman akademik, komunikasi korporat) dan variasi lure penipuan lokal (perbankan, kurir APK, tilang ETLE).
  2. Penerapan sublinear TF scaling ($1 + \log(TF)$) dan regularisasi parameter SVM terkalibrasi (`CalibratedClassifierCV` dengan stratifikasi 5-fold) untuk memastikan probabilitas netral/legitimate pada teks informatif.
  3. Integrasi ekstraksi tautan mendalam (*Deep Embedded Link Inspection*) untuk mengkorelasikan seluruh hyperlink di dalam email langsung ke model deteksi URL.
