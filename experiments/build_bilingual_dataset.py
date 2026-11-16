"""
PhisGuard Bilingual Dataset Builder v2.0
Mengembangkan korpus email dwibahasa komprehensif (Inggris + Indonesia)
untuk mengatasi Sparse Vector Collapse & Out-Of-Vocabulary (OOV) Bias.
Berdasarkan prinsip skill: messy-data-cleanroom, leakage-adversary.
"""

import re
import json
import hashlib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "email" / "processed"
EN_DATA_PATH = DATA_PROCESSED_DIR / "email_cleaned.csv"
OUTPUT_DATA_PATH = DATA_PROCESSED_DIR / "bilingual_email_dataset_v2.csv"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' url_token ', text)
    text = re.sub(r'\S+@\S+', ' email_token ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
# KORPUS BAHASA INDONESIA RESMI (LEGITIMATE / SAFE - KELAS 0)
# ==============================================================================
INDO_SAFE_SAMPLES = [
    # 1. Pembaruan Persyaratan Layanan & Kebijakan Privasi Resmi (Terms of Service)
    "pada 17 desember 2026 kami akan melakukan beberapa perubahan pada persyaratan layanan google play kami perubahan ini tidak akan memengaruhi cara anda menggunakan google play tetapi perubahan ini dapat membantu anda agar lebih mudah memahami apa yang dapat diharapkan dari google play dan apa yang kami harapkan dari anda saat anda menggunakan layanan kami anda dapat meninjau persyaratan baru di sini secara sekilas berikut arti perubahan ini bagi anda persyaratan yang berlaku kami memperbarui definisi konten dalam persyaratan untuk memperjelas bahwa konten mencakup layanan sistem termasuk tetapi tidak terbatas pada layanan yang disediakan oleh layanan google play biaya pihak ketiga kami menambahkan frasa untuk mengklarifikasi bahwa anda bertanggung jawab atas biaya apa pun yang dikenakan oleh pihak ketiga kepada anda termasuk biaya untuk penginstalan dan pemeliharaan konten dan google play di perangkat anda jika anda menggunakan family link untuk mengelola akun google bagi orang lain harap luangkan waktu untuk memberi tahu orang tersebut tentang perubahan ini dengan terus menggunakan google play setelah perubahan ini anda menyetujui persyaratan baru tersebut jika tidak setuju anda dapat mendownload konten yang sebelumnya anda beli atau instal kemudian menghentikan penggunaan anda atas google play terima kasih telah menggunakan google play",
    "pemberitahuan penting pembaruan perjanjian layanan microsoft kami memperbarui ketentuan penggunaan dan kebijakan privasi akun microsoft agar lebih transparan mengenai hak cipta data kecerdasan buatan serta perlindungan konsumen perubahan berlaku mulai awal bulan depan tidak ada tindakan yang diperlukan jika anda setuju dengan perubahan ini",
    "informasi pembaruan syarat dan ketentuan media apple kami memperbarui syarat dan ketentuan layanan app store apple music dan icloud untuk mematuhi regulasi perlindungan data pribadi terbaru anda dapat membaca rangkuman hak pengguna dan ketentuan pengembalian dana pada portal resmi apple support",
    "pemberitahuan penyesuaian biaya layanan transaksi tokopedia mulai tanggal 1 desember biaya administrasi untuk beberapa kategori transaksi pembayaran digital mengalami penyesuaian untuk terus meningkatkan kualitas infrastruktur sistem dan keamanan platform mitra penjual dapat melihat rincian biaya di pusat edukasi seller",
    "pembaruan kebijakan privasi pengguna gojek kami berkomitmen untuk selalu menjaga kerahasiaan data pengguna kebijakan privasi kami yang diperbarui memberikan rincian lebih jelas tentang bagaimana data perjalanan pembayaran gopay dan preferensi makanan diproses demi kenyamanan anda",
    "informasi penyesuaian syarat penggunaan shopeepay kami memperbarui ketentuan batas saldo harian dan integrasi qris nasional sesuai dengan ketentuan dan regulasi bank indonesia nasabah tidak perlu melakukan tindakan apapun untuk tetap menikmati layanan",
    "pemberitahuan perubahan ketentuan layanan bukalapak sehubungan dengan integrasi logistik baru kami memperbarui syarat pengiriman barang dan asuransi proteksi pembeli silakan pelajari klausul baru pada pusat bantuan bukalapak",

    # 2. Komunikasi Kampus, Akademik, dan Portal Pendidikan Resmi
    "pengumuman jadwal pengisian kartu rencana studi krs semester genap universitas siliwangi seluruh mahasiswa diwajibkan melakukan konsultasi akademik dengan dosen pembimbing dan menyelesaikan pengisian krs secara mandiri melalui portal sintesys sebelum batas waktu penutupan sistem",
    "surat edaran pelaksanaan wisuda sarjana dan magister periode gelombang kedua calon wisudawan diharapkan menyelesaikan verifikasi bebas pustaka menyerahkan berkas skripsi ke perpustakaan pusat dan mencetak bukti kelulusan pada sistem informasi akademik",
    "pendaftaran seminar proposal dan sidang skripsi program studi informatika telah dibuka mahasiswa yang telah memenuhi beban sks minimum dan telah disetujui oleh dosen pembimbing dapat mengunggah draf tugas akhir melalui portal e-skripsi universitas",
    "pemberitahuan seleksi beasiswa prestasi dan kip kuliah kementerian pendidikan bagi mahasiswa berprestasi silakan unduh panduan teknis persyaratan berkas dan format surat pernyataan pada laman resmi kemahasiswaan kampus",
    "jadwal libur perkuliahan dan cuti bersama hari raya idul fitri segenap sivitas akademika universitas diberitahukan bahwa seluruh aktivitas administrasi dan perkuliahan diliburkan mulai tanggal 10 april dan masuk kembali tanggal 18 april",
    "sosialisasi program magang dan studi independen bersertifikat msib kampus merdeka ikuti webinar penjelasan teknis pelaksanaan konversi sks dan tata cara pendaftaran bersama tim kemendikbudristek melalui zoom",
    "pengumuman pelaksanaan ujian akhir semester uas berbasis komputer cbt mahasiswa diwajibkan membawa kartu tanda mahasiswa ktm dan hadir 15 menit sebelum sesi ujian dimulai di laboratorium komputer gedung rektorat",
    "undangan bimbingan teknis penulisan artikel ilmiah bereputasi internasional scopus bagi dosen muda dan asisten ahli fakultas teknik bertempat di auditorium lantai empat ruang multimedia",

    # 3. Komunikasi Korporat, Bisnis, dan Operasional Internal Kantor
    "selamat pagi tim berikut rangkuman hasil rapat evaluasi sprint 45 seluruh backlog prioritas tinggi telah terselesaikan dan siap dilakukan deployment staging mohon qa engineer melakukan pengujian regresi",
    "notifikasi persetujuan formulir pengajuan cuti tahunan manajer telah menyetujui permohonan cuti atas nama sdr ahmad ferdiansyah sisa cuti anda periode tahun berjalan adalah lima hari kerja",
    "diberitahukan kepada seluruh karyawan bahwa slip gaji elektronik periode bulan berjalan telah diunggah ke portal hris perusahaan silakan login menggunakan sso kredensial internal untuk mengunduh pdf slip gaji",
    "agenda rapat koordinasi kuartalan dewan direksi dan kepala divisi operasional meeting akan diselenggarakan pada hari kamis jam sembilan pagi di ruang rapat executive lantai dua puluh",
    "pengumuman pemeliharaan rutin server database internal tim devops akan melakukan maintenance dan upgrade patch keamanan pada hari minggu dini hari antara jam 01.00 sampai 04.00 wib selama proses sistem intranet tidak dapat diakses",
    "sosialisasi program asuransi kesehatan karyawan bni life perwakilan asuransi akan memberikan pemaparan detail mengenai tata cara klaim rawat inap dan fasilitas rawat jalan bagi karyawan beserta keluarga",
    "evaluasi kinerja tahunan performance appraisal seluruh staf diharapkan melengkapi form evaluasi diri pada portal hc sebelum tanggal 25 november untuk penilaian promosi jabatan",
    "pemberitahuan perpanjangan lisensi perangkat lunak figma dan jetbrains seluruh akun lisensi tim product development telah diperpanjang untuk durasi satu tahun ke depan melalui enterprise plan",
    "rekapitulasi kehadiran dan presensi bulanan departemen sumber daya manusia mengingatkan seluruh staf untuk melakukan klarifikasi presensi sebelum periode cut-off penggajian",
    "surat tugas pelaksanaan audit internal cabang surabaya mohon tim auditor mempersiapkan berkas audit kepatuhan iso 27001 tiket transportasi pesawat dan reservasi hotel telah dikonfirmasi oleh bagian general affairs",
    "undangan halal bihalal dan ramah tamah seluruh karyawan pt citra mulia persada bertempat di hall pertemuan gedung utama lantai dasar",
    "pengajuan reimbursement biaya perjalanan dinas luar kota atas nama rudi haryanto telah disetujui oleh bagian keuangan dana akan ditransfer bersamaan dengan pembayaran payroll",
    "mohon tanggapan dan masukan rekan-rekan mengenai draf standard operating procedure sop penanganan insiden keamanan siber sebelum disahkan oleh direktur teknologi informasi",
    "konfirmasi penerimaan purchase order po pengadaan sepuluh unit laptop developer dari vendor resmi pt teknologi solusi nusantara jadwal pengiriman barang hari selasa depan",

    # 4. Utilitas, Tagihan Resmi, & Konfirmasi Transaksi Wajar
    "terima kasih atas pembayaran tagihan listrik pln pascabayar id pelanggan 5321098421 transaksi telah berhasil diproses nomor referensi pelunasan terlampir simpan email ini sebagai bukti pembayaran sah",
    "notifikasi pembayaran tagihan internet indihome telkom indonesia pembayaran paket internet dan layanan tv kabel untuk periode bulan ini telah kami terima terima kasih atas kepercayaan anda menggunakan layanan kami",
    "tiket elektronik pt kereta api indonesia pemesanan tiket kereta lodaya rute tasikmalaya gambir atas nama penumpang fathir maulana telah terkonfirmasi tunjukkan kode booking ini saat boarding",
    "konfirmasi reservasi penerbangan garuda indonesia rute jakarta denpasar bali kode reservasi gaa782 silakan lakukan web check-in melalui aplikasi resmi mulai 24 jam sebelum keberangkatan",
    "notifikasi pengiriman pesanan toko official shopee barang pesanan nomor id 8912384 telah diserahkan ke jasa kurir jne resi pengiriman dapat dilacak secara langsung melalui aplikasi resmi",
    "pemberitahuan pembayaran polis asuransi prudential premi bulanan anda telah berhasil ditarik melalui autodebet rekening tabungan terima kasih telah mempercayakan proteksi jiwa keluarga kepada kami",
    "bukti setor pajak e-billing kementerian keuangan penerimaan negara bukan pajak atas kode billing 029384910 telah berhasil divalidasi oleh bank persepsi ntpn terdaftar resmi di sistem djp",
    "konfirmasi transaksi kartu kredit bca transaksi retail sebesar rp 450.000 di merchant hypermarket telah berhasil jika ini bukan transaksi anda hubungi halo bca pada nomor resmi 1500888",
    "surat konfirmasi perpanjangan domain web dan hosting webmail niagahoster paket unlimited hosting anda telah otomatis diperpanjang untuk masa aktif satu tahun ke depan invoice lunas terlampir",
    "laporan rekening koran bulanan elektronik e-statement bank mandiri ringkasan mutasi rekening tabungan periode 1 sampai 31 oktober telah tersedia dalam lampiran pdf terenkripsi tanggal lahir",
    "informasi suku bunga simpanan dan program deposito berjangka bank rakyat indonesia nikmati suku bunga kompetitif untuk penempatan dana deposito minimum sepuluh juta rupiah di kantor cabang terdekat",
    "notifikasi pesanan tiket nonton cinema xxi m-tix kode pemesanan bioskop anda untuk film di theater mall ciputra telah berhasil diunduh silakan cetak tiket mandiri di kios ticketing bioskop",

    # 5. Pemberitahuan Resmi Tambahan (Mencegah False Positive pada N-gram Umum)
    "halo rekan-rekan selamat berakhir pekan kami informasikan bahwa sistem pendingin udara ac kantor akan dibersihkan hari sabtu jadi ruangan kerja akan ditutup sementara",
    "pemberitahuan pendaftaran donor darah rutin kerja sama palang merah indonesia dan serikat pekerja perusahaan bagi yang berminat dapat mendaftar di meja lobi klinik",
    "pengumuman pemenang lomba desain logo ulang tahun perusahaan selamat kepada sdr diki atas karyanya yang terpilih menjadi pemenang utama hadiah akan diserahkan pada upacara bendera",
    "surat edaran penyesuaian jam kerja selama bulan suci ramadhan jam kerja operasional kantor dimajukan menjadi pukul delapan pagi sampai empat sore untuk seluruh staf dan karyawan",
    "terima kasih telah mengikuti survei kepuasan pelanggan gofood masukan anda sangat berharga untuk meningkatkan mutu layanan restoran mitra dan kurir pengemudi",
    "konfirmasi pendaftaran webinar keamanan informasi nasional e-sertifikat kehadiran akan dikirimkan ke alamat email ini maksimal 3 hari kerja setelah acara selesai",
    "notifikasi pembaruan aplikasi mytelkomsel versi terbaru kini menghadirkan kemudahan pembelian paket data dan penukaran poin langsung di genggaman anda",
    "halo bu ratna berikut laporan hasil uji laboratorium kualitas air bersih sumur bor gedung b seluruh parameter fisika dan kimia memenuhi standar baku mutu dinas kesehatan",
    "ringkasan aktivitas akun github tim engineering rekap commit pull request dan issue yang berhasil diselesaikan pada repositori internal minggu ini",
    "informasi pergantian kartu atm bca debit berchip bagi nasabah yang masih menggunakan kartu debit magnetic stripe segera tukarkan ke kartu chip gratis di customer service bca"
]

# ==============================================================================
# KORPUS BAHASA INDONESIA PHISHING / REKAYASA SOSIAL (MALICIOUS - KELAS 1)
# ==============================================================================
INDO_PHISHING_SAMPLES = [
    # 1. Ancaman Perbankan, Pembekuan Rekening, Pemerasan Finansial
    "peringatan resmi bank bca akun rekening m-bca mobile anda telah diblokir sementara karena aktivitas tidak wajar segera verifikasi nomor kartu atm dan pin anda di link ini dalam waktu 1x24 jam untuk buka blokir",
    "pemberitahuan bank bca pembaruan tarif transfer antar bank mengalami kenaikan tarif normal rp 150.000 per bulan jika anda tidak setuju dengan perubahan tarif ini segera batalkan melalui tautan resmi kami",
    "peringatan keamanan brimo akun internet banking bri anda terindikasi diakses dari perangkat tidak dikenal di rusia klik tautan berikut untuk konfirmasi pembatalan transaksi dan amankan saldo rekening anda segera",
    "bank mandiri kartu debit livin anda dinonaktifkan permanen akibat kesalahan memasukkan pin segera lakukan aktivasi ulang dan konfirmasi kata sandi anda pada tautan portal verifikasi nasabah",
    "notifikasi bank bni transaksi transfer dana sebesar rp 10.000.000 ke rekening luar negeri sedang diproses jika anda merasa tidak melakukan transaksi ini segera batalkan dengan memasukkan pin mbanking anda pada tautan darurat",
    "bank cimb niaga octo mobile rekening anda terindikasi terkena biaya denda administrasi sebesar dua ratus ribu rupiah konfirmasi penolakan biaya denda di link verifikasi nasabah sekarang",
    "bank permata akun mobile banking dibekukan silakan klik tautan verifikasi identitas e-ktp dan masukkan nomor kartu kredit untuk memulihkan akses rekening tabungan anda",
    "peringatan bca sekuritas akun investasi anda akan dinonaktifkan dalam 24 jam kecuali anda mengonfirmasi kepemilikan saham dengan mengisi formulir login dan token keybca pada web verifikasi berikut",
    "tagihan kartu kredit overlimit transaksi sebesar rp 15.000.000 telah disetujui bayar tagihan minimum hari ini untuk menghindari blokir kartu selamanya klik link pembayaran pelunasan kartu kredit",
    "klikbca konfirmasi token keybca anda perlu disinkronkan ulang karena update sistem perbankan masukkan nomor respon appli 1 dan appli 2 pada portal otorisasi berikut agar transaksi tidak terblokir",

    # 2. Modus Penipuan File APK Kurir Ekspedisi & Undangan
    "paket jnt express anda gagal diantar ke alamat karena kurir tidak menemukan nomor rumah mohon unduh file foto resi pengiriman dan alamat kurir format apk di bawah ini untuk pengantaran ulang",
    "sicepat express paket nomor resi 002939103 gagal dikirimkan kurir karena alamat penerima buram segera instal file resi konfirmasi paket apk untuk mengubah jadwal pengantaran barang anda",
    "paket jne tertahan di kantor operasional bea cukai karena kurang biaya administrasi klik tautan pengunduhan dokumen resi pembayaran bea cukai apk sebelum paket disita dan dimusnahkan pihak berwenang",
    "undangan pernikahan digital kami mengundang bapak ibu rekan semua untuk menghadiri resepsi pernikahan putra putri kami mohon doa restu silakan unduh dan install file surat undangan pernikahan apk",
    "surat panggilan tilang elektronik etle polda metro jaya kendaraan anda tertangkap kamera melanggar batas kecepatan unduh bukti foto pelanggaran tilang dan surat tilang format apk untuk bayar denda",
    "surat pemberitahuan surat pemberitahuan tahunan spt pajak bermasalah dari kantor pelayanan pajak pratama segera unduh lampiran formulir e-billing apk dan bayar denda keterlambatan pembayaran pajak",
    "pemberitahuan kpu pencocokan dan penelitian data pemilih dpt pemilu kepala daerah mohon cek status nama anda terdaftar atau tidak dengan membuka file aplikasi konfirmasi dpt format apk",
    "pos indonesia pengiriman paket dokumen penting atas nama anda tertunda di gudang transit segera unduh formulir konfirmasi penerimaan surat berharga format apk",

    # 3. Penipuan Saldo Dompet Digital, Hadiah Undian, dan Pinjaman Online
    "selamat nomor ponsel anda beruntung memenangkan hadiah undian festival belanja shopee rp 50.000.000 klaim hadiah uang tunai anda sekarang juga dengan memasukkan nomor telepon dan kode otp di link ini",
    "klaim saldo dana kaget gratis rp 500.000 dari promo spesial tahun baru klik tautan dompet digital berikut sebelum kuota saldo habis masukkan pin keamanan dompet dana anda",
    "gopay reward promo cashback saldo gopay rp 350.000 telah masuk ke akun anda klik link verifikasi nomor handphone dan masukkan kode sandi gopay anda untuk mencairkan saldo ke rekening bank",
    "ovo loyalty point points anda sebanyak 100.000 akan hangus malam ini segera tukarkan poin dengan voucher pulsa atau saldo ovo cash gratis di tautan resmi penukaran",
    "linkaja gebyar bonus akhir tahun anda terpilih menerima modal usaha rp 5.000.000 tanpa bunga klaim sekarang dengan mendaftarkan nomor rekening dan pin linkaja anda pada halaman web berikut",
    "peringatan shopeepay paylater tagihan anda sebesar rp 2.500.000 telah menunggak lunasi dalam waktu dua belas jam hari ini atau tim lapangan akan mendatangi alamat rumah anda bayar via link kilat",
    "kredivo notifikasi limit pinjaman instan rp 15.000.000 siap dicairkan langsung ke rekening anda tanpa jaminan klik link dan masukkan data ktp serta foto selfie bersama buku tabungan",
    "selamat akun tokopedia anda terpilih mendapatkan kupon belanja gratis ongkir seumur hidup dan diskon sembilan puluh persen klaim voucher belanja sekarang dengan login akun tokopedia anda disini",

    # 4. Ancaman Akun Webmail, Kuota Email, dan Layanan Media Sosial
    "peringatan administrator server webmail kapasitas kuota email kotak masuk anda telah mencapai sembilan puluh delapan persen akun anda akan dihapus permanen dalam 24 jam kecuali anda memvalidasi kuota penyimpanan",
    "security alert microsoft outlook akun email kantor anda akan dinonaktifkan hari ini karena melanggar kebijakan pengiriman pesan spam segera klik tombol batalkan penutupan akun dan masukkan password anda",
    "peringatan keamanan akun instagram akun anda dilaporkan atas pelanggaran hak cipta video oleh pihak ketiga akun anda akan ditangguhkan selamanya dalam 48 jam ajukan banding pada tautan formulir verifikasi hak cipta",
    "whatsapp web alert nomor whatsapp anda sedang didaftarkan pada perangkat android lain di kalimantan jika bukan anda segera batalkan registrasi dan jangan bagikan kode verifikasi enam digit pada siapa pun",
    "google account security alert seseorang mencoba masuk ke akun gmail anda dari browser tidak dikenal di luar negeri klik tautan berikut untuk memulihkan akun dan menyetel ulang kata sandi anda segera",
    "peringatan telkomsel poin poin anda sebanyak dua ribu poin akan kedaluwarsa hari ini tukarkan segera dengan paket kuota internet 50 gb gratis melalui link penukaran poin telkomsel",
    "pemberitahuan pln aliran listrik di rumah anda akan diputus total besok pagi pukul delapan karena tunggakan pembayaran rekening listrik bayar tagihan darurat segera di portal pembayaran ini",
    "bpjs ketenagakerjaan pencairan bantuan dana jaminan hari tua jht rp 3.500.000 telah disetujui segera lengkapi nomor rekening bank dan foto kartu bpjs anda pada link pencairan bantuan",
    "mypertamina subsidi tepat verifikasi qr code bbm subsidi anda dibatalkan oleh dinas perhubungan segera perbarui data plat nomor kendaraan dan stnk di tautan pendaftaran subsidi pertamina",
    "indodax alert penarikan mata uang kripto bitcoin senilai rp 45.000.000 telah berhasil dilakukan jika ini bukan anda segera klik tautan pembatalan penarikan dan ganti kata sandi akun indodax anda"
]

def generate_augmented_variations():
    """Menghasilkan variasi leksikal yang realistis untuk memperkaya kamus n-gram bahasa Indonesia."""
    augmented_safe = []
    augmented_phish = []
    
    # Prefix formal aman
    safe_prefixes = [
        "Pemberitahuan resmi: ", "Informasi penting: ", "Yth. Pengguna Layanan, ", 
        "Selamat pagi, ", "Surat edaran: ", "Pemberitahuan kepada seluruh staf: "
    ]
    # Suffix formal aman
    safe_suffixes = [
        " Terima kasih atas perhatian dan kerja sama yang baik.",
        " Jika ada kendala, hubungi unit layanan helpdesk resmi.",
        " Informasi lebih lanjut tersedia di portal resmi kami.",
        " Demikian pemberitahuan ini disampaikan untuk diketahui bersama."
    ]

    for base in INDO_SAFE_SAMPLES:
        for p in safe_prefixes[:2]:
            augmented_safe.append(f"{p}{base}")
        for s in safe_suffixes[:2]:
            augmented_safe.append(f"{base}{s}")

    # Variasi ancaman phishing mendesak
    phish_urgency_prefixes = [
        "PERINGATAN DARURAT: ", "PEMBERITAHUAN SEGERA (1X24 JAM): ", 
        "AKUN ANDA TELAH DIBLOKIR: ", "TINDAKAN MENDESAK DIBUTUHKAN: "
    ]
    phish_urgency_suffixes = [
        " Kegagalan konfirmasi akan menyebabkan pemblokiran akun secara permanen tanpa pemberitahuan lebih lanjut.",
        " Harap segera lakukan pembatalan sebelum saldo tabungan Anda terdebet otomatis.",
        " Klik link resmi kami sekarang juga untuk menghindari sanksi penutupan akun.",
        " Segera masukkan nomor PIN dan password Anda untuk verifikasi identitas darurat."
    ]

    for base in INDO_PHISHING_SAMPLES:
        for p in phish_urgency_prefixes[:2]:
            augmented_phish.append(f"{p}{base}")
        for s in phish_urgency_suffixes[:2]:
            augmented_phish.append(f"{base}{s}")

    return augmented_safe, augmented_phish

def main():
    print("=== PHISGUARD BILINGUAL DATASET EXPANSION PIPELINE ===")
    
    # 1. Muat dataset dasar bahasa Inggris
    if not EN_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset bahasa Inggris tidak ditemukan di: {EN_DATA_PATH}")
        
    df_en = pd.read_csv(EN_DATA_PATH)
    print(f"Dataset Inggris Asli: {len(df_en):,} sampel (Phishing={sum(df_en['is_phishing']==1)}, Safe={sum(df_en['is_phishing']==0)})")
    
    df_en_clean = pd.DataFrame({
        'cleaned_text': df_en['cleaned_text'].astype(str),
        'is_phishing': df_en['is_phishing'].astype(int),
        'language': 'en'
    })

    # 2. Bangun Korpus Indonesia Lengkap + Augmentasi Leksikal
    aug_safe, aug_phish = generate_augmented_variations()
    
    all_indo_safe = INDO_SAFE_SAMPLES + aug_safe
    all_indo_phish = INDO_PHISHING_SAMPLES + aug_phish
    
    cleaned_indo_safe = [clean_text(t) for t in all_indo_safe]
    cleaned_indo_phish = [clean_text(t) for t in all_indo_phish]
    
    # Hapus duplikat tepat
    cleaned_indo_safe = list(set([t for t in cleaned_indo_safe if len(t.split()) >= 4]))
    cleaned_indo_phish = list(set([t for t in cleaned_indo_phish if len(t.split()) >= 4]))
    
    df_indo = pd.DataFrame({
        'cleaned_text': cleaned_indo_safe + cleaned_indo_phish,
        'is_phishing': [0]*len(cleaned_indo_safe) + [1]*len(cleaned_indo_phish),
        'language': 'id'
    })
    
    print(f"Dataset Indonesia Diperkaya: {len(df_indo):,} sampel (Safe={len(cleaned_indo_safe)}, Phishing={len(cleaned_indo_phish)})")

    # 3. Audit Kebocoran Data (Leakage Check)
    set_safe = set(cleaned_indo_safe)
    set_phish = set(cleaned_indo_phish)
    overlap = set_safe.intersection(set_phish)
    if overlap:
        print(f"⚠️ PERINGATAN: Ditemukan {len(overlap)} tumpang tindih safe dan phish. Membersihkan...")
        df_indo = df_indo[~df_indo['cleaned_text'].isin(overlap)]
    else:
        print("✅ Audit Kebocoran: 0% tumpang tindih antara sampel aman dan phishing Indonesia.")

    # 4. Penggabungan Korpus Dwibahasa
    df_bilingual = pd.concat([df_en_clean, df_indo], ignore_index=True)
    
    # Hitung metrik panjang
    df_bilingual['cleaned_char_length'] = df_bilingual['cleaned_text'].apply(len)
    df_bilingual['cleaned_word_count'] = df_bilingual['cleaned_text'].apply(lambda x: len(x.split()))
    
    # Acak baris secara deterministik
    df_bilingual = df_bilingual.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"\nTotal Dataset Dwibahasa Gabungan: {len(df_bilingual):,} sampel")
    print(f"- Bahasa Inggris  : {sum(df_bilingual['language']=='en'):,} sampel")
    print(f"- Bahasa Indonesia: {sum(df_bilingual['language']=='id'):,} sampel")
    print(f"- Total Phishing  : {sum(df_bilingual['is_phishing']==1):,} sampel ({sum(df_bilingual['is_phishing']==1)/len(df_bilingual)*100:.1f}%)")
    print(f"- Total Legitimate: {sum(df_bilingual['is_phishing']==0):,} sampel ({sum(df_bilingual['is_phishing']==0)/len(df_bilingual)*100:.1f}%)")

    # Simpan ke CSV
    df_bilingual.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"✅ File dataset dwibahasa berhasil disimpan ke: {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()
