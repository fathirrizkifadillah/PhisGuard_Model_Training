import urllib.request
import ssl
import zipfile
import io
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/external')
URL_DIR = DATA_DIR / 'url'
EMAIL_DIR = DATA_DIR / 'email'

URL_DIR.mkdir(parents=True, exist_ok=True)
EMAIL_DIR.mkdir(parents=True, exist_ok=True)

print("1. Menyiapkan dataset URL eksternal...")

# 1.1 OpenPhish Live Feed
openphish_path = URL_DIR / 'openphish_live_phishing.csv'
try:
    ctx = ssl.create_default_context()
    with urllib.request.urlopen('https://openphish.com/feed.txt', timeout=10, context=ctx) as r:
        lines = [line.strip() for line in r.read().decode('utf-8', errors='ignore').splitlines() if line.strip()]
    df_openphish = pd.DataFrame({'URL': lines, 'is_phishing': 1, 'source': 'OpenPhish Live Community Feed'})
    df_openphish.to_csv(openphish_path, index=False)
    print(f"- OpenPhish Live Feed berhasil diunduh: {len(df_openphish)} URL ({openphish_path})")
except Exception as e:
    print(f"- Gagal mengunduh OpenPhish: {e}")

# 1.2 Cisco Umbrella Top Legitimate Domains
umbrella_path = URL_DIR / 'umbrella_top_legitimate.csv'
try:
    url_umbrella = 'http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip'
    req = urllib.request.Request(url_umbrella, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        zip_data = io.BytesIO(r.read())
        with zipfile.ZipFile(zip_data) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                raw_lines = [f.readline().decode('utf-8').strip() for _ in range(500)]
    
    clean_domains = []
    for line in raw_lines:
        parts = line.split(',')
        if len(parts) >= 2:
            domain = parts[1].strip()
            # Bentuk format URL resmi dengan https://
            clean_domains.append(f"https://{domain}/")
            
    df_umbrella = pd.DataFrame({'URL': clean_domains, 'is_phishing': 0, 'source': 'Cisco Umbrella Top 1M'})
    df_umbrella.to_csv(umbrella_path, index=False)
    print(f"- Cisco Umbrella Top 500 Legitimate berhasil diunduh: {len(df_umbrella)} URL ({umbrella_path})")
except Exception as e:
    print(f"- Gagal mengunduh Cisco Umbrella: {e}")

# 1.3 Kasus Khusus / Edge Cases URL Riil (Threat Modeling)
curated_url_path = URL_DIR / 'curated_url_edge_cases.csv'
curated_url_data = [
    # Typosquatting & Brand Spoofing Phishing
    {'URL': 'https://paypa1-security-verification.com/login', 'is_phishing': 1, 'category': 'Typosquatting Phish', 'notes': 'Angka 1 menggantikan huruf l pada paypal'},
    {'URL': 'http://apple-id-support-icloud.account-security.top/auth', 'is_phishing': 1, 'category': 'Subdomain Brand Spoof', 'notes': 'Eksploitasi subdomain berlapis dengan TLD .top'},
    {'URL': 'http://192.168.100.55/secure/banking/bca/login.php', 'is_phishing': 1, 'category': 'IP-Based Phish', 'notes': 'Menggunakan IP address langsung tanpa domain'},
    {'URL': 'http://203.0.113.195:8080/portal/microsoft/office365', 'is_phishing': 1, 'category': 'IP with Custom Port', 'notes': 'IP address dengan port non-standar'},
    {'URL': 'https://storage.googleapis.com/phishing-bucket-azure/login.html', 'is_phishing': 1, 'category': 'Cloud Storage Abuse', 'notes': 'Hosting phishing di Google Cloud Storage resmi'},
    {'URL': 'https://bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi.ipfs.dweb.link/', 'is_phishing': 1, 'category': 'IPFS Decentralized Phish', 'notes': 'URL desentralisasi IPFS berkarakter acak panjang'},
    {'URL': 'https://netflix-account-reactivation-billing.info/signin', 'is_phishing': 1, 'category': 'Credential Harvesting', 'notes': 'Kombinasi kata pemicu pada TLD murah .info'},
    {'URL': 'http://secure-login-wellsfargo.com.update-profile.xyz/auth', 'is_phishing': 1, 'category': 'Reverse Subdomain Spoof', 'notes': 'Domain penipu di belakang domain resmi palsu'},
    {'URL': 'https://faceb00k-security-checkpoint.org/confirm.php?user=vict&ref=sec', 'is_phishing': 1, 'category': 'Number Substitution', 'notes': 'Substitusi huruf o dengan angka 0 ganda'},
    {'URL': 'http://download-security-patch-kb5001.update.msi', 'is_phishing': 1, 'category': 'Malicious Extension', 'notes': 'URL berakhiran file installer berbahaya .msi'},
    
    # Legitimate Complex / Edge Cases
    {'URL': 'https://www.google.com/search?q=machine+learning+cybersecurity&oq=ml&sourceid=chrome&ie=UTF-8', 'is_phishing': 0, 'category': 'Legitimate Search Query', 'notes': 'Banyak parameter query pada search engine resmi'},
    {'URL': 'https://drive.google.com/drive/folders/1aBcDeFgHiJkLmNoPqRsTuVwXyZ123456?usp=sharing', 'is_phishing': 0, 'category': 'Legitimate Cloud Sharing', 'notes': 'Link share Google Drive dengan token panjang acak'},
    {'URL': 'https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py', 'is_phishing': 0, 'category': 'Legitimate Developer Code', 'notes': 'Deep path repository kode open source'},
    {'URL': 'https://s3.ap-southeast-1.amazonaws.com/my-corp-bucket/reports/2026/annual-financial-audit.pdf', 'is_phishing': 0, 'category': 'Legitimate AWS S3 Asset', 'notes': 'Subdomain dan path panjang resmi cloud storage'},
    {'URL': 'https://portal.kemenkeu.go.id/layanan/pajak/surat-pemberitahuan-tahunan', 'is_phishing': 0, 'category': 'Legitimate Government Portal', 'notes': 'Domain pemerintah resmi dengan TLD .go.id'},
    {'URL': 'https://id.wikipedia.org/wiki/Kecerdasan_buatan#Pembelajaran_mesin', 'is_phishing': 0, 'category': 'Legitimate Wiki Anchor', 'notes': 'URL ensiklopedia dengan anchor hashtag dan subdomain bahasa'},
    {'URL': 'https://stackoverflow.com/questions/43162506/undefined-reference-to-vtable-when-building-with-cmake', 'is_phishing': 0, 'category': 'Legitimate Tech Community', 'notes': 'Banyak tanda hubung pada slug pertanyaan resmi'},
    {'URL': 'https://www.microsoft.com/en-us/security/business/threat-intelligence', 'is_phishing': 0, 'category': 'Legitimate Corporate Security', 'notes': 'Subdirektori multi-bahasa resmi vendor global'},
    {'URL': 'https://t.co/abcXYZ1234', 'is_phishing': 0, 'category': 'Legitimate Short URL', 'notes': 'URL pendek resmi Twitter/X'},
    {'URL': 'https://arxiv.org/abs/1706.03762', 'is_phishing': 0, 'category': 'Legitimate Academic Paper', 'notes': 'Repositori jurnal ilmiah dengan ID numerik'}
]
df_curated_url = pd.DataFrame(curated_url_data)
df_curated_url.to_csv(curated_url_path, index=False)
print(f"- Curated URL Edge Cases berhasil dibuat: {len(df_curated_url)} kasus ({curated_url_path})")

print("\n2. Menyiapkan dataset Email eksternal...")

# 2.1 Enron External Benchmark (Unseen Corporate Emails)
enron_path = EMAIL_DIR / 'enron_external_benchmark.csv'
try:
    from datasets import load_dataset
    print("- Mengunduh sampel dataset Enron dari Hugging Face...")
    ds_enron = load_dataset('SetFit/enron_spam', split='test[:600]')
    df_enron = pd.DataFrame({
        'text': ds_enron['text'],
        'is_phishing': ds_enron['label'], # 1 = spam/phish, 0 = ham/safe
        'source': 'Enron Corpus (Hugging Face)'
    })
    df_enron.to_csv(enron_path, index=False)
    print(f"- Enron External Benchmark berhasil disimpan: {len(df_enron)} email ({enron_path})")
except Exception as e:
    print(f"- Gagal memuat Enron via datasets library: {e}")

# 2.2 Kasus Nyata / Curated Real-World Email Attack Lures & Normal Workplace Memos
curated_email_path = EMAIL_DIR / 'curated_real_world_email_lures.csv'
curated_email_data = [
    # High-Profile Phishing Attacks / Social Engineering Lures
    {
        'text': 'URGENT: Executive Wire Transfer Authorization Required. Hi Finance Team, I am currently in an off-site confidential acquisition meeting. Please process an urgent wire transfer of $48,500 to our vendor escrow account immediately. Details attached. Send confirmation once wire is completed. Regards, John Doe, Chief Executive Officer.',
        'is_phishing': 1,
        'category': 'CEO Fraud / BEC',
        'notes': 'Penipuan transfer dana mendesak dengan impersonasi eksekutif'
    },
    {
        'text': 'Microsoft 365 Security Notice: Your password expires in 2 hours. Failure to update will result in loss of access to your mailbox and OneDrive. Click the link below to keep your current password and verify credentials: http://msoffice365-verify.com/login?usr=emp',
        'is_phishing': 1,
        'category': 'Credential Harvesting',
        'notes': 'Urgensi palsu pembaruan password cloud dengan link penipuan'
    },
    {
        'text': 'DHL Express: Delivery Exception Notice. Your parcel tracking #8492048 cannot be delivered due to an incorrect destination address. Please download the address verification receipt and pay the $2.99 redelivery fee within 24 hours at http://dhl-parcel-tracking-update.xyz',
        'is_phishing': 1,
        'category': 'Delivery Scam',
        'notes': 'Pemberitahuan paket gagal kirim dengan pungutan biaya kecil'
    },
    {
        'text': 'Internal Revenue Service (IRS): Tax Refund Notification. After the last annual calculation of your fiscal activity, we have determined that you are eligible to receive a tax refund of $1,420.50. Submit your tax refund request online at our secure portal: http://irs-gov-refund-portal.online',
        'is_phishing': 1,
        'category': 'Tax Scam',
        'notes': 'Umpan pengembalian dana pajak dari institusi pemerintah palsu'
    },
    {
        'text': 'IT Helpdesk Alert: System Maintenance and Quota Exceeded. Your email storage is at 99.8% capacity. Outgoing emails have been restricted. Verify your mailbox quota now to avoid permanent account deletion: http://webmail-quota-increase.info/auth',
        'is_phishing': 1,
        'category': 'IT Quota Scare',
        'notes': 'Ancaman kuota email penuh untuk menjebak kredensial webmail'
    },
    {
        'text': 'Zoom Video Communications: You have missed an urgent video conference meeting from HR Department. View the recording and meeting transcript here: http://zoom-meetings-web-conference.top/call/rec',
        'is_phishing': 1,
        'category': 'Remote Work Lure',
        'notes': 'Pemberitahuan panggilan video tak terjawab bertema rapat HR'
    },
    {
        'text': 'DocuSign: Document Pending Signature. Please sign Invoice_94022.pdf requested by Accounting Department. Review document: http://docusign-signature-review.net/doc/94022',
        'is_phishing': 1,
        'category': 'DocuSign Impersonation',
        'notes': 'Penyalahgunaan reputasi platform tanda tangan digital'
    },
    {
        'text': 'Binance Security: Unauthorized withdrawal attempt detected from IP 185.220.101.5. If this was not you, cancel the transaction immediately by securing your wallet: http://binance-security-action.com/wallet',
        'is_phishing': 1,
        'category': 'Crypto Wallet Theft',
        'notes': 'Peringatan penarikan dana fiktif untuk memicu kepanikan investor crypto'
    },

    # Normal Workplace Emails / Legitimate Memos (False Positive Stress-Test)
    {
        'text': 'Sprint Planning Agenda for Monday. Hi team, please find the agenda for our upcoming Sprint 42 planning session attached. We will review backlog items, allocate story points, and discuss technical debt in the payment gateway service. Let me know if you want to add any topics. Best, Sarah.',
        'is_phishing': 0,
        'category': 'Legitimate Agile Meeting',
        'notes': 'Komunikasi harian tim software engineering'
    },
    {
        'text': 'Monthly Financial Budget Report. Dear department heads, please review your Q3 operating expenditure spreadsheet on SharePoint. Submit any variances exceeding 5% by this Friday afternoon. Thank you for your cooperation, Accounting Team.',
        'is_phishing': 0,
        'category': 'Legitimate Finance Memo',
        'notes': 'Email korporat rutin mengenai pelaporan anggaran'
    },
    {
        'text': 'DevOps Notification: Production deployment completed successfully. Release v2.4.1 has been rolled out to the Kubernetes cluster with zero downtime. Monitoring dashboards on Datadog indicate all microservices are healthy.',
        'is_phishing': 0,
        'category': 'Legitimate DevOps Alert',
        'notes': 'Notifikasi otomatis sistem deployment produksi'
    },
    {
        'text': 'Company Townhall Meeting Link and Details. Dear all, join us this Thursday at 3 PM for our quarterly townhall with leadership. We will discuss market expansion and announce employee recognition awards. Link to Google Meet is in your calendar invite.',
        'is_phishing': 0,
        'category': 'Legitimate HR Announcement',
        'notes': 'Pengumuman pertemuan seluruh staf dari manajemen HR'
    },
    {
        'text': 'Security Advisory: Routine Password Policy Update. As part of our annual ISO 27001 compliance audit, all employees are reminded that password rotation can be done directly through your Windows Settings (Ctrl+Alt+Del). Do not share your credentials with anyone.',
        'is_phishing': 0,
        'category': 'Legitimate Security Memo',
        'notes': 'Pengingat keamanan resmi tanpa tautan luar mencurigakan'
    },
    {
        'text': 'GitHub: [phishguard/model-repo] Pull request #14 opened by fathir. Review requested on module url_feature_extractor.py. CI checks passed: 12 tests successful.',
        'is_phishing': 0,
        'category': 'Legitimate GitHub Notification',
        'notes': 'Notifikasi pull request kolaborasi repositori GitHub'
    }
]
df_curated_email = pd.DataFrame(curated_email_data)
df_curated_email.to_csv(curated_email_path, index=False)
print(f"- Curated Real-World Email Lures & Memos berhasil dibuat: {len(df_curated_email)} kasus ({curated_email_path})")

print("\nSemua dataset eksternal dan kasus uji riil berhasil disiapkan!")
