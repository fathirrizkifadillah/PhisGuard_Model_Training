import re
import math
from collections import Counter
from urllib.parse import urlparse
import pandas as pd

POPULAR_TLDS = {
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'info', 'biz',
    'online', 'xyz', 'top', 'site', 'ru', 'de', 'uk', 'id', 'jp', 'cn', 'ca'
}

SUSPICIOUS_EXTS = {
    '.exe', '.zip', '.rar', '.apk', '.bin', '.scr', '.msi', '.iso', '.bat', '.cmd', '.vbs'
}

def extract_single_url(url: str) -> dict:
    url_str = str(url).strip()
    if not url_str.startswith(('http://', 'https://')):
        parsed = urlparse('http://' + url_str)
        https_flag = 0
    else:
        parsed = urlparse(url_str)
        https_flag = 1 if url_str.startswith('https://') else 0

    url_length = len(url_str)
    dot_count = url_str.count('.')
    number_of_digits = sum(c.isdigit() for c in url_str)
    percentage_numeric_chars = number_of_digits / (url_length + 1e-9)

    # Shannon Entropy
    if url_length > 0:
        counts = Counter(url_str)
        url_entropy = -sum((c / url_length) * math.log2(c / url_length) for c in counts.values())
    else:
        url_entropy = 0.0

    # Hostname & IP analysis
    host = parsed.netloc.split(':')[0]
    has_ip = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', host) else 0

    host_parts = host.split('.')
    if has_ip or len(host_parts) <= 1:
        subdomain_count = 0
        tld = ''
        domain_name = host
    else:
        tld = host_parts[-1]
        domain_name = host_parts[-2] if len(host_parts) >= 2 else host_parts[0]
        subdomain_count = max(0, len(host_parts) - 2)

    tld_length = len(tld)
    has_hyphen_in_domain = 1 if '-' in host else 0
    tld_popularity = 1 if tld.lower() in POPULAR_TLDS else 0
    domain_name_length = len(domain_name)

    path = parsed.path
    path_length = len(path) if path else 0
    path_ratio = path_length / (url_length + 1e-9)

    # Query param count
    query = parsed.query
    query_param_count = (query.count('&') + 1) if query else 0

    # Suspicious file extension
    suspicious_file_extension = 1 if any(url_str.lower().endswith(ext) for ext in SUSPICIOUS_EXTS) else 0

    # Token count
    tokens = [t for t in re.split(r'[/_?=&.-]', url_str) if t]
    token_count = len(tokens)

    return {
        'url_length': url_length,
        'has_ip_address': has_ip,
        'dot_count': dot_count,
        'https_flag': https_flag,
        'url_entropy': round(url_entropy, 6),
        'token_count': token_count,
        'subdomain_count': subdomain_count,
        'query_param_count': query_param_count,
        'tld_length': tld_length,
        'path_length': path_length,
        'has_hyphen_in_domain': has_hyphen_in_domain,
        'number_of_digits': number_of_digits,
        'tld_popularity': tld_popularity,
        'suspicious_file_extension': suspicious_file_extension,
        'domain_name_length': domain_name_length,
        'percentage_numeric_chars': round(percentage_numeric_chars, 6),
        'path_ratio': round(path_ratio, 6)
    }

def extract_url_dataframe(urls: list) -> pd.DataFrame:
    records = [extract_single_url(u) for u in urls]
    return pd.DataFrame(records)

if __name__ == '__main__':
    test_urls = [
        'https://keraekken-loagginnusa.godaddysites.com/',
        'http://192.168.1.1/login.php?user=admin&token=12345',
        'https://www.google.com/search?q=cybersecurity'
    ]
    df_feat = extract_url_dataframe(test_urls)
    print("Extracted Features DataFrame Shape:", df_feat.shape)
    print(df_feat[['url_length', 'has_ip_address', 'https_flag', 'url_entropy', 'number_of_digits', 'path_ratio']])
