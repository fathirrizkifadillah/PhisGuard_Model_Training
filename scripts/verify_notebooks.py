import nbformat
import re
from pathlib import Path

def audit_notebook(path_str):
    p = Path(path_str)
    print(f"\n================ AUDITING {p.name} ================")
    if not p.exists():
        print(f"File {p} does not exist!")
        return False
    with open(p, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    md_cells = [c for c in nb.cells if c.cell_type == 'markdown']
    print(f"Total cells: {len(nb.cells)} ({len(md_cells)} markdown, {len(code_cells)} code)")
    
    empty_outputs = []
    has_errors = []
    for idx, c in enumerate(code_cells):
        outputs = c.get('outputs', [])
        if len(outputs) == 0:
            empty_outputs.append(idx)
        for o in outputs:
            if o.get('output_type') == 'error':
                has_errors.append((idx, o.get('ename'), o.get('evalue')))
                
    print(f"Code cells with outputs: {len(code_cells) - len(empty_outputs)} / {len(code_cells)}")
    if empty_outputs:
        print(f"WARNING: Empty output code cell indices: {empty_outputs}")
    else:
        print("ALL code cells have outputs!")
        
    if has_errors:
        print(f"ERRORS found in cells: {has_errors}")
    else:
        print("ZERO errors in notebook execution!")
        
    # Check for emojis
    text_dump = ""
    for c in nb.cells:
        text_dump += c.source + "\n"
        for o in c.get('outputs', []):
            if 'text' in o:
                text_dump += o['text'] + "\n"
            if 'data' in o and 'text/plain' in o['data']:
                text_dump += str(o['data']['text/plain']) + "\n"
                
    # Comprehensive emoji regex pattern
    emoji_pattern = re.compile(
        r'[\U00010000-\U0010ffff]'
        r'|[\u2600-\u27bf]'
        r'|[\u2300-\u23ff]'
        r'|[\u2b50-\u2b55]'
        r'|[\u203c\u2049\u2139\u2194-\u2199\u21a9-\u21aa\u231a\u231b\u23e9-\u23ec\u23f0\u23f3]'
    )
    emojis_found = emoji_pattern.findall(text_dump)
    print(f"Emoji count in notebook: {len(emojis_found)}")
    if emojis_found:
        print(f"Sample emojis: {set(emojis_found[:10])}")
        
    return len(empty_outputs) == 0 and len(has_errors) == 0 and len(emojis_found) == 0

if __name__ == '__main__':
    ok_url = audit_notebook('url.ipynb')
    ok_email = audit_notebook('email.ipynb')
    ok_ext = audit_notebook('external_testing.ipynb')
    print("\n---------------- FINAL VERIFICATION SUMMARY ----------------")
    print(f"url.ipynb              : {'PASSED' if ok_url else 'FAILED'}")
    print(f"email.ipynb            : {'PASSED' if ok_email else 'FAILED'}")
    print(f"external_testing.ipynb : {'PASSED' if ok_ext else 'FAILED'}")
