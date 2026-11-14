import nbformat
from nbclient import NotebookClient
from pathlib import Path

def execute_nb(path_str):
    path = Path(path_str).resolve()
    print(f"Executing {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved {path}!")

if __name__ == '__main__':
    execute_nb('url.ipynb')
