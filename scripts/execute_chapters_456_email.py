import nbformat
from nbclient import NotebookClient
from pathlib import Path

nb_path = Path('email.ipynb').resolve()
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=300, kernel_name='python3')
print("Starting execution of cells 59 to 71 in email.ipynb...")
with client.setup_kernel():
    for idx, cell in enumerate(nb.cells):
        if idx >= 59 and cell.cell_type == 'code':
            print(f"Executing cell {idx}...")
            client.execute_cell(cell, idx)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Cells 59 to 71 in email.ipynb executed and saved successfully!")
