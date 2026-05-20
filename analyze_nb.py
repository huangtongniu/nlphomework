import json
import sys

def analyze(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== {file_path} ===")
    for i, cell in enumerate(nb['cells']):
        ct = cell['cell_type']
        source = "".join(cell['source'])
        if ct == 'markdown':
            lines = [l for l in source.splitlines() if l.strip().startswith('#')]
            if lines:
                print(f"Cell {i} (Markdown): " + " | ".join(lines))
        elif ct == 'code':
            lines = source.splitlines()
            non_empty = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
            first_line = non_empty[0] if non_empty else (lines[0] if lines else "EMPTY")
            print(f"Cell {i} (Code, lines={len(lines)}): {first_line[:80]}")

analyze('58_COMP90042_Project_2025.ipynb')
print("\n" + "="*50 + "\n")
analyze('classificationAS3.ipynb')
