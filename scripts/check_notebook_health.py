#!/usr/bin/env python3
"""
Check notebook health - detect cells with empty source code but outputs.
This helps identify notebooks that were incorrectly edited.
"""
import json
import sys
from pathlib import Path

def check_notebook(notebook_path):
    """Check a notebook for cells with empty source but outputs."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {notebook_path}: {e}")
        return []
    
    issues = []
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        has_output = len(cell.get('outputs', [])) > 0
        cell_type = cell.get('cell_type', 'unknown')
        
        if source.strip() == '' and has_output:
            issues.append({
                'cell_idx': i,
                'cell_type': cell_type,
                'output_count': len(cell.get('outputs', []))
            })
    
    return issues

def main():
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    
    if not notebooks_dir.exists():
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)
    
    notebook_files = sorted(notebooks_dir.glob('*.ipynb'))
    
    if not notebook_files:
        print(f"⚠️  No notebooks found in {notebooks_dir}")
        sys.exit(0)
    
    print("=" * 70)
    print("Notebook Health Check")
    print("=" * 70)
    print()
    
    total_issues = 0
    for nb_path in notebook_files:
        issues = check_notebook(nb_path)
        if issues:
            print(f"⚠️  {nb_path.name}:")
            for issue in issues:
                print(f"   Cell {issue['cell_idx']} ({issue['cell_type']}): "
                      f"Empty source but {issue['output_count']} output(s)")
            total_issues += len(issues)
            print()
        else:
            print(f"✅ {nb_path.name}: OK")
    
    print("=" * 70)
    if total_issues > 0:
        print(f"⚠️  Found {total_issues} cells with empty source code")
        print("   These cells likely lost their source due to incorrect editing.")
        print("   See NOTEBOOK_EDITING_GUIDE.md for prevention tips.")
        sys.exit(1)
    else:
        print("✅ All notebooks are healthy!")
        sys.exit(0)

if __name__ == '__main__':
    main()



