#!/usr/bin/env python3
"""
Comprehensive script to convert all markdown files to Jupyter notebooks
and update the MyST book structure.

Usage:
    python scripts/convert_all_to_notebooks.py
    
This will:
1. Convert all .md files in part1/, part2/, part3/ to .ipynb
2. Update _toc.yml to reference .ipynb files
3. Optionally delete original .md files after conversion
"""

import os
import json
import re
import sys
from pathlib import Path
from typing import List, Dict


def create_notebook_structure() -> Dict:
    """Create the basic Jupyter notebook structure."""
    return {
        'cells': [],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3 (ipykernel)',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.10.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }


def create_markdown_cell(content: str) -> Dict:
    """Create a markdown cell."""
    return {
        'cell_type': 'markdown',
        'id': None,
        'metadata': {},
        'source': content.split('\n') if '\n' in content else [content]
    }


def create_code_cell(content: str) -> Dict:
    """Create a code cell."""
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': None,
        'metadata': {},
        'outputs': [],
        'source': content.split('\n') if '\n' in content else [content]
    }


def parse_markdown_content(md_content: str) -> List[Dict]:
    """
    Parse markdown content and separate code blocks from text.
    Returns a list of cells.
    """
    cells = []
    
    # Split by code fences
    parts = re.split(r'(```[\w]*\n[\s\S]*?```)', md_content)
    
    for part in parts:
        if not part.strip():
            continue
        
        # Check if this is a code block
        code_match = re.match(r'```([\w]*)\n([\s\S]*?)```', part, re.MULTILINE)
        
        if code_match:
            lang = code_match.group(1)
            code = code_match.group(2).rstrip()
            
            # Only create code cells for python code or unspecified language
            if not lang or lang.lower() in ['python', 'py']:
                cells.append(create_code_cell(code))
            else:
                # Keep as markdown if it's another language
                cells.append(create_markdown_cell(part))
        else:
            # Regular markdown content
            cells.append(create_markdown_cell(part.rstrip()))
    
    return cells


def convert_md_to_ipynb(md_path: Path) -> Path:
    """
    Convert a single markdown file to Jupyter notebook.
    Returns the path to the created notebook.
    """
    print(f"Converting: {md_path}")
    
    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Create notebook structure
    notebook = create_notebook_structure()
    
    # Parse content into cells
    cells = parse_markdown_content(md_content)
    notebook['cells'] = cells
    
    # Create output path
    ipynb_path = md_path.with_suffix('.ipynb')
    
    # Write notebook
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"  ✓ Created: {ipynb_path}")
    return ipynb_path


def update_toc_yml(toc_path: Path):
    """Update _toc.yml to reference .ipynb files instead of .md."""
    print(f"\nUpdating {toc_path}...")
    
    with open(toc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace .md with .ipynb for files in part1, part2, part3
    # But keep intro.md as is (or convert it too if needed)
    updated_content = re.sub(
        r'file: (part[123]/[^\s]+)\.md',
        r'file: \1.ipynb',
        content
    )
    
    with open(toc_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"  ✓ Updated: {toc_path}")


def convert_intro_to_notebook(repo_root: Path):
    """Convert intro.md to intro.ipynb if it exists."""
    intro_md = repo_root / 'intro.md'
    if intro_md.exists():
        print(f"\nConverting intro.md...")
        intro_ipynb = convert_md_to_ipynb(intro_md)
        
        # Update _toc.yml root reference
        toc_path = repo_root / '_toc.yml'
        if toc_path.exists():
            with open(toc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            updated_content = content.replace('root: intro', 'root: intro.ipynb')
            
            with open(toc_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"  ✓ Updated root in _toc.yml")


def main():
    """
    Main conversion function.
    """
    print("="*70)
    print("MyST Book Markdown to Jupyter Notebook Converter")
    print("="*70)
    print()
    
    # Determine repository root
    repo_root = Path.cwd()
    if repo_root.name == 'scripts':
        repo_root = repo_root.parent
    
    print(f"Repository root: {repo_root}")
    print()
    
    # Directories to process
    directories = ['part1', 'part2', 'part3']
    
    converted_files = []
    
    # Convert all markdown files in each directory
    for directory in directories:
        dir_path = repo_root / directory
        
        if not dir_path.exists():
            print(f"Warning: Directory {directory} not found")
            continue
        
        print(f"\nProcessing {directory}/...")
        print("-" * 50)
        
        # Find all .md files
        md_files = sorted(dir_path.glob('*.md'))
        
        if not md_files:
            print(f"  No .md files found in {directory}/")
            continue
        
        for md_file in md_files:
            ipynb_path = convert_md_to_ipynb(md_file)
            converted_files.append((md_file, ipynb_path))
    
    # Update _toc.yml
    toc_path = repo_root / '_toc.yml'
    if toc_path.exists():
        print()
        print("="*70)
        update_toc_yml(toc_path)
    
    # Ask about intro.md
    convert_intro_to_notebook(repo_root)
    
    # Summary
    print()
    print("="*70)
    print(f"Conversion Complete!")
    print("="*70)
    print(f"\nConverted {len(converted_files)} files:")
    print(f"  - part1: {len([f for f in converted_files if 'part1' in str(f[0])])} files")
    print(f"  - part2: {len([f for f in converted_files if 'part2' in str(f[0])])} files")
    print(f"  - part3: {len([f for f in converted_files if 'part3' in str(f[0])])} files")
    
    # Ask about deleting .md files
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("\n1. Review the generated notebooks")
    print("2. Test build: jupyter-book build .")
    print("3. If everything looks good, delete the original .md files:")
    print("   find part1 part2 part3 -name '*.md' -delete")
    print("4. Commit and push:")
    print("   git add .")
    print("   git commit -m 'Convert all sections to Jupyter notebooks'")
    print("   git push origin main")
    print()
    
    # Offer to delete .md files
    response = input("\nWould you like to delete the original .md files now? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        print("\nDeleting original .md files...")
        for md_file, _ in converted_files:
            try:
                md_file.unlink()
                print(f"  ✓ Deleted: {md_file}")
            except Exception as e:
                print(f"  ✗ Error deleting {md_file}: {e}")
        print("\nDeletion complete!")
    else:
        print("\nOriginal .md files preserved.")
        print("You can delete them manually later if needed.")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConversion cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
