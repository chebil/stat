#!/usr/bin/env python3
"""
Convert all markdown files in the stat repository to Jupyter notebooks.
This script will:
1. Find all .md files in part1/, part2/, and part3/ directories
2. Convert them to .ipynb format
3. Update _toc.yml to reference .ipynb files
4. Preserve all content including LaTeX equations and code blocks
"""

import os
import json
import re
from pathlib import Path
import yaml


def create_notebook_cell(content, cell_type='markdown'):
    """Create a notebook cell dictionary."""
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': content.split('\n')
    }
    
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    
    return cell


def parse_markdown_to_cells(md_content):
    """Parse markdown content into notebook cells."""
    cells = []
    
    # Split content by code blocks (```python or ```)
    parts = re.split(r'(```(?:python)?\n[\s\S]*?\n```)', md_content)
    
    for part in parts:
        if not part.strip():
            continue
            
        # Check if this is a code block
        code_match = re.match(r'```(?:python)?\n([\s\S]*?)\n```', part)
        if code_match:
            code_content = code_match.group(1)
            cells.append(create_notebook_cell(code_content, 'code'))
        else:
            # This is markdown content
            cells.append(create_notebook_cell(part, 'markdown'))
    
    return cells


def convert_md_to_ipynb(md_file_path, ipynb_file_path):
    """Convert a markdown file to Jupyter notebook format."""
    print(f"Converting {md_file_path} -> {ipynb_file_path}")
    
    # Read markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Parse into cells
    cells = parse_markdown_to_cells(md_content)
    
    # Create notebook structure
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0',
                'mimetype': 'text/x-python',
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'pygments_lexer': 'ipython3',
                'nbconvert_exporter': 'python',
                'file_extension': '.py'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    # Write notebook file
    with open(ipynb_file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Created {ipynb_file_path}")


def update_toc_file(toc_path):
    """Update _toc.yml to reference .ipynb files instead of .md files."""
    print(f"\nUpdating {toc_path}...")
    
    with open(toc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all .md references with .ipynb (except intro.md which we'll keep)
    updated_content = re.sub(
        r'file: (part[123]/[^\n]+)\.md',
        r'file: \1.ipynb',
        content
    )
    
    with open(toc_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"  ✓ Updated {toc_path}")


def main():
    """Main conversion function."""
    print("Starting conversion of markdown files to Jupyter notebooks...\n")
    
    # Get the repository root directory
    repo_root = Path.cwd()
    if repo_root.name == 'scripts':
        repo_root = repo_root.parent
    
    # Directories to process
    directories = ['part1', 'part2', 'part3']
    
    converted_files = []
    
    for directory in directories:
        dir_path = repo_root / directory
        if not dir_path.exists():
            print(f"Warning: Directory {directory} not found")
            continue
        
        print(f"Processing {directory}/...")
        
        # Find all .md files in this directory
        md_files = list(dir_path.glob('*.md'))
        
        for md_file in md_files:
            # Create corresponding .ipynb filename
            ipynb_file = md_file.with_suffix('.ipynb')
            
            # Convert the file
            convert_md_to_ipynb(md_file, ipynb_file)
            converted_files.append(str(md_file.relative_to(repo_root)))
        
        print()
    
    # Update _toc.yml
    toc_path = repo_root / '_toc.yml'
    if toc_path.exists():
        update_toc_file(toc_path)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Converted {len(converted_files)} markdown files to Jupyter notebooks.")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("1. Review the generated .ipynb files")
    print("2. Run 'jupyter-book build .' to build the book")
    print("3. Optionally delete the original .md files if everything looks good")
    print("\nNote: The original .md files have been preserved.")
    print("      You can delete them after verifying the notebooks.")


if __name__ == '__main__':
    main()
