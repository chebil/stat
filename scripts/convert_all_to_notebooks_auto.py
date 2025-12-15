#!/usr/bin/env python3
"""
Automated conversion script for GitHub Actions.
Converts all markdown files to Jupyter notebooks without user interaction.
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
    lines = content.split('\n')
    # Ensure each line ends properly for JSON serialization
    source = [line + '\n' for line in lines[:-1]]
    if lines:
        source.append(lines[-1])  # Last line without \n
    return {
        'cell_type': 'markdown',
        'id': None,
        'metadata': {},
        'source': source
    }


def create_code_cell(content: str) -> Dict:
    """Create a code cell."""
    lines = content.split('\n')
    # Ensure each line ends properly for JSON serialization
    source = [line + '\n' for line in lines[:-1]]
    if lines:
        source.append(lines[-1])  # Last line without \n
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': None,
        'metadata': {},
        'outputs': [],
        'source': source
    }


def parse_markdown_content(md_content: str) -> List[Dict]:
    """
    Parse markdown content and separate code blocks from text.
    Returns a list of cells.
    """
    cells = []
    current_pos = 0
    
    # Find all code blocks
    code_pattern = re.compile(r'```([\w]*)\n(.*?)```', re.MULTILINE | re.DOTALL)
    
    for match in code_pattern.finditer(md_content):
        # Add markdown content before this code block
        if match.start() > current_pos:
            markdown_content = md_content[current_pos:match.start()].strip()
            if markdown_content:
                cells.append(create_markdown_cell(markdown_content))
        
        # Add code cell
        lang = match.group(1)
        code = match.group(2).rstrip('\n')
        
        # Only create code cells for python code or unspecified language
        if not lang or lang.lower() in ['python', 'py', 'python3']:
            if code.strip():
                cells.append(create_code_cell(code))
        else:
            # Keep as markdown if it's another language
            cells.append(create_markdown_cell(match.group(0)))
        
        current_pos = match.end()
    
    # Add any remaining markdown content
    if current_pos < len(md_content):
        markdown_content = md_content[current_pos:].strip()
        if markdown_content:
            cells.append(create_markdown_cell(markdown_content))
    
    # If no cells were created, add the entire content as markdown
    if not cells and md_content.strip():
        cells.append(create_markdown_cell(md_content.strip()))
    
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
    updated_content = re.sub(
        r'file: (part[123]/[^\s]+)\.md',
        r'file: \1.ipynb',
        content
    )
    
    # Also update intro if it's .md
    updated_content = re.sub(
        r'root: intro$',
        r'root: intro.ipynb',
        updated_content,
        flags=re.MULTILINE
    )
    
    with open(toc_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"  ✓ Updated: {toc_path}")


def delete_md_files(converted_files: List[tuple]):
    """Delete original markdown files after conversion."""
    print("\nDeleting original .md files...")
    for md_file, _ in converted_files:
        try:
            md_file.unlink()
            print(f"  ✓ Deleted: {md_file}")
        except Exception as e:
            print(f"  ✗ Error deleting {md_file}: {e}")


def main():
    """
    Main conversion function for automated execution.
    """
    print("="*70)
    print("MyST Book Markdown to Jupyter Notebook Converter (Automated)")
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
    
    # Convert intro.md if it exists
    intro_md = repo_root / 'intro.md'
    if intro_md.exists():
        print(f"\nConverting intro.md...")
        print("-" * 50)
        intro_ipynb = convert_md_to_ipynb(intro_md)
        converted_files.append((intro_md, intro_ipynb))
    
    # Update _toc.yml
    toc_path = repo_root / '_toc.yml'
    if toc_path.exists():
        print()
        print("="*70)
        update_toc_yml(toc_path)
    
    # Delete original .md files
    if converted_files:
        print()
        print("="*70)
        delete_md_files(converted_files)
    
    # Summary
    print()
    print("="*70)
    print(f"Conversion Complete!")
    print("="*70)
    print(f"\nConverted {len(converted_files)} files:")
    part1_count = len([f for f in converted_files if 'part1' in str(f[0])])
    part2_count = len([f for f in converted_files if 'part2' in str(f[0])])
    part3_count = len([f for f in converted_files if 'part3' in str(f[0])])
    intro_count = len([f for f in converted_files if 'intro' in str(f[0])])
    
    print(f"  - part1: {part1_count} files")
    print(f"  - part2: {part2_count} files")
    print(f"  - part3: {part3_count} files")
    if intro_count:
        print(f"  - intro: {intro_count} file")
    
    print("\n" + "="*70)
    print("All files converted successfully!")
    print("Changes will be committed and pushed automatically.")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConversion cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
