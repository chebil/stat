#!/usr/bin/env python3
"""
Fix LaTeX formatting in markdown files.
Replace \[ \] with $$ $$ for display math.
"""

import re
import sys
from pathlib import Path

def fix_latex_formatting(content):
    """Fix LaTeX display math formatting."""
    # Replace \[ with $$
    content = re.sub(r'^\\\[$', r'$$', content, flags=re.MULTILINE)
    # Replace \] with $$
    content = re.sub(r'^\\\]$', r'$$', content, flags=re.MULTILINE)
    
    return content

def process_file(filepath):
    """Process a single markdown file."""
    print(f"Processing: {filepath.name}")
    
    # Read file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False
    
    # Check if file needs fixing
    if not (r'\[' in content or r'\]' in content):
        print(f"  ℹ️  No LaTeX formatting issues found")
        return True
    
    # Fix formatting
    original = content
    fixed = fix_latex_formatting(content)
    
    if fixed == original:
        print(f"  ℹ️  No changes needed")
        return True
    
    # Count changes
    changes = content.count(r'\[') + content.count(r'\]')
    
    # Write back
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"  ✅ Fixed {changes} LaTeX formatting issues")
        return True
    except Exception as e:
        print(f"  ❌ Error writing file: {e}")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix LaTeX formatting in markdown files')
    parser.add_argument('files', nargs='*', help='Markdown files to process')
    parser.add_argument('--all', action='store_true', help='Process all files in part1, part2, part3')
    
    args = parser.parse_args()
    
    files_to_process = []
    
    if args.files:
        files_to_process = [Path(f) for f in args.files]
    elif args.all:
        for part in ['part1', 'part2', 'part3']:
            part_dir = Path(part)
            if part_dir.exists():
                files_to_process.extend(sorted(part_dir.glob('*.md')))
    else:
        # Default: process part3
        part3_dir = Path('part3')
        if part3_dir.exists():
            files_to_process.extend(sorted(part3_dir.glob('*.md')))
    
    if not files_to_process:
        print("No files to process!")
        return 1
    
    print("=" * 60)
    print("Fix LaTeX Formatting Script")
    print("=" * 60)
    print()
    
    success_count = 0
    for filepath in files_to_process:
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
        
        if process_file(filepath):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"✨ Completed! Successfully processed {success_count}/{len(files_to_process)} files")
    print("=" * 60)
    
    return 0 if success_count == len(files_to_process) else 1

if __name__ == '__main__':
    sys.exit(main())
