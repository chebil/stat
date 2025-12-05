#!/usr/bin/env python3
"""
Script to remove error outputs and re-execute code blocks.
"""

import re
import sys
from pathlib import Path

def remove_error_outputs(markdown_text):
    """Remove output sections that contain errors."""
    # Pattern to find error outputs (both inline and block format)
    # Handle cases with optional blank lines before "**Initial"
    pattern1 = r'\n\n\*\*Output:\*\* `Error:[^`]+`\n+(?=\*\*|##|\n|$)'
    pattern2 = r'\n\n\*\*Output:\*\*\n```\nError:[^`]+```\n+(?=\*\*|##|\n|$)'
    
    modified = markdown_text
    
    # Apply pattern1
    matches1 = list(re.finditer(pattern1, markdown_text))
    count1 = len(matches1)
    if count1 > 0:
        modified = re.sub(pattern1, '\n\n', modified)
    
    # Apply pattern2
    matches2 = list(re.finditer(pattern2, markdown_text, re.DOTALL))
    count2 = len(matches2)
    if count2 > 0:
        modified = re.sub(pattern2, '\n\n', modified, flags=re.DOTALL)
    
    count = count1 + count2
    
    return modified, count

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove error outputs from markdown files')
    parser.add_argument('files', nargs='*', help='Markdown files to process')
    
    args = parser.parse_args()
    
    if not args.files:
        # Find all markdown files in part1 and part2
        files = []
        for part in ['part1', 'part2']:
            part_dir = Path(part)
            if part_dir.exists():
                files.extend(sorted(part_dir.glob('ch*.md')))
    else:
        files = [Path(f) for f in args.files]
    
    print("=" * 60)
    print("Remove Error Outputs Script")
    print("=" * 60)
    print()
    
    total_removed = 0
    for filepath in files:
        if not filepath.exists():
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified, count = remove_error_outputs(content)
        
        if count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified)
            print(f"{filepath.name}: Removed {count} error output(s)")
            total_removed += count
    
    print()
    print("=" * 60)
    print(f"✨ Removed {total_removed} error outputs total")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
