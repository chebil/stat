#!/usr/bin/env python3
"""
Script to remove all outputs from markdown files.
"""

import re
import sys
from pathlib import Path

def remove_all_outputs(markdown_text):
    """Remove all output sections."""
    # Pattern to find all outputs (both inline and block format)
    pattern1 = r'\n\n\*\*Output:\*\* `[^`]+`\n+'
    pattern2 = r'\n\n\*\*Output:\*\*\n```\n[^`]+```\n+'
    pattern3 = r'\n\n\*\*Output:\*\*\n```\n[^`]+```\n\n!\[Plot\]\(images/[^\)]+\)\n+'
    pattern4 = r'\n\n\*\*Output:\*\*\n`[^`]+`\n\n!\[Plot\]\(images/[^\)]+\)\n+'
    pattern5 = r'\n\n\*\*Output:\*\*\n!\[Plot\]\(images/[^\)]+\)\n+'
    
    modified = markdown_text
    total_count = 0
    
    # Try all patterns
    for pattern in [pattern3, pattern4, pattern5, pattern2, pattern1]:
        count = len(re.findall(pattern, modified, re.DOTALL))
        if count > 0:
            modified = re.sub(pattern, '\n\n', modified, flags=re.DOTALL)
            total_count += count
    
    return modified, total_count

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove all outputs from markdown files')
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
    print("Remove All Outputs Script")
    print("=" * 60)
    print()
    
    total_removed = 0
    for filepath in files:
        if not filepath.exists():
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified, count = remove_all_outputs(content)
        
        if count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified)
            print(f"{filepath.name}: Removed {count} output(s)")
            total_removed += count
    
    print()
    print("=" * 60)
    print(f"✨ Removed {total_removed} outputs total")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
