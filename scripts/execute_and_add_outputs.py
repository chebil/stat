#!/usr/bin/env python3
"""
Script to execute Python code blocks in markdown files and add real outputs with plot images.
"""

import re
import sys
import io
import contextlib
from pathlib import Path
import warnings
import hashlib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def extract_code_blocks(markdown_text):
    """Extract all Python code blocks from markdown."""
    pattern = r'```python\n(.*?)```'
    matches = re.finditer(pattern, markdown_text, re.DOTALL)
    blocks = []
    for match in matches:
        blocks.append({
            'code': match.group(1),
            'start': match.start(),
            'end': match.end(),
            'full_match': match.group(0)
        })
    return blocks

def execute_code_with_context(code, exec_globals, code_hash, images_dir):
    """Execute Python code with persistent context and capture output."""
    # Create string buffers to capture stdout/stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    plot_path = None
    
    # Suppress warnings and capture output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Close any existing plots
                plt.close('all')
                
                # Execute the code with persistent globals
                exec(code, exec_globals)
                
                # Check if plot was created
                has_plot = len(plt.get_fignums()) > 0
                if has_plot:
                    # Save the plot
                    plot_filename = f"output_{code_hash}.png"
                    plot_path = images_dir / plot_filename
                    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                    plt.close('all')
                
            except Exception as e:
                plt.close('all')
                return f"Error: {type(e).__name__}: {e}", None
    
    # Get output
    output = stdout_buffer.getvalue()
    errors = stderr_buffer.getvalue()
    
    # Format output
    result = ""
    if output.strip():
        result = output.strip()
    
    if not result and not plot_path:
        result = "(No output)"
    
    return result, plot_path

def has_output_after_block(markdown_text, block_end_pos):
    """Check if there's already an output section after this code block."""
    # Look at the next 200 characters after the code block
    next_section = markdown_text[block_end_pos:block_end_pos + 200]
    # Check for output with or without double newlines
    return bool(re.match(r'\s*\*\*Output:\*\*', next_section))

def add_outputs_to_markdown(markdown_text, filepath):
    """Add outputs after each Python code block."""
    blocks = extract_code_blocks(markdown_text)
    
    if not blocks:
        return markdown_text, 0
    
    # Setup images directory
    images_dir = filepath.parent / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Import modules once for the file context
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from collections import Counter
        import scipy.stats as stats
        
        # Create persistent execution environment for the entire file
        file_globals = {
            '__builtins__': __builtins__,
            'np': np,
            'pd': pd,
            'plt': plt,
            'Counter': Counter,
            'stats': stats,
        }
    except ImportError as e:
        return markdown_text, 0
    
    # Execute ALL blocks to build context, even if they have outputs
    outputs = []
    for i, block in enumerate(blocks):
        # Check if output already exists
        has_output = has_output_after_block(markdown_text, block['end'])
        
        # Generate hash for the code block
        code_hash = hashlib.md5(f"{filepath.name}_{i}_{block['code']}".encode()).hexdigest()[:12]
        
        # Always execute to maintain context
        output_text, plot_path = execute_code_with_context(block['code'], file_globals, code_hash, images_dir)
        
        # Only save output if block doesn't already have one
        if has_output:
            outputs.append(None)
        else:
            outputs.append((output_text, plot_path))
    
    # Process blocks in reverse order to maintain positions
    modified_text = markdown_text
    added_count = 0
    
    for block, output in zip(reversed(blocks), reversed(outputs)):
        if output is None:
            continue
        
        output_text, plot_path = output
        
        # Format output section
        output_section = "\n\n**Output:**\n"
        
        if output_text and output_text not in ["(No output)", ""]:
            if '\n' in output_text or len(output_text) > 80:
                output_section += f"```\n{output_text}\n```\n"
            else:
                output_section += f"`{output_text}`\n"
        
        if plot_path:
            if output_text and output_text not in ["(No output)", ""]:
                output_section += "\n"
            output_section += f"![Plot](images/{plot_path.name})\n"
        
        if not plot_path and (not output_text or output_text == "(No output)"):
            output_section = "\n\n**Output:** `(No output)`\n"
        
        # Insert output after code block
        modified_text = (
            modified_text[:block['end']] +
            output_section +
            modified_text[block['end']:]
        )
        added_count += 1
    
    return modified_text, added_count

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
    
    # Add outputs
    modified_content, count = add_outputs_to_markdown(content, filepath)
    
    if count == 0:
        print(f"  ℹ️  No new outputs to add (all blocks already have outputs)")
        return True
    
    # Write back
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"  ✅ Added {count} output(s)")
        return True
    except Exception as e:
        print(f"  ❌ Error writing file: {e}")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add real outputs to Python code blocks in markdown files')
    parser.add_argument('files', nargs='*', help='Markdown files to process (if empty, processes all in part1 and part2)')
    parser.add_argument('--part1', action='store_true', help='Process all files in part1/')
    parser.add_argument('--part2', action='store_true', help='Process all files in part2/')
    
    args = parser.parse_args()
    
    # Determine which files to process
    files_to_process = []
    
    if args.files:
        files_to_process = [Path(f) for f in args.files]
    else:
        # Default: process both part1 and part2
        if args.part1 or (not args.part1 and not args.part2):
            part1_dir = Path('part1')
            if part1_dir.exists():
                files_to_process.extend(sorted(part1_dir.glob('ch*.md')))
        
        if args.part2 or (not args.part1 and not args.part2):
            part2_dir = Path('part2')
            if part2_dir.exists():
                files_to_process.extend(sorted(part2_dir.glob('ch*.md')))
    
    if not files_to_process:
        print("No files to process!")
        return 1
    
    print("=" * 60)
    print("Execute and Add Outputs Script (with Plot Images)")
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
