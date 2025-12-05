#!/usr/bin/env python3
"""Test the context persistence"""

import re

markdown = """
```python
x = 5
print(f"x = {x}")
```

**Output:**
```
x = 5
```

```python
print(f"x squared = {x**2}")
```
"""

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

def has_output_after_block(markdown_text, block_end_pos):
    """Check if there's already an output section after this code block."""
    next_section = markdown_text[block_end_pos:block_end_pos + 200]
    return bool(re.match(r'\s*\*\*Output:\*\*', next_section))

blocks = extract_code_blocks(markdown)
print(f"Found {len(blocks)} blocks")

for i, block in enumerate(blocks):
    print(f"\nBlock {i+1}:")
    print(f"  Code: {block['code'][:50]}...")
    print(f"  Has output: {has_output_after_block(markdown, block['end'])}")

# Simulate execution
file_globals = {'__builtins__': __builtins__}

print("\n\nExecuting blocks:")
for i, block in enumerate(blocks):
    if has_output_after_block(markdown, block['end']):
        print(f"Block {i+1}: Skipped (has output)")
        # THIS IS THE BUG! We skip execution so x doesn't get defined!
    else:
        print(f"Block {i+1}: Would execute")
        exec(block['code'], file_globals)
        print(f"  Context now has: {list(file_globals.keys())[-5:]}")
