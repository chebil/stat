# Converting Markdown to Jupyter Notebooks - Guide

This repository now has an automated GitHub Actions workflow that converts all markdown files (`.md`) to Jupyter notebooks (`.ipynb`) for use with Jupyter Book.

## Quick Start - Run the Conversion

### Option 1: Via GitHub Web Interface (Easiest)

1. Go to your repository on GitHub: [https://github.com/chebil/stat](https://github.com/chebil/stat)

2. Click on the **"Actions"** tab at the top

3. In the left sidebar, click on **"Convert Markdown to Jupyter Notebooks"**

4. Click the **"Run workflow"** button (on the right side)

5. Keep the branch as `main` and click the green **"Run workflow"** button

6. Wait for the workflow to complete (usually takes 1-2 minutes)
   - You'll see a green checkmark when it's done
   - If there's an error, you'll see a red X

7. Refresh your repository page - all `.md` files will be converted to `.ipynb`!

### Option 2: Via GitHub CLI

If you have [GitHub CLI](https://cli.github.com/) installed:

```bash
gh workflow run "Convert Markdown to Jupyter Notebooks" --repo chebil/stat
```

## What the Workflow Does

When you trigger the workflow, it automatically:

1. ✅ Converts all `.md` files in `part1/`, `part2/`, and `part3/` to `.ipynb` notebooks
2. ✅ Converts `intro.md` to `intro.ipynb`
3. ✅ Updates `_toc.yml` to reference `.ipynb` files instead of `.md` files
4. ✅ Preserves all content including:
   - LaTeX equations
   - Code blocks (converted to executable cells)
   - Images and links
   - MyST markdown directives
5. ✅ Deletes the original `.md` files (after successful conversion)
6. ✅ Commits and pushes all changes to the `main` branch

## Files That Will Be Converted

### Part 1 (10 files)
- `part1/chapter01.md` → `part1/chapter01.ipynb`
- `part1/chapter02.md` → `part1/chapter02.ipynb`
- `part1/ch01_datasets.md` → `part1/ch01_datasets.ipynb`
- `part1/ch01_plotting.md` → `part1/ch01_plotting.ipynb`
- `part1/ch01_summarizing.md` → `part1/ch01_summarizing.ipynb`
- `part1/ch01_plots_summaries.md` → `part1/ch01_plots_summaries.ipynb`
- `part1/ch01_australian_pizzas.md` → `part1/ch01_australian_pizzas.ipynb`
- `part1/ch01_you_should.md` → `part1/ch01_you_should.ipynb`
- `part1/ch02_2d_data.md` → `part1/ch02_2d_data.ipynb`
- `part1/ch02_correlation.md` → `part1/ch02_correlation.ipynb`

### Part 2 (16 files)
- All chapter and section markdown files

### Part 3 (19 files)
- All chapter and section markdown files

### Root (1 file)
- `intro.md` → `intro.ipynb`

**Total: 46+ files**

## After Conversion

Once the workflow completes:

1. Pull the latest changes:
   ```bash
   git pull origin main
   ```

2. Build your Jupyter Book:
   ```bash
   jupyter-book build .
   ```

3. Your book will now use Jupyter notebooks throughout!

## Verification

To verify the conversion was successful:

```bash
# Check that .ipynb files were created
find part1 part2 part3 -name "*.ipynb" | wc -l

# Check that .md files were removed
find part1 part2 part3 -name "*.md" | wc -l  # Should be 0

# View the updated table of contents
cat _toc.yml
```

## Troubleshooting

### Workflow Failed?

1. Check the Actions log for error messages
2. Common issues:
   - Merge conflicts (ensure main branch is clean)
   - File permission issues
   - Syntax errors in markdown files

### Need to Re-run?

If you need to convert again:
1. Restore the `.md` files from git history if needed
2. Run the workflow again following the steps above

### Manual Conversion

If you prefer to run the conversion locally:

```bash
# Clone the repository
git clone https://github.com/chebil/stat.git
cd stat

# Install dependencies
pip install jupyter-book nbformat

# Run the interactive script
python scripts/convert_all_to_notebooks.py

# Or run the automated script
python scripts/convert_all_to_notebooks_auto.py

# Commit and push
git add .
git commit -m "Convert all sections to Jupyter notebooks"
git push origin main
```

## Technical Details

### Scripts Available

1. **`scripts/convert_md_to_ipynb.py`** - Basic conversion utility
2. **`scripts/convert_all_to_notebooks.py`** - Interactive conversion with prompts
3. **`scripts/convert_all_to_notebooks_auto.py`** - Automated conversion (used by GitHub Actions)

### Workflow File

The workflow is defined in `.github/workflows/convert-to-notebooks.yml`

### Code Block Handling

The converter intelligently handles code blocks:
- `` ```python `` or `` ```py `` blocks → Code cells
- `` ``` `` (no language) → Code cells
- `` ```bash ``, `` ```sql ``, etc. → Markdown cells (preserved as code blocks)

### Cell Structure

All notebooks are created with:
- Python 3 kernel
- IPython support
- Empty outputs (cells need to be executed)
- Proper metadata for Jupyter Book

## Support

If you encounter any issues:
1. Check the [GitHub Actions logs](https://github.com/chebil/stat/actions)
2. Review the conversion scripts in the `scripts/` directory
3. Open an issue if you find a bug

---

**Ready to convert?** Go to the [Actions tab](https://github.com/chebil/stat/actions/workflows/convert-to-notebooks.yml) and click "Run workflow"!
