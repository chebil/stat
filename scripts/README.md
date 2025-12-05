# Statistics Book - Plot Generation Scripts

This directory contains scripts to generate and upload statistical plots for the Jupyter Book.

## Overview

Two scripts work together to create and deploy publication-quality plots:

1. **`generate_plots.py`** - Generates all statistical plots locally
2. **`upload_images_to_github.py`** - Uploads generated plots to GitHub repository

## Quick Start

### Step 1: Generate Plots

```bash
cd ~/stat  # Navigate to your repository
python scripts/generate_plots.py
```

This will:
- Create `part1/images/` and `part2/images/` directories
- Generate 10 statistical plots (PNG format, 300 DPI)
- Create a README documenting all plots

**Generated files:**
- `part1/images/fig_1_1_bar_charts.png` - Gender and goals bar charts
- `part1/images/fig_1_2_histograms.png` - Net worth and cheese histograms
- `part1/images/fig_1_3_conditional_histograms.png` - Body temperature by gender
- `part1/images/fig_1_4_standard_normal.png` - Standard normal distribution
- `part1/images/fig_1_5_boxplots.png` - Box plot comparisons
- `part1/images/fig_2_1_scatter_correlations.png` - Different correlation types
- `part1/images/fig_2_2_height_weight.png` - Height vs weight regression
- `part1/images/fig_10_1_iris_scatter.png` - Iris dataset visualization
- `part1/images/fig_10_2_scatterplot_matrix.png` - Scatterplot matrix
- `part2/images/fig_normal_distributions.png` - Normal distributions with parameters

### Step 2: Upload to GitHub

**Option A: Using the upload script (recommended)**

```bash
# Set your GitHub personal access token
export GITHUB_TOKEN='your_github_token_here'

# Run the upload script
python scripts/upload_images_to_github.py
```

**Option B: Using git directly**

```bash
git add part1/images/ part2/images/
git commit -m "Add generated statistical plots"
git push origin main
```

## Creating a GitHub Personal Access Token

If you don't have a GitHub token:

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "stat-plots-upload"
4. Select scopes:
   - ✅ `repo` (Full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. Set it in your environment:
   ```bash
   export GITHUB_TOKEN='ghp_xxxxxxxxxxxxxxxxxxxxx'
   ```

## Requirements

### Python Packages

```bash
pip install matplotlib numpy pandas seaborn requests
```

Or use the project's requirements file:

```bash
pip install -r requirements.txt
```

## Script Details

### generate_plots.py

**Purpose:** Generate all statistical visualization plots

**Features:**
- Creates publication-quality plots (300 DPI)
- Uses consistent styling (seaborn whitegrid)
- Generates both Part 1 and Part 2 plots
- Creates documentation (README.md)

**Customization:**

You can modify the plots by editing `generate_plots.py`:
- Change colors: Edit the `color` parameters
- Adjust sizes: Modify `figsize` tuples
- Change DPI: Update `dpi=300` in `savefig()` calls
- Add new plots: Add new function calls in `main()`

### upload_images_to_github.py

**Purpose:** Upload generated images to GitHub repository

**Features:**
- Uses GitHub REST API
- Handles both new files and updates
- Provides progress feedback
- Base64 encodes binary files automatically

**Configuration:**

Edit these variables at the top of the script:
```python
OWNER = 'chebil'  # Your GitHub username
REPO = 'stat'     # Repository name
BRANCH = 'main'    # Target branch
```

## Troubleshooting

### Plots not showing in Jupyter Book?

1. **Check image paths in markdown files:**
   ```markdown
   ```{figure} images/fig_1_1_bar_charts.png
   :name: fig-bar-charts
   :width: 90%
   ```
   ```

2. **Verify directory structure:**
   ```
   stat/
   ├── part1/
   │   ├── ch01_plotting.md
   │   └── images/
   │       ├── fig_1_1_bar_charts.png
   │       └── ...
   ```

3. **Rebuild Jupyter Book:**
   ```bash
   jupyter-book clean stat/
   jupyter-book build stat/
   ```

### Upload script fails?

**Error: "GITHUB_TOKEN not set"**
- Solution: Export your GitHub token:
  ```bash
  export GITHUB_TOKEN='your_token'
  ```

**Error: "401 Unauthorized"**
- Solution: Token may be invalid or expired. Generate a new one.

**Error: "404 Not Found"**
- Solution: Check `OWNER` and `REPO` values in the script.

### Import errors?

```bash
pip install --upgrade matplotlib numpy pandas seaborn
```

## Manual Workflow (Alternative)

If the automated scripts don't work, you can:

1. **Generate plots locally:**
   ```bash
   python scripts/generate_plots.py
   ```

2. **View generated files:**
   ```bash
   ls -lh part1/images/
   ```

3. **Upload via GitHub web interface:**
   - Go to https://github.com/chebil/stat
   - Navigate to `part1/images/`
   - Click "Add file" → "Upload files"
   - Drag and drop the images
   - Commit changes

## Maintenance

### Adding new plots

1. Edit `scripts/generate_plots.py`
2. Add your plot generation function
3. Call it from `main()`
4. Run the script
5. Upload using the upload script or git

### Updating existing plots

Just re-run both scripts:
```bash
python scripts/generate_plots.py
python scripts/upload_images_to_github.py
```

The upload script automatically detects existing files and updates them.

## Resources

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [GitHub API Documentation](https://docs.github.com/en/rest)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

## Support

For issues or questions:
1. Check this README
2. Review script output for error messages
3. Verify all requirements are installed
4. Check GitHub Actions logs for build errors

## License

These scripts are part of the statistics textbook project and inherit its license.
