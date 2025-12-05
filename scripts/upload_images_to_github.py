#!/usr/bin/env python3
"""
Upload generated statistical plots to GitHub repository
Run this script after generating the plots with generate_plots.py
"""

import os
import base64
from pathlib import Path
import requests
import json

# Configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
OWNER = 'chebil'
REPO = 'stat'
BRANCH = 'main'

if not GITHUB_TOKEN:
    print("Error: GITHUB_TOKEN environment variable not set")
    print("Please set it with: export GITHUB_TOKEN='your_github_token'")
    print("You can create a token at: https://github.com/settings/tokens")
    exit(1)

# GitHub API base URL
BASE_URL = f'https://api.github.com/repos/{OWNER}/{REPO}/contents'

headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def get_file_sha(file_path):
    """Get the SHA of an existing file (needed for updates)"""
    url = f'{BASE_URL}/{file_path}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['sha']
    return None

def upload_file(local_path, github_path, message):
    """Upload or update a file on GitHub"""
    
    # Read file content
    with open(local_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode('utf-8')
    
    # Check if file exists
    sha = get_file_sha(github_path)
    
    # Prepare request
    url = f'{BASE_URL}/{github_path}'
    data = {
        'message': message,
        'content': content,
        'branch': BRANCH
    }
    
    if sha:
        data['sha'] = sha
        action = 'Updating'
    else:
        action = 'Creating'
    
    # Upload
    print(f"  {action} {github_path}...", end=' ')
    response = requests.put(url, headers=headers, json=data)
    
    if response.status_code in [200, 201]:
        print("✓")
        return True
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"    {response.json().get('message', 'Unknown error')}")
        return False

def main():
    print("="*70)
    print("Uploading Statistical Plots to GitHub")
    print("="*70)
    
    # Define files to upload
    files_to_upload = [
        # Part 1 images
        ('part1/images/fig_1_1_bar_charts.png', 'part1/images/fig_1_1_bar_charts.png'),
        ('part1/images/fig_1_2_histograms.png', 'part1/images/fig_1_2_histograms.png'),
        ('part1/images/fig_1_3_conditional_histograms.png', 'part1/images/fig_1_3_conditional_histograms.png'),
        ('part1/images/fig_1_4_standard_normal.png', 'part1/images/fig_1_4_standard_normal.png'),
        ('part1/images/fig_1_5_boxplots.png', 'part1/images/fig_1_5_boxplots.png'),
        ('part1/images/fig_2_1_scatter_correlations.png', 'part1/images/fig_2_1_scatter_correlations.png'),
        ('part1/images/fig_2_2_height_weight.png', 'part1/images/fig_2_2_height_weight.png'),
        ('part1/images/fig_10_1_iris_scatter.png', 'part1/images/fig_10_1_iris_scatter.png'),
        ('part1/images/fig_10_2_scatterplot_matrix.png', 'part1/images/fig_10_2_scatterplot_matrix.png'),
        ('part1/images/README.md', 'part1/images/README.md'),
        # Part 2 images
        ('part2/images/fig_normal_distributions.png', 'part2/images/fig_normal_distributions.png'),
    ]
    
    success_count = 0
    fail_count = 0
    
    print(f"\nUploading {len(files_to_upload)} files...\n")
    
    for local_path, github_path in files_to_upload:
        if Path(local_path).exists():
            if upload_file(local_path, github_path, f"Add {Path(github_path).name}"):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"  ✗ File not found: {local_path}")
            fail_count += 1
    
    print("\n" + "="*70)
    print(f"Upload complete: {success_count} succeeded, {fail_count} failed")
    print("="*70)
    
    if fail_count == 0:
        print("\n✓ All images uploaded successfully!")
        print(f"\nView your book at: https://{OWNER}.github.io/{REPO}/")
    else:
        print("\n⚠ Some uploads failed. Please check the errors above.")

if __name__ == '__main__':
    main()
