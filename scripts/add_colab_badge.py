#!/usr/bin/env python3
"""Add an "Open in Colab" badge cell to notebooks.

Inserts a markdown cell at the top of each .ipynb file (excluding build/venv
folders) if a Colab badge is not already present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_OWNER = "chebil"
REPO_NAME = "stat"
REPO_BRANCH = "main"

COLAB_BADGE_IMAGE = "https://colab.research.google.com/assets/colab-badge.svg"
COLAB_BASE_URL = "https://colab.research.google.com/github"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "_build",
    "node_modules",
    ".ipynb_checkpoints",
}


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def build_badge_cell(relative_path: str) -> dict:
    colab_url = f"{COLAB_BASE_URL}/{REPO_OWNER}/{REPO_NAME}/blob/{REPO_BRANCH}/{relative_path}"
    badge_md = (
        f"<a href=\"{colab_url}\" target=\"_blank\" rel=\"noopener noreferrer\">"
        f"<img src=\"{COLAB_BADGE_IMAGE}\" alt=\"Open In Colab\"/>"
        "</a>"
    )
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [badge_md],
    }


def has_colab_badge(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    source = "".join(cell.get("source", []))
    return "colab.research.google.com" in source or "Open In Colab" in source


def process_notebook(path: Path, repo_root: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        nb = json.load(handle)

    cells = nb.get("cells", [])
    if cells and has_colab_badge(cells[0]):
        relative_path = path.relative_to(repo_root).as_posix()
        cells[0] = build_badge_cell(relative_path)
        nb["cells"] = cells

        with path.open("w", encoding="utf-8") as handle:
            json.dump(nb, handle, ensure_ascii=False, indent=1)
            handle.write("\n")

        return True

    relative_path = path.relative_to(repo_root).as_posix()
    badge_cell = build_badge_cell(relative_path)
    nb["cells"] = [badge_cell] + cells

    with path.open("w", encoding="utf-8") as handle:
        json.dump(nb, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    return True


def main() -> None:
    repo_root = Path(os.getcwd()).resolve()
    updated = 0
    for path in repo_root.rglob("*.ipynb"):
        if should_skip(path):
            continue
        if process_notebook(path, repo_root):
            updated += 1

    print(f"Updated {updated} notebook(s).")


if __name__ == "__main__":
    main()
