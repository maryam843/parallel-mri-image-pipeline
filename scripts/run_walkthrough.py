"""Execute notebooks/walkthrough.ipynb top to bottom and save it with its outputs.

    python scripts/run_walkthrough.py            # synthetic images
    MRI_USE_REAL_DATA=1 python scripts/run_walkthrough.py   # real images in data/

Execution metadata (timestamps) is removed before saving, so the saved
notebook holds only code, outputs and execution counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "walkthrough.ipynb"


def run(path: Path = NOTEBOOK) -> int:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    ).execute()
    errors = 0
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
        errors += sum(o.get("output_type") == "error" for o in cell.get("outputs", []))
    nb.metadata.pop("widgets", None)
    nbformat.write(nb, path)
    code_cells = sum(c.cell_type == "code" for c in nb.cells)
    print(f"executed {path.name}: {code_cells} code cells, {errors} error cells")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(run())
