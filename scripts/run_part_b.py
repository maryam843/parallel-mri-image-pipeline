"""Execute the Part B notebook headless and save it with its outputs.

    MRI_PART_B_LICENCE=CC-BY-4.0 MRI_PART_B_DATA_DIR=<data folder> python scripts/run_part_b.py

The data folder is the one that holds Training/ and Testing/. The CNN trains
with --torch-threads CPU threads (default: MRI_TORCH_THREADS, otherwise 4);
the thread count is part of what makes two runs give identical weights.

The notebook writes results/part_b/ and figures/part_b/. Before saving,
execution metadata (timestamps) is removed and every output is scrubbed of the
home folder, the data folder, the user name and the host name, so the saved
notebook holds only code, outputs and execution counts.
"""

from __future__ import annotations

import argparse
import getpass
import os
import platform
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "part_b_larger_dataset_colab.ipynb"


def private_strings() -> list[str]:
    """Local strings that must not appear in saved outputs, longest first."""
    values = {str(Path.home()), os.environ.get("MRI_PART_B_DATA_DIR", "")}
    with_user = {getpass.getuser(), platform.node()}
    values |= {v for v in with_user if len(v) >= 3}
    return sorted((v for v in values if v), key=len, reverse=True)


def scrub(text: str, private: list[str]) -> str:
    for value in private:
        text = re.sub(re.escape(value), "<redacted>", text, flags=re.I)
    return text


def clean(nb: nbformat.NotebookNode, private: list[str]) -> int:
    """Remove execution metadata and private strings in place; return error-cell count."""
    errors = 0
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
        for output in cell.get("outputs", []):
            errors += output.get("output_type") == "error"
            if "text" in output:
                output["text"] = scrub(output["text"], private)
            for key, value in output.get("data", {}).items():
                if key.startswith("text/") and isinstance(value, str):
                    output["data"][key] = scrub(value, private)
            if "traceback" in output:
                output["traceback"] = [scrub(t, private) for t in output["traceback"]]
    nb.metadata.pop("widgets", None)
    return errors


def run(path: Path = NOTEBOOK, timeout: int = 6 * 3600) -> int:
    """Execute top to bottom; stop at the first error and then leave the file unchanged."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        print(scrub(str(exc), private_strings())[-3000:])
        print(f"{path.name}: stopped at an error; the notebook file was not changed")
        return 1
    errors = clean(nb, private_strings())
    nbformat.write(nb, path)
    code_cells = sum(c.cell_type == "code" for c in nb.cells)
    print(f"executed {path.name}: {code_cells} code cells, {errors} error cells")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the Part B notebook headless.")
    parser.add_argument("--torch-threads", type=int, default=None, help="CPU threads for the CNN")
    args = parser.parse_args(argv)
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            parser.error("--torch-threads must be at least 1")
        os.environ["MRI_TORCH_THREADS"] = str(args.torch_threads)  # read by the notebook kernel
    return run()


if __name__ == "__main__":
    sys.exit(main())
