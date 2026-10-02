"""Second look at the Part B near-duplicate candidates: pixel similarity of each pair.

    MRI_PART_B_DATA_DIR=<data folder> python scripts/check_duplicate_candidates.py

Reads results/part_b/duplicate_candidates.csv (written by the Part B notebook),
loads both images of every pair, resizes them to 128x128 grayscale and writes
the Pearson correlation and mean absolute difference per pair
(duplicate_candidates_pixel.csv) and a summary per scope
(duplicate_candidates_pixel.json) to results/part_b/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from mri_pipeline import part_b
from mri_pipeline.experiments import write_csv, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results") / "part_b")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args(argv)
    root = part_b.data_dir_from_env()
    if root is None:
        print(f"set {part_b.DATA_ENV} to the folder that holds Training/ and Testing/")
        return 1
    with open(args.results / "duplicate_candidates.csv", newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    summary, rows = part_b.check_candidates(root, candidates, size=args.size)
    keep = ["scope", "class_a", "file_a", "class_b", "file_b", "dhash_bits", "phash_bits"]
    write_csv(
        args.results / "duplicate_candidates_pixel.csv",
        [
            {
                **{k: r[k] for k in keep},
                "pearson_r": round(r["pearson_r"], 4),
                "mean_abs_diff": round(r["mean_abs_diff"], 2),
            }
            for r in rows
        ],
    )
    write_json(args.results / "duplicate_candidates_pixel.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
