"""Part B: how many Testing images of each class have a near duplicate in Training.

    python scripts/part_b_leakage_by_class.py

Reads results/part_b/dataset_counts.json, duplicate_candidates.csv and (if
present) duplicate_candidates_pixel.csv, all written earlier by the notebook and
scripts/check_duplicate_candidates.py, and writes
results/part_b/leakage_by_class.json. The duplicate check is not re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mri_pipeline import part_b
from mri_pipeline.experiments import write_json
from mri_pipeline.report import read_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results") / "part_b")
    args = parser.parse_args(argv)
    counts = json.loads((args.results / "dataset_counts.json").read_text(encoding="utf-8"))
    candidates = read_csv(args.results / "duplicate_candidates.csv")
    pixel_path = args.results / "duplicate_candidates_pixel.csv"
    pixel = read_csv(pixel_path) if pixel_path.exists() else None
    out = part_b.leakage_by_class(counts, candidates, pixel)
    duplicates = args.results / "duplicates.json"
    if duplicates.exists():  # must agree with the count the duplicate check wrote
        near = json.loads(duplicates.read_text(encoding="utf-8"))["near"]["candidates"]
        expected = near["across_splits"]["testing_images_with_candidate_in_training"]
        if expected != out["total_with_candidate_in_training"]:
            print(f"mismatch: {out['total_with_candidate_in_training']} vs {expected}")
            return 1
    write_json(args.results / "leakage_by_class.json", out)
    for row in out["per_class"]:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
