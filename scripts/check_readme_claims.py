"""Check that every number in README.md comes from results/.

    python scripts/check_readme_claims.py [--readme README.md] [--results results]

1. Every block between <!-- BEGIN:name --> and <!-- END:name --> must equal
   the table that mri_pipeline.report renders from results/ right now
   (Part A from results/, Part B from results/part_b/).
2. Every other number in the README prose (outside code blocks, links and those
   blocks) must either match a value in results/ at the precision written, or
   be a structural number listed in STRUCTURAL below with its reason.
3. Forbidden claims and style rules (no "state of the art", no clinical
   claims, no em dashes) must not appear.
Exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import sys
from pathlib import Path

from mri_pipeline.report import render_all

# Numbers that describe the method or the setup, not a measured result.
STRUCTURAL = {  # fmt: skip
    "3202": "course code DSAI3202",
    "3.12": "Python version",
    "1": "count/sigma start",
    "2": "count",
    "3": "count",
    "4": "count",
    "5": "folds/images",
    "6": "filters",
    "7": "images per sample (input + 6 filters)",
    "9": "fast Hessian sigma max",
    "8": "synthetic images per class in the README example",
    "10": "count",
    "16": "full-mode subset size",
    "24": "GLCM features per image",
    "40": "scaling subset size",
    "64": "raw-pixel baseline size",
    "99": "full Hessian sigma max",
    "100": "percent",
    "168": "GLCM features total",
    "255": "grey levels",
    "256": "GLCM levels",
    "0": "range start",
    "85": "coverage target",
    "25": "duplicate files",
    "50": "GitHub size warning (MB)",
    "13": "synthetic test images",
    "0.5": "blob in synthetic data",
    # Direct-cast examples; asserted in tests/test_filters_features.py
    # (test_direct_cast_behaviour), not measured results.
    "200": "direct-cast example input",
    "56": "direct-cast example output (256 - 200)",
    "3.7": "direct-cast example entropy value",
    "175": "direct-cast example output (943 mod 256)",
    # Part B: dataset credits, not measured results.
    "4.0": "licence version, CC BY 4.0",
}
FORBIDDEN = [
    r"state[ -]of[ -]the[ -]art",
    "—",
    r"clinical[- ]grade",
    r"diagnos(?:es|ing) (?:brain )?tumou?rs",
]
# Per-fold or per-pair files: prose may not quote single rows from them.
SKIPPED_CSV = {
    "ml_cv_folds.csv",
    "duplicate_candidates.csv",
    "duplicate_candidates_pixel.csv",
    "cnn_history.csv",
}
# A number, optionally followed by "x" (speedups), not part of a word or a longer number.
# "1,600" is read as one number (thousands separator); "1, 2" stays two numbers.
NUMBER = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?=x?(?:[^\w.,]|[.,](?!\d)|$))"
)


def strip_non_prose(text: str) -> str:
    text = re.sub(r"<!-- BEGIN:(\w+) -->.*?<!-- END:\1 -->", " ", text, flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)  # link targets
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return text


def result_values(results: Path) -> set[float]:
    values: set[float] = set()

    def add(v) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, int | float):
            values.add(float(v))
            if 0 < abs(v) <= 1.5:
                values.add(float(v) * 100)  # the same value written as a percentage
        elif isinstance(v, dict):
            for x in v.values():
                add(x)
        elif isinstance(v, list):
            for x in v:
                add(x)
        elif isinstance(v, str):
            with contextlib.suppress(ValueError):
                add(float(v))

    folders = [results, results / "part_b"]
    for path in sorted(p for folder in folders for p in folder.glob("*.json")):
        add(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(p for folder in folders for p in folder.glob("*.csv")):
        if path.name.endswith("_runs.csv") or path.name in SKIPPED_CSV:
            continue  # prose may only quote summaries
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                add(list(row.values()))
    ds = json.loads((results / "dataset_summary.json").read_text(encoding="utf-8"))
    found = sum(ds["found_by_lowercase_jpg_glob"].values())
    missed = ds["n_images"] - found
    values.update({float(found), float(missed), 100.0 * missed / ds["n_images"]})
    return values


def matches(token: str, values: set[float]) -> bool:
    decimals = len(token.split(".")[1]) if "." in token else 0
    target = float(token)
    return any(round(v, decimals) == target for v in values)


def check(readme: Path, results: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    problems: list[str] = []
    tables = render_all(results)
    blocks = dict(re.findall(r"<!-- BEGIN:(\w+) -->\n(.*?)<!-- END:\1 -->", text, flags=re.S))
    for name, body in blocks.items():
        if name not in tables:
            problems.append(f"table {name}: no results to render it from")
        elif body.strip() != tables[name].strip():
            problems.append(
                f"table {name}: README differs from results (run scripts/update_readme_tables.py)"
            )
    values = result_values(results)
    for line in strip_non_prose(text).splitlines():
        for token in NUMBER.findall(line):
            token = token.replace(",", "")
            if token in STRUCTURAL or matches(token, values):
                continue
            problems.append(f"prose number {token!r} not found in results/: ...{line.strip()[:90]}")
    for pattern in FORBIDDEN:
        for m in re.finditer(pattern, text, flags=re.I):
            problems.append(f"forbidden text {m.group(0)!r}")
    if re.search("[\U0001f300-\U0001faff☀-➿]", text):
        problems.append("emoji found")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="README numbers vs results/")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    problems = check(args.readme, args.results)
    for p in problems:
        print(p)
    print(f"check_readme_claims: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
