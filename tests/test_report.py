"""Rendering of mean +/- sd in the README tables (single runs have no spread)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mri_pipeline import report

HEADER = [
    "config", "strategy", "processes", "chunksize", "effective_chunksize", "runs",
    "mean_s", "sd_s", "speedup_mean", "speedup_sd", "efficiency",
]  # fmt: skip


def write_scaling(tmp_path: Path, runs: int) -> Path:
    rows = [
        ["sequential", "sequential", 1, "n/a", "n/a", runs, 10.0, 0.5 if runs > 1 else 0.0,
         1.0, 0.0, 1.0],
        ["map_p2", "map", 2, "auto", 2, runs, 5.5, 0.25 if runs > 1 else 0.0,
         1.82, 0.04 if runs > 1 else 0.0, 0.91],
    ]  # fmt: skip
    with open(tmp_path / "scaling_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    meta = {
        "task": "filters", "n_images": 4, "mode": "fast", "repeats": runs, "resize": None,
        "largest_image_pixel_share": 0.5,
        "machine": {"usable_cores": 2, "cpu_model": "test cpu", "start_method": "spawn"},
    }  # fmt: skip
    (tmp_path / "scaling_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_mean_sd_helper():
    assert report.mean_sd("3.865", "0.0", "1", 2) == "3.87 (1 run)"
    assert report.mean_sd(2.307, 0.066, 3, 2) == "2.31 +/- 0.07"


def test_scaling_table_single_run_shows_1_run(tmp_path):
    table = report.scaling_table(write_scaling(tmp_path, runs=1))
    assert "+/- 0.00" not in table
    assert "| Sequential loop (no pool) | 1 | 10.00 (1 run) | 1.00x | 1.00 |" in table
    assert "| Pool.map | 2 | 5.50 (1 run) | 1.82x (1 run) | 0.91 |" in table


def test_scaling_table_repeated_runs_keep_mean_sd(tmp_path):
    table = report.scaling_table(write_scaling(tmp_path, runs=3))
    assert "| Sequential loop (no pool) | 1 | 10.00 +/- 0.50 | 1.00x | 1.00 |" in table
    assert "| Pool.map | 2 | 5.50 +/- 0.25 | 1.82x +/- 0.04 | 0.91 |" in table
    assert "(1 run)" not in table


def test_profile_table_shows_the_current_glcm_label(tmp_path: Path):
    header = "stage,total_s,per_image_s,share_pct\n"
    rows = "GLCM (7 images x 24 features),1.0,0.2,50.0\nTotal,2.0,0.4,100.0\n"
    (tmp_path / "profile_fast.csv").write_text(header + rows, encoding="utf-8")
    (tmp_path / "profile_fast_meta.json").write_text('{"n_images": 5}', encoding="utf-8")
    table = report.profile_table(tmp_path, "fast")
    assert "| GLCM (7 filtered images x 24 features per image) | 1.00 | 0.20 | 50.0 % |" in table
    assert "| **Total** |" in table
