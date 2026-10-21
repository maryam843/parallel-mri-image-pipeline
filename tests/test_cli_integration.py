"""End-to-end runs of the CLI on a small synthetic image set (no real data)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mri_pipeline import cli
from mri_pipeline.report import render_all

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mri_pipeline", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=600,
    )


@pytest.mark.slow
def test_module_help():
    proc = _run("--help")
    assert proc.returncode == 0
    for command in ("run-scaling", "run-ml", "profile", "inventory", "verify-data"):
        assert command in proc.stdout


def test_subcommand_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run-scaling", "--help"])
    assert exc.value.code == 0
    assert "--images" in capsys.readouterr().out


def test_inventory_and_verify(tmp_path, synthetic_dir, capsys):
    manifest = tmp_path / "manifest.csv"
    assert (
        cli.main(
            [
                "inventory",
                "--data-dir",
                str(synthetic_dir),
                "--out-dir",
                str(tmp_path),
                "--write-manifest",
                str(manifest),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "dataset_summary.json").read_text())
    assert summary["n_images"] == 13
    assert summary["n_duplicate_files"] == 1
    assert summary["class_counts"] == {"no": 6, "yes": 7}
    assert (
        cli.main(["verify-data", "--data-dir", str(synthetic_dir), "--manifest", str(manifest)])
        == 0
    )
    rows = manifest.read_text().splitlines()
    manifest.write_text("\n".join(rows[:-1]) + "\n")
    assert (
        cli.main(["verify-data", "--data-dir", str(synthetic_dir), "--manifest", str(manifest)])
        == 1
    )
    assert "extra: 1" in capsys.readouterr().out


def test_profile_command(tmp_path, synthetic_dir):
    assert (
        cli.main(
            [
                "profile",
                "--data-dir",
                str(synthetic_dir),
                "--out-dir",
                str(tmp_path),
                "--images",
                "2",
            ]
        )
        == 0
    )
    rows = read_rows(tmp_path / "profile_fast.csv")
    stages = [r["stage"] for r in rows]
    assert "Hessian" in stages and stages[-1] == "Total"
    total = float(rows[-1]["total_s"])
    assert sum(float(r["total_s"]) for r in rows[:-1]) == pytest.approx(total, rel=1e-6)
    other = next(r for r in rows if r["stage"].startswith("Other"))
    assert float(other["total_s"]) >= 0.0  # stages never add up to more than the total


def test_run_scaling_command(tmp_path, synthetic_dir):
    assert (
        cli.main(
            [
                "run-scaling",
                "--data-dir",
                str(synthetic_dir),
                "--out-dir",
                str(tmp_path),
                "--images",
                "6",
                "--processes",
                "1",
                "2",
                "--repeats",
                "2",
            ]
        )
        == 0
    )
    meta = json.loads((tmp_path / "scaling_meta.json").read_text())
    assert meta["parallel_equals_sequential"] is True
    assert meta["n_images"] == 6 and meta["repeats"] == 2
    assert "hostname" not in json.dumps(meta).lower()
    assert set(meta["machine"]) == {  # no kernel version, host or user name
        "os", "machine", "cpu_model", "os_cpu_count", "physical_cores", "usable_cores",
        "python", "start_method", "packages",
    }  # fmt: skip
    summary = read_rows(tmp_path / "scaling_summary.csv")
    assert [r["config"] for r in summary] == ["sequential", "map_p1", "map_p2"]
    assert float(summary[0]["speedup_mean"]) == 1.0
    strategies = read_rows(tmp_path / "strategy_summary.csv")
    assert len(strategies) == 4
    runs = read_rows(tmp_path / "scaling_runs.csv")
    assert len(runs) == 3 * 2


def test_run_ml_command_and_reproducible(tmp_path, larger_synthetic_dir):
    """Same seed -> identical features and identical CV scores."""
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        assert (
            cli.main(
                [
                    "run-ml",
                    "--fast",
                    "--data-dir",
                    str(larger_synthetic_dir),
                    "--out-dir",
                    str(out),
                    "--seeds",
                    "0",
                    "1",
                    "--folds",
                    "3",
                    "--processes",
                    "2",
                ]
            )
            == 0
        )
        outs.append(out)
    meta_a, meta_b = (json.loads((o / "ml_meta.json").read_text()) for o in outs)
    assert meta_a["feature_sha256"] == meta_b["feature_sha256"]
    assert (outs[0] / "ml_cv_folds.csv").read_text() == (outs[1] / "ml_cv_folds.csv").read_text()
    assert meta_a["n_images"] == 21 and meta_a["n_duplicate_files"] == 1
    assert all(
        v == 0
        for k, v in meta_a["groups_shared_between_train_and_test"].items()
        if k.startswith("grouped")
    )
    summary = read_rows(outs[0] / "ml_cv_summary.csv")
    grouped = [(r["features"], r["model"]) for r in summary if r["cv"] == "grouped"]
    assert grouped[0] == ("none", "majority")
    assert ("glcm_direct_cast", "rf") in grouped and len(grouped) == 10


def test_make_synthetic_command(tmp_path):
    assert cli.main(["make-synthetic", "--out", str(tmp_path / "d"), "--per-class", "3"]) == 0
    assert len(list((tmp_path / "d" / "yes").iterdir())) == 4


def test_render_tables_from_run(tmp_path, synthetic_dir):
    cli.main(
        ["profile", "--data-dir", str(synthetic_dir), "--out-dir", str(tmp_path), "--images", "2"]
    )
    cli.main(
        [
            "run-scaling",
            "--data-dir",
            str(synthetic_dir),
            "--out-dir",
            str(tmp_path),
            "--images",
            "4",
            "--processes",
            "2",
            "--repeats",
            "1",
        ]
    )
    cli.main(
        [
            "run-ml",
            "--data-dir",
            str(synthetic_dir),
            "--out-dir",
            str(tmp_path),
            "--seeds",
            "0",
            "--folds",
            "3",
            "--processes",
            "2",
        ]
    )
    tables = render_all(tmp_path)
    assert {"profile_fast", "scaling", "strategies", "ml", "leakage"} <= set(tables)
    assert "ablation" in tables["ml"]
    assert "Sequential loop (no pool)" in tables["scaling"]


def test_run_scaling_features_task(tmp_path, synthetic_dir):
    assert (
        cli.main(
            [
                "run-scaling",
                "--data-dir",
                str(synthetic_dir),
                "--out-dir",
                str(tmp_path),
                "--images",
                "4",
                "--processes",
                "2",
                "--repeats",
                "1",
                "--no-strategies",
                "--task",
                "features",
                "--tag",
                "feat",
            ]
        )
        == 0
    )
    meta = json.loads((tmp_path / "scaling_meta_feat.json").read_text())
    assert meta["task"] == "features" and meta["parallel_equals_sequential"] is True
    eq = json.loads((tmp_path / "parallel_equivalence_feat.json").read_text())
    assert list(eq["per_config"]["map_p2"]) == ["features"]
    assert not (tmp_path / "strategy_summary_feat.csv").exists()


def test_run_scaling_rejects_unknown_task(tmp_path, synthetic_dir):
    from mri_pipeline.experiments import run_scaling

    with pytest.raises(ValueError, match="task"):
        run_scaling(
            synthetic_dir, tmp_path, n_images=2, process_counts=(1,), repeats=1, task_name="gpu"
        )
