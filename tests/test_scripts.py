"""Tests for the repository scripts: README claim checker and PII scanner."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


claims = load("check_readme_claims")
pii = load("pii_scan")
tables = load("update_readme_tables")


def test_repo_readme_claims_pass():
    """T11 on the real README and the committed results/."""
    assert claims.check(ROOT / "README.md", ROOT / "results") == []


@pytest.fixture
def results_copy(tmp_path: Path) -> Path:
    out = tmp_path / "results"
    shutil.copytree(ROOT / "results", out)
    return out


def test_claims_detect_changed_number(tmp_path: Path, results_copy: Path):
    readme = tmp_path / "README.md"
    readme.write_text("The pipeline found 253 images and ran 97.13x faster.\n", encoding="utf-8")
    problems = claims.check(readme, results_copy)
    assert len(problems) == 1 and "97.13" in problems[0]


def test_claims_detect_stale_table(tmp_path: Path, results_copy: Path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- BEGIN:scaling -->\n| old |\n<!-- END:scaling -->\n", encoding="utf-8")
    assert any("table scaling" in p for p in claims.check(readme, results_copy))
    assert tables.update(readme, results_copy) == ["scaling"]
    assert claims.check(readme, results_copy) == []


def test_claims_forbidden_text(tmp_path: Path, results_copy: Path):
    readme = tmp_path / "README.md"
    readme.write_text("A state of the art, clinical-grade result \u2014 wow\n", encoding="utf-8")
    problems = claims.check(readme, results_copy)
    assert sum("forbidden" in p for p in problems) == 3


def test_pii_scan_text_patterns():
    secret = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
    text = "\n".join(
        [
            "contact: someone" + "@" + "university.edu.qa",
            "ok: 123456+someone" + "@" + "users.noreply.github.com",
            "server " + ".".join(["10", "1", "2", "3"]) + " and version 5.0.0.93",
            "path " + "/home/" + "alice/project",
            secret,
            "hello student 60012345",
        ]
    )
    hits = pii.scan_text(text, "f", {"5.0.0.93"}, ["60012345"])
    kinds = sorted(h.split(": ")[1] for h in hits)
    assert kinds == ["aws_key", "denylist term #1", "email", "home_path", "ipv4"]
    assert secret not in "\n".join(hits)  # values are masked in the output


def test_pii_scan_paths():
    assert pii.check_path("docs/brief.pdf", 10, "x")
    assert pii.check_path("data/yes/Y1.jpg", 10, "x")
    assert pii.check_path("big.bin", 6 * 1024 * 1024, "x")
    assert pii.check_path("figures/scaling.png", 10, "x") == []


@pytest.mark.slow
def test_pii_scan_history_on_temp_repo(tmp_path: Path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "1+test@users.noreply.github.com")
    (tmp_path / "a.txt").write_text("clean\n")
    git("add", ".")
    git("commit", "-q", "-m", "clean")
    assert pii.main(["--history", "--root", str(tmp_path)]) == 0
    (tmp_path / "a.txt").write_text("mail me: x" + "@" + "corp.example.net\n")
    git("commit", "-q", "-am", "leak")
    (tmp_path / "a.txt").write_text("clean again\n")
    git("commit", "-q", "-am", "remove")
    # the working tree is clean but history still has the leak
    assert pii.main(["--root", str(tmp_path)]) == 0
    assert pii.main(["--history", "--root", str(tmp_path)]) == 1


def test_repo_worktree_is_clean():
    """T12 (working tree). The CI job also runs --history."""
    assert pii.main(["--root", str(ROOT)]) == 0


def test_results_have_no_machine_identifiers():
    for path in (ROOT / "results").glob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        assert "hostname" not in text and "node" not in json.loads(text).get("machine", {})


@pytest.mark.slow
def test_make_figures(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_figures.py"),
            "--results",
            str(ROOT / "results"),
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "ml_accuracy.png",
        "profile_share.png",
        "scaling_speedup.png",
    ]


def test_claims_detect_empty_table_block(tmp_path: Path, results_copy: Path):
    """An empty generated block must fail, not pass silently."""
    readme = tmp_path / "README.md"
    readme.write_text("<!-- BEGIN:ml -->\n<!-- END:ml -->\n", encoding="utf-8")
    assert any("table ml" in p for p in claims.check(readme, results_copy))
    assert tables.update(readme, results_copy) == ["ml"]
    assert claims.check(readme, results_copy) == []


def test_claims_cover_part_b_numbers_and_tables(tmp_path: Path, results_copy: Path):
    part_b = results_copy / "part_b"
    shutil.rmtree(part_b, ignore_errors=True)
    part_b.mkdir()
    counts = {
        "found": {"Training": {"a": 1400}, "Testing": {"a": 400}},
        "expected": {"Training": {"a": 1400}, "Testing": {"a": 400}},
        "total_found": 1800,
        "total_expected": 1800,
        "differences": [],
    }
    (part_b / "dataset_counts.json").write_text(json.dumps(counts), encoding="utf-8")
    (part_b / "cnn_history.csv").write_text("seed,epoch,train_loss\n0,1,0.4321\n", "utf-8")
    from mri_pipeline.report import part_b_counts_table

    table = part_b_counts_table(part_b)
    readme = tmp_path / "README.md"
    readme.write_text(
        f"Part B found 1800 images.\n\n<!-- BEGIN:part_b_counts -->\n{table}\n"
        "<!-- END:part_b_counts -->\n",
        encoding="utf-8",
    )
    assert claims.check(readme, results_copy) == []
    readme.write_text("Part B found 1801 images, loss 0.4321.\n", encoding="utf-8")
    problems = claims.check(readme, results_copy)
    assert len(problems) == 2  # a wrong count, and a number only in a per-epoch file
    readme.write_text(
        "<!-- BEGIN:part_b_counts -->\n| old |\n<!-- END:part_b_counts -->\n", "utf-8"
    )
    assert any("table part_b_counts" in p for p in claims.check(readme, results_copy))


def test_check_duplicate_candidates_script(tmp_path: Path, monkeypatch):
    from mri_pipeline import part_b
    from mri_pipeline.synthetic import make_split_dataset

    script = load("check_duplicate_candidates")
    make_split_dataset(tmp_path / "data", per_class={"Training": 3, "Testing": 1})
    records = part_b.list_split_records(tmp_path / "data")
    hashes = {s: [part_b.image_hashes(r.path) for r in records[s]] for s in part_b.SPLITS}
    _, candidates = part_b.near_duplicates(records, hashes)
    out = tmp_path / "results"
    out.mkdir()
    from mri_pipeline.experiments import write_csv
    from mri_pipeline.report import read_csv as read_csv_rows

    write_csv(out / "duplicate_candidates.csv", candidates)
    monkeypatch.delenv(part_b.DATA_ENV, raising=False)
    assert script.main(["--results", str(out)]) == 1
    monkeypatch.setenv(part_b.DATA_ENV, str(tmp_path / "data"))
    assert script.main(["--results", str(out), "--size", "32"]) == 0
    summary = json.loads((out / "duplicate_candidates_pixel.json").read_text(encoding="utf-8"))
    assert summary["image_size"] == 32
    rows = read_csv_rows(out / "duplicate_candidates_pixel.csv")
    assert len(rows) == len(candidates)


def test_run_part_b_scaling_script(tmp_path: Path, monkeypatch):
    from mri_pipeline import part_b
    from mri_pipeline.synthetic import make_split_dataset

    script = load("run_part_b_scaling")
    assert script.process_counts(12) == [1, 2, 4, 6, 8, 10, 12]
    assert script.process_counts(4) == [1, 2, 4]
    assert script.process_counts(7) == [1, 2, 4, 6, 7]
    monkeypatch.delenv(part_b.DATA_ENV, raising=False)
    assert script.main([]) == 1
    make_split_dataset(tmp_path / "data", per_class={"Training": 2, "Testing": 1})
    monkeypatch.setenv(part_b.DATA_ENV, str(tmp_path / "data"))
    args = ["--results", str(tmp_path / "res"), "--figures", str(tmp_path / "fig"),
            "--images", "4", "--resize", "32", "--repeats", "2",
            "--processes", "1", "2"]  # fmt: skip
    assert script.main(args) == 0
    env = json.loads((tmp_path / "res" / "scaling_environment.json").read_text(encoding="utf-8"))
    assert {"memory_at_start", "benchmark_seconds"} <= set(env)
    assert (tmp_path / "fig" / "scaling_speedup.png").is_file()
    assert (tmp_path / "res" / "scaling_summary.csv").is_file()
