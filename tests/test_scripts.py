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
