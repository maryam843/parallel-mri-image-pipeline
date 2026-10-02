"""T13: notebooks execute top to bottom headless."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def execute(path: Path) -> nbformat.NotebookNode:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB)}}
    ).execute()
    return nb


@pytest.mark.slow
def test_walkthrough_runs_headless(monkeypatch):
    monkeypatch.delenv("MRI_USE_REAL_DATA", raising=False)
    nb = execute(NB / "walkthrough.ipynb")
    text = json.dumps([c.get("outputs", []) for c in nb.cells])
    assert "identical outputs: True" in text
    assert "groups shared between train and test, grouped CV: 0" in text


@pytest.mark.slow
def test_part_b_stops_at_licence_gate(monkeypatch):
    """Without the licence confirmation the first code cell stops execution."""
    monkeypatch.delenv("MRI_PART_B_LICENCE", raising=False)
    with pytest.raises(CellExecutionError, match="licence not confirmed"):
        execute(NB / "part_b_larger_dataset_colab.ipynb")


@pytest.mark.slow
def test_part_b_gate_stays_closed_for_a_wrong_value(monkeypatch):
    monkeypatch.setenv("MRI_PART_B_LICENCE", "yes")
    with pytest.raises(CellExecutionError, match="licence not confirmed"):
        execute(NB / "part_b_larger_dataset_colab.ipynb")


@pytest.mark.slow
def test_part_b_runs_outside_colab_on_synthetic_data(monkeypatch, tmp_path):
    """Whole notebook, tiny settings, synthetic 4-class data: every output file is written."""
    pytest.importorskip("torch")
    from mri_pipeline.synthetic import make_split_dataset

    make_split_dataset(tmp_path / "data", per_class={"Training": 5, "Testing": 2})
    monkeypatch.setenv("MRI_PART_B_LICENCE", "CC-BY-4.0")
    monkeypatch.setenv("MRI_PART_B_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MRI_PART_B_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("MRI_PART_B_QUICK", "1")
    nb = execute(NB / "part_b_larger_dataset_colab.ipynb")
    text = json.dumps([c.get("outputs", []) for c in nb.cells])
    assert "running outside Colab" in text
    assert "Not on Colab: using the mri_pipeline installed" in text
    assert "parallel outputs identical to sequential: True" in text
    assert str(tmp_path) not in text  # no local path in the outputs
    results = tmp_path / "out" / "results" / "part_b"
    for name in (
        "dataset_counts.json", "duplicates.json", "duplicate_candidates.csv", "image_sizes.json",
        "scaling_summary.csv", "per_class_runs.csv", "per_class_summary.csv",
        "confusion_matrices.json", "cnn_history.csv", "run_info.json",
    ):  # fmt: skip
        assert (results / name).is_file(), name
    info = json.loads((results / "run_info.json").read_text(encoding="utf-8"))
    assert info["settings"]["quick_test_settings"] is True
    assert info["test_images"]["test_without_flagged"] < info["test_images"]["test_as_provided"]
    figures = sorted(p.name for p in (tmp_path / "out" / "figures" / "part_b").iterdir())
    assert figures == [
        "cnn_training_loss.png", "confusion_matrices.png", "per_class_f1.png",
        "scaling_speedup.png",
    ]  # fmt: skip


def test_part_b_never_writes_credentials():
    text = (NB / "part_b_larger_dataset_colab.ipynb").read_text(encoding="utf-8")
    assert "kaggle.json" not in text
    assert "LICENCE_CONFIRMED = False" in text


# ------------------------------------------------ Part B install cell
import importlib.util  # noqa: E402
import re  # noqa: E402
import types  # noqa: E402

_spec = importlib.util.spec_from_file_location("pii_scan", ROOT / "scripts" / "pii_scan.py")
pii = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pii)


def install_cell_source() -> str:
    nb = nbformat.read(NB / "part_b_larger_dataset_colab.ipynb", as_version=4)
    cells = [c.source for c in nb.cells if "def install_mri_pipeline" in c.source]
    assert len(cells) == 1
    return cells[0]


def load_install_cell() -> dict:
    """Execute the cell's definitions only (not the final Colab-only install call)."""
    src = install_cell_source().split("\nif IN_COLAB:")[0]
    namespace: dict = {"__name__": "install_cell"}
    exec(compile(src, "install_cell", "exec"), namespace)
    return namespace


def test_part_b_licence_gate_is_still_the_first_code_cell():
    nb = nbformat.read(NB / "part_b_larger_dataset_colab.ipynb", as_version=4)
    first_code = next(c for c in nb.cells if c.cell_type == "code")
    assert "LICENCE_CONFIRMED = False" in first_code.source
    assert "raise SystemExit" in first_code.source
    assert "CC BY 4.0" in first_code.source
    assert '"MRI_PART_B_LICENCE"' in first_code.source


def test_part_b_colab_only_steps_are_guarded():
    src = install_cell_source()
    assert "if IN_COLAB:\n    print(install_mri_pipeline())" in src
    nb = nbformat.read(NB / "part_b_larger_dataset_colab.ipynb", as_version=4)
    colab = [c.source for c in nb.cells if "kagglehub" in c.source or "!pip" in c.source]
    assert colab and all("IN_COLAB" in s for s in colab)


def test_part_b_install_cell_has_no_credentials():
    src = install_cell_source()
    hits = pii.scan_text(src, "install_cell", set(), [])
    assert hits == []
    assert not re.search(r"https://[^/\s:@]+:[^@\s]+@", src)  # no user:secret@host URLs
    assert not re.search(r"(?i)token\s*=\s*[\"'][^\"']+[\"']", src)  # no token literal
    assert not re.search(r"(?i)print\([^)]*token", src)  # token never printed
    assert "userdata.get(name)" in src and '_colab_secret("GITHUB_TOKEN")' in src
    order = [src.index(k) for k in ("# (a)", "# (b)", "# (c)", "# (d)")]
    assert order == sorted(order)


class _Result:
    def __init__(self, code: int = 0):
        self.returncode = code


def test_part_b_install_path_c_keeps_token_out_of_argv(tmp_path):
    ns = load_install_cell()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("env") or {}))
        return _Result(0)

    ns["_local_copy"] = lambda: None
    ns["_colab_secret"] = lambda name: "fake-secret-value-123" if name == "GITHUB_TOKEN" else None
    ns["subprocess"] = types.SimpleNamespace(run=fake_run)
    ns["CLONE_DIR"] = tmp_path / "clone"
    pip_calls = []

    def fake_pip(*args, env=None):
        pip_calls.append(args)
        return len(pip_calls) > 1  # (b) public install fails, install after clone works

    ns["_pip_install"] = fake_pip
    assert ns["install_mri_pipeline"]().startswith("(c)")
    ((cmd, env),) = calls
    assert cmd[:2] == ["git", "clone"]
    assert all("fake-secret" not in part for part in cmd)
    assert "fake-secret" not in " ".join(str(a) for a in pip_calls)
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_part_b_install_path_d_stops_with_the_three_options():
    ns = load_install_cell()
    ns["_local_copy"] = lambda: None
    ns["_colab_secret"] = lambda name: None
    ns["_pip_install"] = lambda *a, **k: False
    with pytest.raises(SystemExit) as exc:
        ns["install_mri_pipeline"]()
    message = str(exc.value)
    assert "(a)" in message and "(b)" in message and "(c)" in message
    assert "GITHUB_TOKEN" in message
