"""Requirements must cover every third-party import and be pinned."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
}


def third_party_imports() -> set[str]:
    found = set()
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top != "mri_pipeline":
                    found.add(top)
    return found


def parse(text: str) -> dict[str, str | None]:
    out = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==\s*[\w.]+)?", line)
        out[m.group(1).lower()] = m.group(2)
    return out


def check(requirements_text: str) -> None:
    reqs = parse(requirements_text)
    for module in third_party_imports():
        package = IMPORT_TO_PACKAGE[module]
        assert package in reqs, f"{module} is imported but {package} is not listed"
        assert reqs[package], f"{package} is not pinned"


def test_requirements_cover_imports_and_are_pinned():
    check((ROOT / "requirements.txt").read_text(encoding="utf-8"))


def test_dev_requirements_pinned():
    reqs = parse((ROOT / "requirements-dev.txt").read_text(encoding="utf-8"))
    assert reqs and all(reqs.values())
    assert "-r requirements.txt" in (ROOT / "requirements-dev.txt").read_text()


def test_no_unused_heavy_dependencies():
    reqs = parse((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    for unused in ("keras", "tensorflow", "seaborn", "jupyter"):
        assert unused not in reqs
