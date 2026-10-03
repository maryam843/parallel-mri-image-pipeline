"""Rewrite the generated tables in README.md and docs/RESULTS.md from results/.

    python scripts/update_readme_tables.py

Only the text between <!-- BEGIN:name --> and <!-- END:name --> markers is
replaced; the prose around the tables is written by hand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from mri_pipeline.report import render_all

DOCUMENTS = (Path("README.md"), Path("docs") / "RESULTS.md")


def update(readme: Path, results: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    tables = render_all(results)
    changed = []

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in tables:
            raise SystemExit(f"no results for table {name!r}")
        changed.append(name)
        return f"<!-- BEGIN:{name} -->\n{tables[name]}\n<!-- END:{name} -->"

    text = re.sub(r"<!-- BEGIN:(\w+) -->\n.*?<!-- END:\1 -->", replace, text, flags=re.S)
    readme.write_text(text, encoding="utf-8")
    return changed


if __name__ == "__main__":
    for document in DOCUMENTS:
        if document.exists():
            names = update(document, Path("results"))
            print(f"{document}: updated tables:", ", ".join(names))
    sys.exit(0)
