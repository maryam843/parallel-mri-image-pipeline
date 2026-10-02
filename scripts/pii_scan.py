"""Scan the repository (working tree or full git history) for secrets and personal data.

    python scripts/pii_scan.py              # tracked + untracked (not ignored) files
    python scripts/pii_scan.py --history    # every blob and commit in every ref

Extra private terms (for example an ID number or a name) can be checked without
writing them into the repository: put them, comma-separated, in the
PII_DENYLIST environment variable. Exit code 1 if anything is found.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_EMAIL = re.compile(r"^\d+\+[\w-]+@users\.noreply\.github\.com$")
FORBIDDEN_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".dcm", ".nii", ".ipynb_checkpoints"}

# Some patterns are assembled from parts so this file does not match itself.
PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "ipv4": re.compile(
        r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])"
    ),
    "home_path": re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\+Users\\+)[A-Za-z_][\w.-]*"),
    "root_or_tmp_path": re.compile(r"(?<![\w.])/(?:root|tmp)/[\w.-]+"),
    "internal_hostname": re.compile(
        r"\b[\w-]+\.(?:internal|local|lan|corp)\b|\bip-\d+-\d+-\d+-\d+\b"
    ),
    "aws_key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "private_key": re.compile("-----BEGIN " + r"(?:[A-Z]+ )?PRIVATE KEY-----"),
    "slack_or_openai_key": re.compile(r"\b(?:xox[abprs]-[\w-]{10,}|sk-[A-Za-z0-9]{32,})"),
    "kaggle_key": re.compile(r"\"key\"\s*:\s*\"[0-9a-f]{32}\""),
}

# Documentation IP-like values that are not addresses (version numbers).
IPV4_ALLOW = {"0.0.0.0", "127.0.0.1"}
EMAIL_ALLOW_DOMAINS = ("example.com", "example.org")


def version_strings(root: Path) -> set[str]:
    """Pinned versions look like IPv4 addresses (e.g. 5.0.0.93); allow them."""
    out = set()
    for req in root.glob("requirements*.txt"):
        out.update(re.findall(r"==\s*([\d.]+)", req.read_text(encoding="utf-8")))
    return out


def scan_text(text: str, where: str, allow_ip: set[str], deny: list[str]) -> list[str]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                value = match.group(0)
                if kind == "email" and (
                    ALLOWED_EMAIL.match(value) or value.lower().endswith(EMAIL_ALLOW_DOMAINS)
                ):
                    continue
                if kind == "ipv4" and (value in allow_ip or value in IPV4_ALLOW):
                    continue
                hits.append(f"{where}:{lineno}: {kind}: {mask(value)}")
        lowered = line.lower()
        for term in deny:
            if term and term.lower() in lowered:
                hits.append(f"{where}:{lineno}: denylist term #{deny.index(term) + 1}")
    return hits


def mask(value: str) -> str:
    return value if len(value) <= 6 else value[:3] + "..." + value[-2:]


def check_path(name: str, size: int, where: str) -> list[str]:
    hits = []
    suffix = Path(name).suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES or any(part in FORBIDDEN_SUFFIXES for part in Path(name).parts):
        hits.append(f"{where}: forbidden file type: {name}")
    if name.startswith("data/") or ".ipynb_checkpoints" in name or "__pycache__" in name:
        hits.append(f"{where}: forbidden location: {name}")
    if size > MAX_FILE_BYTES:
        hits.append(f"{where}: file larger than 5 MB: {name} ({size} bytes)")
    return hits


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def scan_worktree(root: Path, allow_ip: set[str], deny: list[str]) -> tuple[int, list[str]]:
    names = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    hits: list[str] = []
    count = 0
    for raw in filter(None, names):
        name = raw.decode("utf-8")
        path = root / name
        if not path.is_file():
            continue
        count += 1
        data = path.read_bytes()
        hits += check_path(name, len(data), name)
        hits += scan_text(data.decode("utf-8", errors="replace"), name, allow_ip, deny)
    return count, hits


def scan_history(root: Path, allow_ip: set[str], deny: list[str]) -> tuple[int, list[str]]:
    hits: list[str] = []
    # commit metadata and messages
    log = git(root, "log", "--all", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce%x00%B%x01")
    for entry in log.decode("utf-8").split("\x01"):
        if not entry.strip():
            continue
        sha, an, ae, cn, ce, body = entry.strip("\n").split("\x00", 5)
        where = f"commit {sha[:10]}"
        for email in (ae, ce):
            if not ALLOWED_EMAIL.match(email):
                hits.append(f"{where}: non-noreply identity {mask(email)}")
        hits += scan_text(f"{an}\n{cn}\n{body}", where, allow_ip, deny)
    # every blob reachable from any ref
    objects = git(root, "rev-list", "--objects", "--all").decode("utf-8").splitlines()
    count = 0
    for line in objects:
        if " " not in line:
            continue
        sha, name = line.split(" ", 1)
        if git(root, "cat-file", "-t", sha).strip() != b"blob":
            continue
        count += 1
        data = git(root, "cat-file", "blob", sha)
        where = f"{name}@{sha[:10]}"
        hits += check_path(name, len(data), where)
        hits += scan_text(data.decode("utf-8", errors="replace"), where, allow_ip, deny)
    return count, hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="secret / personal-data scan")
    parser.add_argument("--history", action="store_true", help="scan all commits and blobs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    deny = [t.strip() for t in os.environ.get("PII_DENYLIST", "").split(",") if t.strip()]
    allow_ip = version_strings(args.root)
    scan = scan_history if args.history else scan_worktree
    count, hits = scan(args.root, allow_ip, deny)
    scope = "history (all refs)" if args.history else "working tree"
    for hit in hits:
        print(hit)
    print(f"pii_scan: {scope}: {count} files scanned, {len(deny)} denylist terms, {len(hits)} hits")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
