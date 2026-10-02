"""Image discovery, loading and hashing.

A pattern like ``glob('yes/*.jpg')`` is case-sensitive on Linux and macOS and
never matches ``.jpeg`` or ``.png``, so it silently skips part of this dataset.
Here the extension check is done on the lower-cased suffix, so every supported
image is found on every operating system.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
CLASS_LABELS = {"yes": 1, "no": 0}
DEFAULT_DATA_DIR = Path("data") / "brain_tumor_dataset"


@dataclass(frozen=True)
class ImageRecord:
    """One image file on disk with its class label and content hash."""

    path: Path
    class_name: str
    label: int
    sha256: str


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def is_image_file(path: str | Path) -> bool:
    """True if the file has a supported image extension, in any letter case."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_images(data_dir: str | Path, classes: dict[str, int] | None = None) -> list[ImageRecord]:
    """List every image under ``data_dir/<class>/`` in a stable order.

    The order is sorted by (class name, file name) so that runs are
    reproducible regardless of the order the file system returns entries.
    """
    classes = CLASS_LABELS if classes is None else classes
    data_dir = Path(data_dir)
    records: list[ImageRecord] = []
    for class_name in sorted(classes):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class folder: {class_dir}")
        for path in sorted(class_dir.iterdir(), key=lambda p: p.name):
            if path.is_file() and is_image_file(path):
                records.append(
                    ImageRecord(path, class_name, classes[class_name], sha256_file(path))
                )
    return records


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """Read an image as 8-bit grayscale, optionally resized to ``size x size``.

    ``cv2.imdecode`` on the raw bytes is used instead of ``cv2.imread`` because
    ``imread`` fails on non-ASCII paths on Windows.
    """
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    if size is not None:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def load_images(records: list[ImageRecord], size: int | None = None) -> list[np.ndarray]:
    """Load every record's image (same order as ``records``)."""
    return [load_image(r.path, size=size) for r in records]


def duplicate_groups(hashes: list[str]) -> np.ndarray:
    """Map each hash to an integer group id; identical files share an id."""
    ids: dict[str, int] = {}
    return np.array([ids.setdefault(h, len(ids)) for h in hashes], dtype=np.int64)


def count_by_class_and_extension(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    """Counts like ``{'yes': {'.JPG': 68, '.jpg': 86, ...}, ...}`` (case kept)."""
    counts: dict[str, dict[str, int]] = {}
    for r in records:
        per_class = counts.setdefault(r.class_name, {})
        per_class[r.path.suffix] = per_class.get(r.path.suffix, 0) + 1
    return counts


def write_manifest(records: list[ImageRecord], out_csv: str | Path) -> None:
    """Write a checksum manifest: class, file name, SHA-256 (no absolute paths)."""
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["class", "file_name", "sha256"])
        for r in sorted(records, key=lambda r: (r.class_name, r.path.name)):
            writer.writerow([r.class_name, r.path.name, r.sha256])


def verify_manifest(data_dir: str | Path, manifest_csv: str | Path) -> dict[str, list[str]]:
    """Compare a local copy of the dataset with a manifest.

    Returns ``{'missing': [...], 'mismatched': [...], 'extra': [...]}`` with
    ``class/file_name`` entries; all three lists are empty for a perfect match.
    """
    expected: dict[str, str] = {}
    with open(manifest_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            expected[f"{row['class']}/{row['file_name']}"] = row["sha256"]
    found = {f"{r.class_name}/{r.path.name}": r.sha256 for r in list_images(data_dir)}
    return {
        "missing": sorted(set(expected) - set(found)),
        "mismatched": sorted(k for k in expected.keys() & found.keys() if expected[k] != found[k]),
        "extra": sorted(set(found) - set(expected)),
    }
