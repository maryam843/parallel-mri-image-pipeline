from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mri_pipeline import io


def test_finds_every_extension_in_any_case(synthetic_dir: Path) -> None:
    records = io.list_images(synthetic_dir)
    counts = io.count_by_class_and_extension(records)
    assert counts["no"] == {".jpg": 2, ".JPG": 2, ".jpeg": 1, ".png": 1}
    assert counts["yes"] == {".jpg": 3, ".JPG": 2, ".jpeg": 1, ".png": 1}
    assert len(records) == 13
    assert all(r.path.suffix != ".txt" for r in records)


def test_labels_and_stable_order(synthetic_dir: Path) -> None:
    records = io.list_images(synthetic_dir)
    assert [r.label for r in records] == [0] * 6 + [1] * 7
    assert records == io.list_images(synthetic_dir)
    names = [r.path.name for r in records if r.class_name == "no"]
    assert names == sorted(names)


def test_case_insensitive_multi_extension_loading_finds_every_image(synthetic_dir: Path) -> None:
    """.jpg, .JPG, .jpeg and .png are all found; a lowercase '*.jpg' glob would miss some."""
    records = io.list_images(synthetic_dir)
    counts = (
        sum(r.class_name == "yes" for r in records),
        sum(r.class_name == "no" for r in records),
    )
    assert counts == (7, 6)
    suffixes = {r.path.suffix for r in records}
    assert {".jpg", ".JPG", ".jpeg", ".png"} <= suffixes
    lowercase_jpg_only = [r for r in records if r.path.suffix == ".jpg"]
    assert len(lowercase_jpg_only) < len(records)


def test_load_image_grayscale_and_resize(synthetic_dir: Path) -> None:
    record = io.list_images(synthetic_dir)[0]
    image = io.load_image(record.path)
    assert image.dtype == np.uint8 and image.ndim == 2
    small = io.load_image(record.path, size=32)
    assert small.shape == (32, 32)


def test_load_image_rejects_non_image(tmp_path: Path) -> None:
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not really a jpeg")
    with pytest.raises(ValueError, match="could not decode"):
        io.load_image(bad)


def test_non_ascii_path(tmp_path: Path, synthetic_dir: Path) -> None:
    src = io.list_images(synthetic_dir)[0].path
    target = tmp_path / "foldér م" / "img one.png"
    target.parent.mkdir()
    target.write_bytes(src.read_bytes())
    assert io.load_image(target).ndim == 2


def test_missing_class_folder(tmp_path: Path) -> None:
    (tmp_path / "yes").mkdir()
    with pytest.raises(FileNotFoundError, match="missing class folder"):
        io.list_images(tmp_path)


def test_hashes_find_the_duplicate(synthetic_dir: Path) -> None:
    records = io.list_images(synthetic_dir)
    groups = io.duplicate_groups([r.sha256 for r in records])
    assert len(set(groups.tolist())) == len(records) - 1
    dup = [r for r in records if r.path.name.startswith("copy_of_")][0]
    same = [r for r in records if r.sha256 == dup.sha256]
    assert len(same) == 2


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    f = tmp_path / "x.bin"
    f.write_bytes(b"abc" * 1000)
    assert io.sha256_file(f, chunk_size=7) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_manifest_round_trip(tmp_path: Path, synthetic_dir: Path) -> None:
    records = io.list_images(synthetic_dir)
    manifest = tmp_path / "manifest.csv"
    io.write_manifest(records, manifest)
    text = manifest.read_text(encoding="utf-8")
    assert str(synthetic_dir) not in text  # no absolute paths
    assert io.verify_manifest(synthetic_dir, manifest) == {
        "missing": [],
        "mismatched": [],
        "extra": [],
    }
    lines = text.splitlines()
    tampered = [lines[0], lines[1].rsplit(",", 1)[0] + "," + "0" * 64] + lines[3:]
    manifest.write_text("\n".join(tampered + ["no,ghost.jpg," + "1" * 64]) + "\n")
    diff = io.verify_manifest(synthetic_dir, manifest)
    assert len(diff["mismatched"]) == 1
    assert diff["missing"] == ["no/ghost.jpg"]
    assert len(diff["extra"]) == 1
