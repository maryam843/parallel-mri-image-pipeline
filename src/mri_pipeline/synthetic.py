"""Small synthetic image sets for tests, CI and demos (no real MRI data needed).

The images only need to exercise the pipeline: different sizes, mixed file
extensions in different letter cases, and one exact duplicate file. "yes"
images get a bright blob so that a classifier has something to learn.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

EXTENSION_CYCLE = (".jpg", ".JPG", ".jpeg", ".png")


def synthetic_image(rng: np.random.Generator, size: tuple[int, int], tumour: bool) -> np.ndarray:
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = 60 + 40 * np.sin(xx / max(w, 1) * np.pi) + rng.normal(0, 12, size=(h, w))
    if tumour:
        cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
        radius = max(3, min(h, w) // 6)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))
        base = base + 150 * blob
    return np.clip(base, 0, 255).astype(np.uint8)


def write_image(path: Path, image: np.ndarray) -> None:
    """Encode by the lower-cased extension, write under the given name."""
    ok, buffer = cv2.imencode(path.suffix.lower(), image)
    if not ok:  # pragma: no cover - OpenCV always encodes these formats
        raise ValueError(f"could not encode {path}")
    path.write_bytes(buffer.tobytes())


def make_dataset(out_dir: str | Path, n_per_class: int = 6, seed: int = 0) -> dict[str, int]:
    """Create ``out_dir/yes`` and ``out_dir/no`` and return file counts.

    One extra file in ``yes`` is a byte-for-byte copy of the first ``yes``
    image, so the set contains exactly one exact duplicate.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    counts = {}
    for class_name, tumour in (("yes", True), ("no", False)):
        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            size = (int(rng.integers(40, 97)), int(rng.integers(40, 97)))
            ext = EXTENSION_CYCLE[i % len(EXTENSION_CYCLE)]
            write_image(
                class_dir / f"{class_name}_{i:03d}{ext}", synthetic_image(rng, size, tumour)
            )
        counts[class_name] = n_per_class
    first = sorted((out_dir / "yes").iterdir())[0]
    (out_dir / "yes" / f"copy_of_{first.name}").write_bytes(first.read_bytes())
    counts["yes"] += 1
    # A non-image file that a loader must ignore.
    (out_dir / "no" / "notes.txt").write_text("not an image\n", encoding="utf-8")
    return counts


def make_split_dataset(
    out_dir: str | Path,
    classes: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary"),
    per_class: dict[str, int] | None = None,
    seed: int = 0,
) -> dict[str, dict[str, int]]:
    """Create ``out_dir/<split>/<class>/`` folders like the Part B dataset.

    Class k > 0 gets a bright blob of a class-specific size; the first class
    named "notumor" (if any) gets none. The first Training image of the first
    class is copied byte for byte into Testing (one cross-split exact copy),
    and the second one is copied with a small brightness change (one
    near duplicate). Returns the file counts per split and class.
    """
    per_class = per_class or {"Training": 6, "Testing": 3}
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    counts: dict[str, dict[str, int]] = {}
    for split, n in per_class.items():
        counts[split] = {}
        for k, class_name in enumerate(classes):
            class_dir = out_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                image = synthetic_image(rng, (64, 64), tumour=False)
                if class_name != "notumor":
                    yy, xx = np.mgrid[0:64, 0:64]
                    radius = 4 + 4 * k
                    blob = np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * radius**2))
                    image = np.clip(image + 150 * blob, 0, 255).astype(np.uint8)
                write_image(class_dir / f"{split[:2]}-{class_name}_{i:03d}.jpg", image)
            counts[split][class_name] = n
    first_class = classes[0]
    train_files = sorted((out_dir / "Training" / first_class).iterdir())
    test_dir = out_dir / "Testing" / first_class
    (test_dir / "Te-copy_000.jpg").write_bytes(train_files[0].read_bytes())
    near = cv2.imdecode(np.fromfile(str(train_files[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    write_image(
        test_dir / "Te-near_000.png", np.clip(near.astype(int) + 3, 0, 255).astype(np.uint8)
    )
    counts["Testing"][first_class] += 2
    return counts
