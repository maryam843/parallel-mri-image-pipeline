"""Part B: the same pipeline on a larger 4-class brain MRI dataset.

Dataset checks (file counts, exact duplicates by SHA-256, near duplicates by
two 64-bit perceptual hashes), the GLCM worker that loads its own image, and
per-class metrics. The CNN baseline lives in ``mri_pipeline.cnn`` so that this
module does not need PyTorch.

The data folder and the licence confirmation come from environment variables,
so no local path is written into the notebook or the results.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mri_pipeline import io
from mri_pipeline.features import glcm_vector

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
SPLITS = ("Training", "Testing")
EXPECTED_PER_CLASS = {"Training": 1400, "Testing": 400}
DATA_ENV = "MRI_PART_B_DATA_DIR"
LICENCE_ENV = "MRI_PART_B_LICENCE"
LICENCE_ID = "CC-BY-4.0"
HASHES = ("dhash", "phash")
NEAR_THRESHOLDS = (0, 2, 4, 8)
# A pair is a near-duplicate candidate when both hashes differ in at most this many of 64 bits.
NEAR_MAX_BITS = 4


def licence_confirmed(environ: Mapping[str, str] = os.environ) -> bool:
    """True only if the licence variable names the dataset's licence exactly."""
    return environ.get(LICENCE_ENV, "").strip().upper() == LICENCE_ID


def data_dir_from_env(environ: Mapping[str, str] = os.environ) -> Path | None:
    value = environ.get(DATA_ENV, "").strip()
    return Path(value) if value else None


def class_labels(classes: Sequence[str] = CLASSES) -> dict[str, int]:
    return {name: i for i, name in enumerate(classes)}


def list_split_records(
    root: str | Path, classes: Sequence[str] = CLASSES
) -> dict[str, list[io.ImageRecord]]:
    """Every image of each split, sorted by (class, file name), with SHA-256."""
    labels = class_labels(classes)
    return {split: io.list_images(Path(root) / split, labels) for split in SPLITS}


def count_check(
    records: Mapping[str, Sequence[io.ImageRecord]],
    expected_per_class: Mapping[str, int] = EXPECTED_PER_CLASS,
    classes: Sequence[str] = CLASSES,
) -> dict[str, Any]:
    """Found vs expected images per split and class; every difference is listed."""
    found = {
        split: {c: sum(r.class_name == c for r in recs) for c in classes}
        for split, recs in records.items()
    }
    expected = {split: dict.fromkeys(classes, expected_per_class[split]) for split in found}
    differences = [
        {"split": s, "class": c, "found": found[s][c], "expected": expected[s][c]}
        for s in found
        for c in classes
        if found[s][c] != expected[s][c]
    ]
    extensions: dict[str, int] = {}
    for recs in records.values():
        for r in recs:
            extensions[r.path.suffix] = extensions.get(r.path.suffix, 0) + 1
    total_found = sum(sum(v.values()) for v in found.values())
    total_expected = sum(sum(v.values()) for v in expected.values())
    return {
        "found": found,
        "expected": expected,
        "total_found": total_found,
        "total_expected": total_expected,
        "matches_expected": not differences,
        "differences": differences,
        "extensions": dict(sorted(extensions.items())),
    }


# ----------------------------------------------------------- perceptual hashes


def _bits_to_int(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits.astype(np.uint8).ravel()).tobytes(), "big")


def dhash(image: np.ndarray, size: int = 8) -> int:
    """Difference hash: resize to (size+1) x size, compare horizontal neighbours."""
    small = cv2.resize(image, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


def phash(image: np.ndarray, size: int = 8, scale: int = 4) -> int:
    """DCT hash: resize to 32x32, keep the 8x8 lowest frequencies, compare with their median."""
    small = cv2.resize(image, (size * scale, size * scale), interpolation=cv2.INTER_AREA)
    low = cv2.dct(small.astype(np.float32))[:size, :size]
    return _bits_to_int(low > np.median(low))


def image_hashes(path: str | Path) -> dict[str, int]:
    """Worker: both perceptual hashes and the image shape (8-bit grayscale)."""
    image = io.load_image(path)
    return {
        "dhash": dhash(image),
        "phash": phash(image),
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
    }


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bit differences between 64-bit hashes (broadcasting)."""
    return np.bitwise_count(np.bitwise_xor(a, b)).astype(np.int64)


def as_hash_array(values: Sequence[int]) -> np.ndarray:
    return np.array(values, dtype=np.uint64)


def close_pairs(
    a: np.ndarray, b: np.ndarray | None = None, max_bits: int = NEAR_MAX_BITS
) -> np.ndarray:
    """Index pairs (i, j, distance) with distance <= max_bits.

    With ``b=None`` the pairs are within ``a`` (i < j); otherwise ``i`` indexes
    ``a`` and ``j`` indexes ``b``.
    """
    rows = []
    for i in range(len(a)):
        other = a[i + 1 :] if b is None else b
        d = hamming(a[i], other)
        for j in np.flatnonzero(d <= max_bits):
            rows.append((i, int(j) + (i + 1 if b is None else 0), int(d[j])))
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def _groups(n: int, pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Connected components (size >= 2) of the pair graph, by union-find."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(i)] = find(j)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    return [m for m in members.values() if len(m) > 1]


def exact_duplicates(records: Mapping[str, Sequence[io.ImageRecord]]) -> dict[str, Any]:
    """Byte-identical files (same SHA-256) within each split and across splits."""
    out: dict[str, Any] = {}
    for split, recs in records.items():
        hashes = [r.sha256 for r in recs]
        counts: dict[str, int] = {}
        for h in hashes:
            counts[h] = counts.get(h, 0) + 1
        dup = {h: c for h, c in counts.items() if c > 1}
        classes_per_hash = {h: {r.class_name for r in recs if r.sha256 == h} for h in dup}
        out[f"within_{split}"] = {
            "files": len(hashes),
            "unique_files": len(counts),
            "duplicate_groups": len(dup),
            "extra_copies": sum(c - 1 for c in dup.values()),
            "groups_spanning_classes": sum(len(v) > 1 for v in classes_per_hash.values()),
        }
    train = {r.sha256 for r in records["Training"]}
    test = [r for r in records["Testing"] if r.sha256 in train]
    out["across_splits"] = {
        "testing_files_with_exact_copy_in_training": len(test),
        "shared_hashes": len({r.sha256 for r in test}),
    }
    return out


def near_duplicates(
    records: Mapping[str, Sequence[io.ImageRecord]],
    hashes: Mapping[str, Sequence[Mapping[str, int]]],
    thresholds: Sequence[int] = NEAR_THRESHOLDS,
    max_bits: int = NEAR_MAX_BITS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Near-duplicate counts per hash and threshold, and the candidate pairs.

    A candidate pair has both dHash and pHash within ``max_bits`` bits. Returns
    (summary, candidate pair rows). Pairs are found within Training, within
    Testing and across the two splits.
    """
    arrays = {
        split: {h: as_hash_array([x[h] for x in hashes[split]]) for h in HASHES} for split in SPLITS
    }
    scopes = {
        "within_Training": ("Training", None),
        "within_Testing": ("Testing", None),
        "across_splits": ("Testing", "Training"),
    }
    top = max(max(thresholds), max_bits)
    counts: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    summary_candidates: dict[str, Any] = {}
    for scope, (sa, sb) in scopes.items():
        found = {}
        for h in HASHES:
            b = None if sb is None else arrays[sb][h]
            found[h] = close_pairs(arrays[sa][h], b, top)
        counts[scope] = {
            h: {f"pairs_within_{t}_bits": int((found[h][:, 2] <= t).sum()) for t in thresholds}
            for h in HASHES
        }
        d_pairs = {(int(i), int(j)): int(d) for i, j, d in found["dhash"] if d <= max_bits}
        both = []
        for i, j, d in found["phash"]:
            if d <= max_bits and (int(i), int(j)) in d_pairs:
                both.append((int(i), int(j), d_pairs[(int(i), int(j))], int(d)))
        rec_a = records[sa]
        rec_b = records[sb] if sb is not None else rec_a
        for i, j, dd, dp in both:
            a, b = rec_a[i], rec_b[j]
            candidates.append(
                {
                    "scope": scope,
                    "split_a": sa,
                    "class_a": a.class_name,
                    "file_a": a.path.name,
                    "split_b": sb or sa,
                    "class_b": b.class_name,
                    "file_b": b.path.name,
                    "dhash_bits": dd,
                    "phash_bits": dp,
                    "exact_copy": a.sha256 == b.sha256,
                }
            )
        entry: dict[str, Any] = {
            "candidate_pairs": len(both),
            "pairs_with_different_classes": sum(
                rec_a[i].class_name != rec_b[j].class_name for i, j, _, _ in both
            ),
            "pairs_that_are_exact_copies": sum(
                rec_a[i].sha256 == rec_b[j].sha256 for i, j, _, _ in both
            ),
        }
        if sb is None:
            groups = _groups(len(rec_a), [(i, j) for i, j, _, _ in both])
            entry["candidate_groups"] = len(groups)
            entry["images_in_candidate_groups"] = sum(len(g) for g in groups)
        else:
            entry["testing_images_with_candidate_in_training"] = len({i for i, *_ in both})
            entry["training_images_involved"] = len({j for _, j, *_ in both})
        summary_candidates[scope] = entry
    summary = {
        "hash_bits": 64,
        "hashes": {
            "dhash": "difference hash: 9x8 grayscale (INTER_AREA), horizontal neighbour compare",
            "phash": "DCT hash: 32x32 grayscale, 8x8 lowest DCT frequencies vs their median",
        },
        "thresholds_bits": list(thresholds),
        "pairs_by_hash_and_threshold": counts,
        "candidate_rule": f"dHash and pHash both within {max_bits} bits",
        "candidate_max_bits": max_bits,
        "candidates": summary_candidates,
    }
    return summary, candidates


def flagged_testing_files(candidates: Sequence[Mapping[str, Any]]) -> set[str]:
    """Testing files ('class/file') with an exact or near-duplicate candidate in Training."""
    return {f"{c['class_a']}/{c['file_a']}" for c in candidates if c["scope"] == "across_splits"}


# ------------------------------------------------------------- GLCM features


def glcm_from_path(path: str | Path, size: int = 256, mode: str = "fast") -> np.ndarray:
    """Worker: load one image (resized to size x size) and return its 168 GLCM features.

    Loading inside the worker means only a file path goes to each process
    and only 168 numbers come back.
    """
    return glcm_vector(io.load_image(path, size=size), mode)


# ------------------------------------------------------------------- metrics


def per_class_rows(
    y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str] = CLASSES
) -> list[dict[str, Any]]:
    """Precision, recall, F1 and support per class, then macro average and accuracy."""
    # Imported here so that pool workers, which import this module, do not load sklearn.
    from sklearn.metrics import precision_recall_fscore_support

    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rows = [
        {
            "class": name,
            "precision": float(p[k]),
            "recall": float(r[k]),
            "f1": float(f[k]),
            "support": int(s[k]),
        }
        for k, name in enumerate(classes)
    ]
    rows.append(
        {
            "class": "macro avg",
            "precision": float(np.mean(p)),
            "recall": float(np.mean(r)),
            "f1": float(np.mean(f)),
            "support": int(np.sum(s)),
        }
    )
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    rows.append(
        {
            "class": "accuracy",
            "precision": accuracy,
            "recall": accuracy,
            "f1": accuracy,
            "support": int(np.sum(s)),
        }
    )
    return rows


def confusion(
    y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str] = CLASSES
) -> list[list[int]]:
    """Rows = true class, columns = predicted class, in ``classes`` order."""
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()


def summarise_runs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean and sd over seeds for every (model, eval_set, class)."""
    keys: list[tuple[str, str, str]] = []
    for r in rows:
        key = (r["model"], r["eval_set"], r["class"])
        if key not in keys:
            keys.append(key)
    out = []
    for model, eval_set, name in keys:
        group = [
            r for r in rows if (r["model"], r["eval_set"], r["class"]) == (model, eval_set, name)
        ]
        entry: dict[str, Any] = {"model": model, "eval_set": eval_set, "class": name}
        for metric in ("precision", "recall", "f1"):
            values = np.array([float(r[metric]) for r in group])
            entry[f"{metric}_mean"] = round(float(values.mean()), 4)
            entry[f"{metric}_sd"] = round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 4)
        entry["support"] = int(group[0]["support"])
        entry["seeds"] = len(group)
        out.append(entry)
    return out


# -------------------------------------------------------------------- memory


def memory_snapshot() -> dict[str, float | None]:
    """Total and available RAM in GB (and free commit on Windows), or None if unknown."""
    gb = 2**30
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total_gb": None, "available_gb": None, "free_commit_gb": None}
        return {
            "total_gb": round(status.ullTotalPhys / gb, 1),
            "available_gb": round(status.ullAvailPhys / gb, 1),
            "free_commit_gb": round(status.ullAvailPageFile / gb, 1),
        }
    info: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                info[key] = float(value.split()[0]) * 1024
    except OSError:
        pass
    total, available = info.get("MemTotal"), info.get("MemAvailable")
    return {
        "total_gb": round(total / gb, 1) if total else None,
        "available_gb": round(available / gb, 1) if available else None,
        "free_commit_gb": None,
    }


# ------------------------------------------------- pixel check of candidates


def pixel_similarity(path_a: str | Path, path_b: str | Path, size: int = 128) -> dict[str, float]:
    """Pearson correlation and mean absolute difference (0-255) of two images
    resized to ``size x size`` grayscale: a second look at a hash candidate."""
    a = io.load_image(path_a, size=size).astype(np.float64).ravel()
    b = io.load_image(path_b, size=size).astype(np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        corr = 1.0 if np.array_equal(a, b) else 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {"pearson_r": corr, "mean_abs_diff": float(np.mean(np.abs(a - b)))}


def _pair_similarity(pair: tuple[str, str], size: int = 128) -> dict[str, float]:
    return pixel_similarity(pair[0], pair[1], size)


def check_candidates(
    root: str | Path,
    candidates: Sequence[Mapping[str, Any]],
    size: int = 128,
    processes: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pixel similarity of every candidate pair, summarised per scope."""
    from mri_pipeline.parallel import run

    root = Path(root)
    pairs = [
        (
            str(root / c["split_a"] / c["class_a"] / c["file_a"]),
            str(root / c["split_b"] / c["class_b"] / c["file_b"]),
        )
        for c in candidates
    ]
    sims = run(_pair_similarity, pairs, processes=processes, worker_threads=1) if pairs else []
    rows = [{**dict(c), **s} for c, s in zip(candidates, sims, strict=True)]
    summary: dict[str, Any] = {
        "image_size": size,
        "measures": "Pearson r and mean absolute difference (0-255) of the two grayscale images",
        "r_thresholds": [0.99, 0.95, 0.90, 0.80],
    }
    for scope in ("within_Training", "within_Testing", "across_splits"):
        mine = [r for r in rows if r["scope"] == scope]
        if not mine:
            summary[scope] = {"pairs": 0}
            continue
        r = np.array([x["pearson_r"] for x in mine])
        mad = np.array([x["mean_abs_diff"] for x in mine])
        summary[scope] = {
            "pairs": len(mine),
            "pearson_r_min": round(float(r.min()), 4),
            "pearson_r_median": round(float(np.median(r)), 4),
            "pairs_r_at_least_0.99": int((r >= 0.99).sum()),
            "pairs_r_at_least_0.95": int((r >= 0.95).sum()),
            "pairs_r_at_least_0.90": int((r >= 0.90).sum()),
            "pairs_r_below_0.80": int((r < 0.80).sum()),
            "mean_abs_diff_median": round(float(np.median(mad)), 2),
            "mean_abs_diff_max": round(float(mad.max()), 2),
        }
    across = [r for r in rows if r["scope"] == "across_splits"]
    for t in (0.99, 0.95, 0.90):
        files = {f"{r['class_a']}/{r['file_a']}" for r in across if r["pearson_r"] >= t}
        summary[f"testing_images_with_training_pair_r_at_least_{t:.2f}"] = len(files)
    return summary, rows
