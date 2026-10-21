"""The measured experiments: profiling, scaling and the ML comparison.

Every run writes CSV/JSON into an output folder together with the seed, the
machine, library versions and UTC timestamps, so each number in the README can
be traced to a file.
"""

from __future__ import annotations

import csv
import fnmatch
import functools
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from mri_pipeline import io
from mri_pipeline.features import glcm_vector, image_features, raw_pixel_features
from mri_pipeline.filters import FILTER_NAMES, apply_filters
from mri_pipeline.parallel import (
    auto_chunksize,
    available_cores,
    get_context,
    start_method_from_env,
    timed_run,
)
from mri_pipeline.profile import profile_pipeline
from mri_pipeline.train import MODEL_NAMES, cross_validate, leaked_groups, make_splits

FEATURE_ORDER = ("none", "raw_pixels_64", "glcm_minmax", "glcm_direct_cast")
PACKAGES = (
    "numpy",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "opencv-python-headless",
    "matplotlib",
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "win32":  # platform.processor() only gives the CPU family there
        try:
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def physical_cores() -> int | None:
    """Physical CPU cores (os.cpu_count() counts hardware threads), or None if unknown."""
    if sys.platform == "win32":
        command = "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return int(out) if out.isdigit() else None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    cores = set()
    for block in text.split("\n\n"):
        fields = dict(
            (k.strip(), v.strip()) for k, _, v in (ln.partition(":") for ln in block.splitlines())
        )
        if "core id" in fields:
            cores.add((fields.get("physical id", "0"), fields["core id"]))
    return len(cores) or None


def machine_info(start_method: str | None = None) -> dict[str, Any]:
    """Machine and software description (no hostname, user name or paths)."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "os_cpu_count": os.cpu_count(),
        "physical_cores": physical_cores(),
        "usable_cores": available_cores(),
        "python": platform.python_version(),
        "start_method": start_method or start_method_from_env() or get_context().get_start_method(),
        "packages": versions,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def select_subset(records: list[io.ImageRecord], n_images: int, seed: int) -> list[io.ImageRecord]:
    """A class-balanced subset of unique files (first copy of each hash)."""
    seen: set[str] = set()
    unique: list[io.ImageRecord] = []
    for r in records:
        if r.sha256 not in seen:
            seen.add(r.sha256)
            unique.append(r)
    rng = np.random.default_rng(seed)
    by_class = {
        c: [r for r in unique if r.class_name == c] for c in sorted({r.class_name for r in unique})
    }
    per_class = n_images // len(by_class)
    chosen: list[io.ImageRecord] = []
    for items in by_class.values():
        take = min(per_class, len(items))
        idx = sorted(rng.choice(len(items), size=take, replace=False).tolist())
        chosen.extend(items[i] for i in idx)
    leftovers = [r for r in unique if r not in chosen]
    rng.shuffle(leftovers)
    chosen.extend(leftovers[: max(0, n_images - len(chosen))])
    return sorted(chosen, key=lambda r: (r.class_name, r.path.name))


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def compare_outputs(
    reference: list[dict[str, np.ndarray]] | list[np.ndarray],
    other: list[dict[str, np.ndarray]] | list[np.ndarray],
) -> dict[str, dict[str, bool]]:
    """Per filter: are all images identical (array_equal) and allclose?
    Plain arrays (feature vectors) are compared under the key "features"."""
    if reference and not isinstance(reference[0], dict):
        pairs = list(zip(reference, other, strict=True))
        return {
            "features": {
                "array_equal": all(np.array_equal(a, b) for a, b in pairs),
                "allclose": all(a.shape == b.shape and np.allclose(a, b) for a, b in pairs),
            }
        }
    out = {}
    for name in FILTER_NAMES:
        pairs = [(a[name], b[name]) for a, b in zip(reference, other, strict=True)]
        out[name] = {
            "array_equal": all(np.array_equal(a, b) for a, b in pairs),
            "allclose": all(a.shape == b.shape and np.allclose(a, b) for a, b in pairs),
        }
    return out


# --------------------------------------------------------------------- profile


def run_profile(
    data_dir: Path, out_dir: Path, n_images: int = 5, mode: str = "fast", seed: int = 0
) -> list[dict[str, Any]]:
    started = utc_now()
    records = select_subset(io.list_images(data_dir), n_images, seed)
    images = io.load_images(records)
    rows = profile_pipeline(images, mode)
    for row in rows:
        row["mode"] = mode
        row["n_images"] = len(images)
    write_csv(out_dir / f"profile_{mode}.csv", rows)
    write_json(
        out_dir / f"profile_{mode}_meta.json",
        {
            "what": "cProfile of the sequential pipeline (6 filters + GLCM) per stage",
            "mode": mode,
            "n_images": len(images),
            "seed": seed,
            "images": [f"{r.class_name}/{r.path.name}" for r in records],
            "image_shapes": [list(map(int, im.shape)) for im in images],
            "machine": machine_info(),
            "started_utc": started,
            "finished_utc": utc_now(),
        },
    )
    return rows


# --------------------------------------------------------------------- scaling

TASK_DESCRIPTIONS = {
    "filters": (
        "filter stage (6 filters per image, the 7 arrays returned to the parent), "
        "timed end to end including pool start-up"
    ),
    "features": (
        "6 filters + GLCM per image inside the worker, only the 168-value feature "
        "vector returned to the parent, timed end to end including pool start-up"
    ),
}


def run_scaling(
    data_dir: Path,
    out_dir: Path,
    n_images: int = 40,
    process_counts: tuple[int, ...] = (1, 2, 3, 4),
    repeats: int = 3,
    mode: str = "fast",
    seed: int = 0,
    compare_strategies: bool = True,
    start_method: str | None = None,
    tag: str = "",
    task_name: str = "filters",
    resize: int | None = None,
    classes: dict[str, int] | None = None,
    worker_threads: int | None = None,
) -> dict[str, Any]:
    """Time the filter stage (6 filters per image, full arrays returned to the
    parent) sequentially and with Pool.map on 1..N processes.

    Repeats are interleaved (all configurations once, then again) so that
    slow drifts of the machine affect every configuration equally.
    ``classes`` maps class folder names to labels (default: the yes/no dataset);
    ``worker_threads`` limits BLAS/OpenMP threads in each pool worker.
    """
    started = utc_now()
    suffix = f"_{tag}" if tag else ""
    records = select_subset(io.list_images(data_dir, classes), n_images, seed)
    images = io.load_images(records, size=resize)
    if task_name == "filters":
        task = functools.partial(apply_filters, mode=mode)
    elif task_name == "features":
        task = functools.partial(glcm_vector, mode=mode)
    else:
        raise ValueError(f"task must be 'filters' or 'features', got {task_name!r}")
    max_p = max(process_counts)

    configs: list[dict[str, Any]] = [
        {"config": "sequential", "strategy": "sequential", "processes": 1, "chunksize": "n/a"}
    ]
    for p in process_counts:
        configs.append(
            {"config": f"map_p{p}", "strategy": "map", "processes": p, "chunksize": "auto"}
        )
    strategy_configs: list[dict[str, Any]] = []
    if compare_strategies:
        strategy_configs = [
            {"config": "map_auto", "strategy": "map", "processes": max_p, "chunksize": "auto"},
            {"config": "map_c1", "strategy": "map", "processes": max_p, "chunksize": 1},
            {
                "config": "imap_unordered_c1",
                "strategy": "imap_unordered",
                "processes": max_p,
                "chunksize": 1,
            },
            {
                "config": "imap_unordered_auto",
                "strategy": "imap_unordered",
                "processes": max_p,
                "chunksize": "auto",
            },
        ]

    runs: list[dict[str, Any]] = []
    reference: list[dict[str, np.ndarray]] | None = None
    equivalence: dict[str, Any] = {}
    for r in range(repeats):
        for cfg in configs + strategy_configs:
            chunk = None if cfg["chunksize"] in ("auto", "n/a") else int(cfg["chunksize"])
            result = timed_run(
                task,
                images,
                strategy=cfg["strategy"],
                processes=cfg["processes"],
                chunksize=chunk,
                start_method=start_method,
                worker_threads=worker_threads,
            )
            if cfg["strategy"] == "sequential" and reference is None:
                reference = result.results
            elif r == 0 and reference is not None:
                equivalence[cfg["config"]] = compare_outputs(reference, result.results)
            runs.append({**cfg, "repeat": r, "seconds": round(result.seconds, 4)})
            del result

    def summarise(names: list[str]) -> list[dict[str, Any]]:
        t_seq, _ = mean_sd([x["seconds"] for x in runs if x["config"] == "sequential"])
        rows = []
        for name in names:
            times = [x["seconds"] for x in runs if x["config"] == name]
            cfg = next(x for x in runs if x["config"] == name)
            m, sd = mean_sd(times)
            if cfg["strategy"] == "sequential":
                s_m, s_sd = 1.0, 0.0  # the baseline, by definition
            else:
                s_m, s_sd = mean_sd([t_seq / t for t in times])
            rows.append(
                {
                    "config": name,
                    "strategy": cfg["strategy"],
                    "processes": cfg["processes"],
                    "chunksize": cfg["chunksize"],
                    "effective_chunksize": (
                        auto_chunksize(len(images), cfg["processes"])
                        if cfg["chunksize"] == "auto"
                        else cfg["chunksize"]
                    ),
                    "runs": len(times),
                    "mean_s": round(m, 3),
                    "sd_s": round(sd, 3),
                    "speedup_mean": round(s_m, 3),
                    "speedup_sd": round(s_sd, 3),
                    "efficiency": round(s_m / cfg["processes"], 3),
                }
            )
        return rows

    scaling_rows = summarise([c["config"] for c in configs])
    write_csv(
        out_dir / f"scaling_runs{suffix}.csv",
        [x for x in runs if x["config"] in {c["config"] for c in configs}],
    )
    write_csv(out_dir / f"scaling_summary{suffix}.csv", scaling_rows)
    strategy_rows: list[dict[str, Any]] = []
    if strategy_configs:
        strategy_rows = summarise([c["config"] for c in strategy_configs])
        write_csv(
            out_dir / f"strategy_runs{suffix}.csv",
            [x for x in runs if x["config"] in {c["config"] for c in strategy_configs}],
        )
        write_csv(out_dir / f"strategy_summary{suffix}.csv", strategy_rows)

    all_equal = all(
        v["array_equal"] and v["allclose"] for cfg in equivalence.values() for v in cfg.values()
    )
    meta = {
        "what": TASK_DESCRIPTIONS[task_name],
        "task": task_name,
        "mode": mode,
        "n_images": len(images),
        "subset_seed": seed,
        "images": [f"{r.class_name}/{r.path.name}" for r in records],
        "resize": resize,
        "total_pixels": int(sum(im.size for im in images)),
        "largest_image_pixel_share": round(
            max(im.size for im in images) / sum(im.size for im in images), 4
        ),
        "largest_image_shape": list(max((im.shape for im in images), key=lambda sh: sh[0] * sh[1])),
        "process_counts": list(process_counts),
        "worker_threads": worker_threads,
        "repeats": repeats,
        "timer": "time.perf_counter",
        "speedup_baseline": "mean of the sequential loop (no pool)",
        "parallel_equals_sequential": all_equal,
        "equivalence_checked_on": "repeat 0 of every parallel configuration vs the sequential loop",
        "machine": machine_info(start_method),
        "started_utc": started,
        "finished_utc": utc_now(),
    }
    write_json(out_dir / f"scaling_meta{suffix}.json", meta)
    write_json(
        out_dir / f"parallel_equivalence{suffix}.json",
        {"all_equal": all_equal, "per_config": equivalence},
    )
    return {"scaling": scaling_rows, "strategies": strategy_rows, "meta": meta}


# -------------------------------------------------------------------------- ML


def array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def run_ml(
    data_dir: Path,
    out_dir: Path,
    mode: str = "fast",
    seeds: tuple[int, ...] = (0, 1, 2),
    n_splits: int = 5,
    processes: int | None = None,
    start_method: str | None = None,
) -> dict[str, Any]:
    """All images, features extracted in parallel, duplicate-grouped
    stratified K-fold CV, repeated over seeds, against baselines."""
    started = utc_now()
    records = io.list_images(data_dir)
    images = io.load_images(records)
    y = np.array([r.label for r in records], dtype=np.int64)
    groups = io.duplicate_groups([r.sha256 for r in records])
    processes = processes or available_cores()

    extracted = timed_run(
        functools.partial(image_features, mode=mode),
        images,
        strategy="map",
        processes=processes,
        start_method=start_method,
    )
    feature_sets = {
        "raw_pixels_64": np.stack([raw_pixel_features(im) for im in images]),
        "glcm_minmax": np.stack([f["glcm_minmax"] for f in extracted.results]),
        "glcm_direct_cast": np.stack([f["glcm_direct"] for f in extracted.results]),
    }

    fold_rows: list[dict[str, Any]] = []
    leak_check: dict[str, int] = {}
    for scheme in ("grouped", "leaky"):
        for seed in seeds:
            leak_check[f"{scheme}_seed{seed}"] = leaked_groups(
                make_splits(y, groups, seed, n_splits, scheme), groups
            )
            fold_rows += [
                {"features": "none", **row}
                for row in cross_validate(
                    feature_sets["raw_pixels_64"], y, groups, "majority", seed, n_splits, scheme
                )
            ]
            for fs_name, X in feature_sets.items():
                for model in MODEL_NAMES[1:]:
                    fold_rows += [
                        {"features": fs_name, **row}
                        for row in cross_validate(X, y, groups, model, seed, n_splits, scheme)
                    ]

    summary: list[dict[str, Any]] = []
    keys = sorted(
        {(r["cv"], r["features"], r["model"]) for r in fold_rows},
        key=lambda k: (k[0] != "grouped", FEATURE_ORDER.index(k[1]), MODEL_NAMES.index(k[2])),
    )
    for cv, fs, model in keys:
        rows = [r for r in fold_rows if (r["cv"], r["features"], r["model"]) == (cv, fs, model)]
        entry: dict[str, Any] = {"cv": cv, "features": fs, "model": model, "folds": len(rows)}
        for metric in ("accuracy", "macro_f1", "tumour_recall"):
            m, sd = mean_sd([r[metric] for r in rows])
            entry[f"{metric}_mean"] = round(m, 4)
            entry[f"{metric}_sd"] = round(sd, 4)
        summary.append(entry)

    write_csv(out_dir / "ml_cv_folds.csv", fold_rows)
    write_csv(out_dir / "ml_cv_summary.csv", summary)
    counts = io.count_by_class_and_extension(records)
    meta = {
        "what": (
            "stratified group K-fold CV (group = file SHA-256) on all images; "
            "'leaky' rows ignore groups and exist only to measure leakage"
        ),
        "mode": mode,
        "n_images": len(records),
        "n_unique_files": int(len(set(groups.tolist()))),
        "n_duplicate_files": int(len(records) - len(set(groups.tolist()))),
        "class_counts": {c: int(sum(v.values())) for c, v in counts.items()},
        "extension_counts": counts,
        "seeds": list(seeds),
        "n_splits": n_splits,
        "metrics_note": "mean and sd over all folds of all seeds (n_splits x len(seeds) folds)",
        "tumour_label": 1,
        "groups_shared_between_train_and_test": leak_check,
        "feature_extraction": {
            "processes": processes,
            "seconds": round(extracted.seconds, 3),
            "strategy": "Pool.map",
        },
        "feature_sha256": {k: array_sha256(v) for k, v in feature_sets.items()},
        "feature_shapes": {k: list(v.shape) for k, v in feature_sets.items()},
        "machine": machine_info(start_method),
        "started_utc": started,
        "finished_utc": utc_now(),
    }
    write_json(out_dir / "ml_meta.json", meta)
    return {"summary": summary, "meta": meta}


def count_case_sensitive_glob(folder: Path, pattern: str = "*.jpg") -> int:
    """Files a case-sensitive glob would find (as on Linux), on every operating system.

    glob.glob is case-insensitive on Windows, so it is not used for this count.
    """
    return sum(
        1
        for p in Path(folder).iterdir()
        if not p.name.startswith(".") and fnmatch.fnmatchcase(p.name, pattern)
    )


def run_inventory(data_dir: Path, out_dir: Path, manifest: Path | None = None) -> dict[str, Any]:
    records = io.list_images(data_dir)
    hashes = [r.sha256 for r in records]
    counts = io.count_by_class_and_extension(records)
    lowercase_jpg = {c: count_case_sensitive_glob(Path(data_dir) / c) for c in sorted(counts)}
    summary = {
        "n_images": len(records),
        "class_counts": {c: int(sum(v.values())) for c, v in counts.items()},
        "extension_counts": counts,
        "n_unique_files": len(set(hashes)),
        "n_duplicate_files": len(hashes) - len(set(hashes)),
        "duplicate_groups": sum(1 for h in set(hashes) if hashes.count(h) > 1),
        "cross_class_duplicate_groups": sum(
            1 for h in set(hashes) if len({r.class_name for r in records if r.sha256 == h}) > 1
        ),
        "found_by_lowercase_jpg_glob": lowercase_jpg,
    }
    write_json(out_dir / "dataset_summary.json", summary)
    if manifest is not None:
        io.write_manifest(records, manifest)
    return summary
