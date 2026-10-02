"""Markdown tables for the README, built only from files in results/.

``scripts/check_readme_claims.py`` regenerates these tables and fails if the
README copy differs, so the README cannot drift away from the results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value: str | float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def profile_table(results: Path, mode: str) -> str:
    rows = read_csv(results / f"profile_{mode}.csv")
    meta = json.loads((results / f"profile_{mode}_meta.json").read_text(encoding="utf-8"))
    out = [
        f"Sequential pipeline, {meta['n_images']} images, {mode} mode, cProfile cumulative time "
        f"(`results/profile_{mode}.csv`)",
        "",
        "| Stage | Total (s) | Per image (s) | Share |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        name = f"**{r['stage']}**" if r["stage"] == "Total" else r["stage"]
        out.append(
            f"| {name} | {_f(r['total_s'], 2)} | {_f(r['per_image_s'], 2)} | "
            f"{_f(r['share_pct'], 1)} % |"
        )
    return "\n".join(out)


def _size_note(meta: dict) -> str:
    size = f"resized to {meta['resize']}x{meta['resize']}" if meta.get("resize") else "native sizes"
    share = meta.get("largest_image_pixel_share")
    if share is None:
        return size
    return f"{size}, largest image = {100 * share:.1f} % of all pixels"


def mean_sd(mean: str | float, sd: str | float, runs: str | int, digits: int) -> str:
    """'mean +/- sd' for repeated runs; a single run has no spread, so say so."""
    if int(runs) < 2:
        return f"{_f(mean, digits)} (1 run)"
    return f"{_f(mean, digits)} +/- {_f(sd, digits)}"


def scaling_table(results: Path, suffix: str = "") -> str:
    rows = read_csv(results / f"scaling_summary{suffix}.csv")
    meta = json.loads((results / f"scaling_meta{suffix}.json").read_text(encoding="utf-8"))
    m = meta["machine"]
    out = [
        f"Task `{meta.get('task', 'filters')}`: {meta['n_images']} images, "
        f"{meta['mode']} mode, {_size_note(meta)}, {meta['repeats']} run(s) each, "
        f"{m['usable_cores']} cores ({m['cpu_model']}), start method `{m['start_method']}` "
        f"(`results/scaling_summary{suffix}.csv`)",
        "",
        "| Configuration | Processes | Time (s), mean +/- sd | Speedup | Efficiency |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        label = "Sequential loop (no pool)" if r["strategy"] == "sequential" else "Pool.map"
        if r["strategy"] == "sequential":
            speedup = f"{_f(r['speedup_mean'], 2)}x"  # the baseline, 1.00x by definition
        elif int(r["runs"]) < 2:
            speedup = f"{_f(r['speedup_mean'], 2)}x (1 run)"
        else:
            speedup = f"{_f(r['speedup_mean'], 2)}x +/- {_f(r['speedup_sd'], 2)}"
        out.append(
            f"| {label} | {r['processes']} | {mean_sd(r['mean_s'], r['sd_s'], r['runs'], 2)} | "
            f"{speedup} | {_f(r['efficiency'], 2)} |"
        )
    return "\n".join(out)


STRATEGY_LABELS = {
    "map_auto": "Pool.map, chunksize auto",
    "map_c1": "Pool.map, chunksize 1",
    "imap_unordered_c1": "imap_unordered, chunksize 1",
    "imap_unordered_auto": "imap_unordered, chunksize auto",
}


def strategy_table(results: Path, suffix: str = "") -> str:
    rows = read_csv(results / f"strategy_summary{suffix}.csv")
    out = [
        "| Strategy | Processes | Chunk size used | Time (s), mean +/- sd | Speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        out.append(
            f"| {STRATEGY_LABELS.get(r['config'], r['config'])} | {r['processes']} | "
            f"{r['effective_chunksize']} | "
            f"{mean_sd(r['mean_s'], r['sd_s'], r['runs'], 2)} | {_f(r['speedup_mean'], 2)}x |"
        )
    return "\n".join(out)


FEATURE_LABELS = {
    "none": "none",
    "raw_pixels_64": "raw pixels 64x64",
    "glcm_minmax": "GLCM, min-max rescale",
    "glcm_direct_cast": "GLCM, direct uint8 cast (ablation)",
}
MODEL_LABELS = {
    "majority": "Majority class",
    "logreg": "LogReg",
    "svm": "SVM (RBF)",
    "rf": "Random forest",
}


def ml_table(results: Path, cv: str = "grouped") -> str:
    rows = [r for r in read_csv(results / "ml_cv_summary.csv") if r["cv"] == cv]
    meta = json.loads((results / "ml_meta.json").read_text(encoding="utf-8"))
    out = [
        f"{meta['n_images']} images, {meta['n_splits']}-fold stratified CV grouped by file hash, "
        f"seeds {', '.join(map(str, meta['seeds']))}, mean +/- sd over {rows[0]['folds']} folds "
        "(`results/ml_cv_summary.csv`)",
        "",
        "| Features | Model | Accuracy | Macro F1 | Tumour recall |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        out.append(
            f"| {FEATURE_LABELS[r['features']]} | {MODEL_LABELS[r['model']]} | "
            f"{_f(r['accuracy_mean'], 3)} +/- {_f(r['accuracy_sd'], 3)} | "
            f"{_f(r['macro_f1_mean'], 3)} +/- {_f(r['macro_f1_sd'], 3)} | "
            f"{_f(r['tumour_recall_mean'], 3)} +/- {_f(r['tumour_recall_sd'], 3)} |"
        )
    return "\n".join(out)


def leakage_table(results: Path) -> str:
    rows = read_csv(results / "ml_cv_summary.csv")
    index = {(r["cv"], r["features"], r["model"]): r for r in rows}
    out = [
        "| Features | Model | Accuracy, grouped CV | Accuracy, leaky CV | Difference |",
        "|---|---|---:|---:|---:|",
    ]
    for fs in ("raw_pixels_64", "glcm_minmax", "glcm_direct_cast"):
        for model in ("logreg", "svm", "rf"):
            g = float(index[("grouped", fs, model)]["accuracy_mean"])
            leak = float(index[("leaky", fs, model)]["accuracy_mean"])
            out.append(
                f"| {FEATURE_LABELS[fs]} | {MODEL_LABELS[model]} | {g:.3f} | {leak:.3f} | "
                f"{leak - g:+.3f} |"
            )
    return "\n".join(out)


TABLES = {
    "profile_full": lambda r: profile_table(r, "full"),
    "profile_fast": lambda r: profile_table(r, "fast"),
    "scaling": scaling_table,
    "strategies": strategy_table,
    "ml": ml_table,
    "leakage": leakage_table,
}


def render_all(results: Path) -> dict[str, str]:
    """Every README table, keyed by the name used in its BEGIN/END markers."""
    rendered = {}
    for name, func in TABLES.items():
        try:
            rendered[name] = func(results)
        except FileNotFoundError:
            continue
    for tag in ("full_subset", "full_resized_256", "features_task", "resized_256"):
        if (results / f"scaling_summary_{tag}.csv").exists():
            rendered[f"scaling_{tag}"] = scaling_table(results, f"_{tag}")
    return rendered
