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


# Stage names as written by earlier runs, shown with the current wording.
STAGE_LABELS = {
    "GLCM (7 images x 24 features)": "GLCM (7 filtered images x 24 features per image)",
}


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
        stage = STAGE_LABELS.get(r["stage"], r["stage"])
        name = f"**{stage}**" if stage == "Total" else stage
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


def _cores_note(machine: dict) -> str:
    if machine.get("physical_cores"):
        return (
            f"{machine['usable_cores']} logical cores, {machine['physical_cores']} physical "
            f"({machine['cpu_model']})"
        )
    return f"{machine['usable_cores']} cores ({machine['cpu_model']})"


def scaling_table(results: Path, suffix: str = "", where: str = "results") -> str:
    rows = read_csv(results / f"scaling_summary{suffix}.csv")
    meta = json.loads((results / f"scaling_meta{suffix}.json").read_text(encoding="utf-8"))
    m = meta["machine"]
    out = [
        f"Task `{meta.get('task', 'filters')}`: {meta['n_images']} images, "
        f"{meta['mode']} mode, {_size_note(meta)}, {meta['repeats']} run(s) each, "
        f"{_cores_note(m)}, start method `{m['start_method']}` "
        f"(`{where}/scaling_summary{suffix}.csv`)",
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


# ------------------------------------------------------------------ Part B

PART_B_DIR = "part_b"
PART_B_MODELS = {"glcm_rf": "GLCM + random forest", "small_cnn": "Small CNN"}
PART_B_EVAL_SETS = {
    "test_as_provided": "Testing split as provided",
    "test_without_flagged": "Testing without cross-split duplicate candidates",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def part_b_counts_table(part_b: Path) -> str:
    counts = _json(part_b / "dataset_counts.json")
    classes = list(next(iter(counts["found"].values())))
    out = [
        f"Images found vs expected: {counts['total_found']} of {counts['total_expected']}, "
        f"differences: {len(counts['differences'])} (`results/part_b/dataset_counts.json`)",
        "",
        "| Split | " + " | ".join(classes) + " | Total |",
        "|---|" + "---:|" * (len(classes) + 1),
    ]
    for split, found in counts["found"].items():
        expected = counts["expected"][split]
        cells = [
            str(found[c]) if found[c] == expected[c] else f"{found[c]} (expected {expected[c]})"
            for c in classes
        ]
        out.append(f"| {split} | " + " | ".join(cells) + f" | {sum(found.values())} |")
    return "\n".join(out)


def part_b_duplicates_table(part_b: Path) -> str:
    dup = _json(part_b / "duplicates.json")
    exact, near = dup["exact"], dup["near"]
    cand = near["candidates"]
    pairs = near["pairs_by_hash_and_threshold"]
    t = near["candidate_max_bits"]
    key = f"pairs_within_{t}_bits"
    scopes = ("within_Training", "within_Testing", "across_splits")
    rows: list[tuple[str, ...]] = [
        (
            "Exact copies (SHA-256): extra copies inside the split",
            str(exact["within_Training"]["extra_copies"]),
            str(exact["within_Testing"]["extra_copies"]),
            "n/a",
        ),
        (
            "Exact copies (SHA-256): Testing files with a copy in Training",
            "n/a",
            "n/a",
            str(exact["across_splits"]["testing_files_with_exact_copy_in_training"]),
        ),
        *(
            (f"{label} pairs within {t} bits", *(str(pairs[s][h][key]) for s in scopes))
            for h, label in (("dhash", "dHash"), ("phash", "pHash"))
        ),
        (
            f"Candidate pairs (dHash and pHash both within {t} bits)",
            *(str(cand[s]["candidate_pairs"]) for s in scopes),
        ),
        (
            "Candidate pairs with different class labels",
            *(str(cand[s]["pairs_with_different_classes"]) for s in scopes),
        ),
        (
            "Testing images with a candidate in Training",
            "n/a",
            "n/a",
            str(cand["across_splits"]["testing_images_with_candidate_in_training"]),
        ),
    ]
    out = [
        "64-bit perceptual hashes, pairs counted by Hamming distance "
        "(`results/part_b/duplicates.json`)",
        "",
        "| Check | Within Training | Within Testing | Across splits |",
        "|---|---:|---:|---:|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _pm(row: dict, metric: str) -> str:
    return f"{_f(row[f'{metric}_mean'], 3)} +/- {_f(row[f'{metric}_sd'], 3)}"


def _find(rows: list[dict[str, str]], model: str, eval_set: str, name: str) -> dict | None:
    for r in rows:
        if (r["model"], r["eval_set"], r["class"]) == (model, eval_set, name):
            return r
    return None


def part_b_models_table(part_b: Path) -> str:
    rows = read_csv(part_b / "per_class_summary.csv")
    info = _json(part_b / "run_info.json")
    seeds = ", ".join(map(str, info["seeds"]))
    out = [
        f"Trained on the Training split, seeds {seeds}, mean +/- sd over seeds "
        "(`results/part_b/per_class_summary.csv`)",
        "",
        "| Model | Test set | Test images | Accuracy | Macro F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for eval_set, eval_label in PART_B_EVAL_SETS.items():
        for model, label in PART_B_MODELS.items():
            acc = _find(rows, model, eval_set, "accuracy")
            macro = _find(rows, model, eval_set, "macro avg")
            if acc and macro:
                out.append(
                    f"| {label} | {eval_label} | {acc['support']} | {_pm(acc, 'f1')} | "
                    f"{_pm(macro, 'f1')} |"
                )
    return "\n".join(out)


def part_b_per_class_table(part_b: Path, eval_set: str = "test_as_provided") -> str:
    rows = [r for r in read_csv(part_b / "per_class_summary.csv") if r["eval_set"] == eval_set]
    out = [
        f"{PART_B_EVAL_SETS[eval_set]}, mean +/- sd over seeds "
        "(`results/part_b/per_class_summary.csv`)",
        "",
        "| Model | Class | Precision | Recall | F1 | Support |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, label in PART_B_MODELS.items():
        for r in rows:
            if r["model"] == model and r["class"] not in ("accuracy", "macro avg"):
                out.append(
                    f"| {label} | {r['class']} | {_pm(r, 'precision')} | {_pm(r, 'recall')} | "
                    f"{_pm(r, 'f1')} | {r['support']} |"
                )
    return "\n".join(out)


PART_B_STEP_LABELS = {
    "count_check": "Count check (list and SHA-256 every file)",
    "duplicate_check": "Duplicate check (perceptual hashes and pair search)",
    "scaling": "Speedup benchmark (all configurations and repeats)",
    "glcm_features": "GLCM features for all images (parallel)",
    "random_forest": "Random forest, fit and predict (all seeds)",
    "cnn": "Small CNN, load, train and predict (all seeds)",
}


def part_b_times_table(part_b: Path) -> str:
    info = _json(part_b / "run_info.json")
    out = [
        "Wall-clock time of each step in one run of the notebook (`results/part_b/run_info.json`)",
        "",
        "| Step | Time (s) |",
        "|---|---:|",
    ]
    for key, label in PART_B_STEP_LABELS.items():
        if key in info["step_seconds"]:
            out.append(f"| {label} | {_f(info['step_seconds'][key], 1)} |")
    return "\n".join(out)


PART_B_TABLES = {
    "part_b_counts": part_b_counts_table,
    "part_b_duplicates": part_b_duplicates_table,
    "part_b_scaling": lambda p: scaling_table(p, where="results/part_b"),
    "part_b_models": part_b_models_table,
    "part_b_per_class": part_b_per_class_table,
    "part_b_times": part_b_times_table,
}


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
    for name, func in PART_B_TABLES.items():
        try:
            rendered[name] = func(results / PART_B_DIR)
        except FileNotFoundError:
            continue
    return rendered
