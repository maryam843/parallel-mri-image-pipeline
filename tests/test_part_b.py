"""Part B helpers: dataset checks, perceptual hashes, metrics, thread limit, tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from mri_pipeline import io, part_b, report
from mri_pipeline.experiments import run_scaling
from mri_pipeline.parallel import THREAD_VARIABLES, run, worker_thread_limit
from mri_pipeline.synthetic import make_split_dataset, synthetic_image


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("part_b_data")
    make_split_dataset(out, per_class={"Training": 4, "Testing": 2}, seed=3)
    return out


@pytest.fixture(scope="module")
def records(split_dir: Path) -> dict[str, list[io.ImageRecord]]:
    return part_b.list_split_records(split_dir)


def test_licence_and_data_dir_from_environment():
    assert part_b.licence_confirmed({part_b.LICENCE_ENV: " cc-by-4.0 "})
    assert not part_b.licence_confirmed({part_b.LICENCE_ENV: "yes"})
    assert not part_b.licence_confirmed({})
    assert part_b.data_dir_from_env({part_b.DATA_ENV: "some/folder"}) == Path("some/folder")
    assert part_b.data_dir_from_env({part_b.DATA_ENV: "  "}) is None


def test_count_check_lists_every_difference(records):
    counts = part_b.count_check(records)
    assert counts["found"]["Training"] == dict.fromkeys(part_b.CLASSES, 4)
    assert counts["found"]["Testing"]["glioma"] == 4  # 2 + the exact and the near copy
    assert counts["total_found"] == 4 * 4 + 2 * 4 + 2
    assert counts["total_expected"] == 7200
    assert not counts["matches_expected"]
    assert len(counts["differences"]) == 8
    assert counts["extensions"] == {".jpg": 25, ".png": 1}
    exact = part_b.count_check(records, {"Training": 4, "Testing": 2})
    assert [d["class"] for d in exact["differences"]] == ["glioma"]


def test_hashes_are_stable_and_separate_different_images(rng):
    image = synthetic_image(rng, (80, 80), tumour=True)
    brighter = np.clip(image.astype(int) + 3, 0, 255).astype(np.uint8)
    other = synthetic_image(np.random.default_rng(99), (80, 80), tumour=False).T.copy()
    for fn in (part_b.dhash, part_b.phash):
        h = part_b.as_hash_array([fn(image), fn(brighter), fn(other)])
        assert 0 <= h[0] < 2**64
        assert part_b.hamming(h[0], h[0]) == 0
        assert part_b.hamming(h[0], h[1]) <= 4
    assert (
        part_b.hamming(
            part_b.as_hash_array([part_b.dhash(image)]), part_b.as_hash_array([part_b.dhash(other)])
        )[0]
        > 4
    )


def test_close_pairs_within_and_across():
    a = part_b.as_hash_array([0b0000, 0b0001, 0b1111_0000])
    within = part_b.close_pairs(a, max_bits=1)
    assert within.tolist() == [[0, 1, 1]]
    b = part_b.as_hash_array([0b1111_0001])
    assert part_b.close_pairs(a, b, max_bits=1).tolist() == [[2, 0, 1]]
    assert part_b.close_pairs(a, b, max_bits=0).shape == (0, 3)


def test_groups_are_connected_components():
    groups = part_b._groups(6, [(0, 1), (1, 2), (4, 5)])
    assert sorted(sorted(g) for g in groups) == [[0, 1, 2], [4, 5]]


def test_duplicate_checks_find_the_planted_copies(records):
    exact = part_b.exact_duplicates(records)
    assert exact["across_splits"]["testing_files_with_exact_copy_in_training"] == 1
    assert exact["within_Training"]["extra_copies"] == 0
    hashes = {s: [part_b.image_hashes(r.path) for r in records[s]] for s in part_b.SPLITS}
    assert hashes["Training"][0]["height"] == 64
    summary, candidates = part_b.near_duplicates(records, hashes)
    across = summary["candidates"]["across_splits"]
    assert across["candidate_pairs"] >= 2
    assert across["pairs_that_are_exact_copies"] == 1
    flagged = part_b.flagged_testing_files(candidates)
    assert {"glioma/Te-copy_000.jpg", "glioma/Te-near_000.png"} <= flagged
    assert all(c["dhash_bits"] <= 4 and c["phash_bits"] <= 4 for c in candidates)
    assert summary["pairs_by_hash_and_threshold"]["across_splits"]["phash"]["pairs_within_0_bits"]
    assert "candidate_groups" in summary["candidates"]["within_Training"]


def test_glcm_from_path_returns_168_features(records):
    vector = part_b.glcm_from_path(records["Training"][0].path, size=32)
    assert vector.shape == (168,)
    assert np.isfinite(vector).all()


def test_per_class_rows_confusion_and_summary():
    y_true = [0, 0, 1, 1, 2, 3]
    y_pred = [0, 1, 1, 1, 2, 2]
    rows = part_b.per_class_rows(y_true, y_pred)
    by_class = {r["class"]: r for r in rows}
    assert by_class["glioma"]["precision"] == 1.0 and by_class["glioma"]["recall"] == 0.5
    assert by_class["pituitary"]["f1"] == 0.0  # never predicted: zero, not an error
    assert by_class["accuracy"]["f1"] == pytest.approx(4 / 6)
    assert by_class["macro avg"]["support"] == 6
    assert part_b.confusion(y_true, y_pred) == [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
    ]
    runs = [
        {"model": "m", "eval_set": "e", "seed": s, "class": "glioma", "precision": p,
         "recall": 1.0, "f1": 0.5, "support": 2}
        for s, p in ((0, 0.5), (1, 0.7))
    ]  # fmt: skip
    (summary,) = part_b.summarise_runs(runs)
    assert summary["precision_mean"] == 0.6
    assert summary["precision_sd"] == pytest.approx(0.1414, abs=1e-4)
    assert summary["seeds"] == 2
    assert part_b.summarise_runs(runs[:1])[0]["precision_sd"] == 0.0


def test_memory_snapshot_has_numbers():
    snap = part_b.memory_snapshot()
    assert set(snap) == {"total_gb", "available_gb", "free_commit_gb"}
    assert snap["total_gb"] is None or snap["total_gb"] > 0


def test_worker_thread_limit_sets_and_restores(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with worker_thread_limit(1):
        assert all(os.environ[name] == "1" for name in THREAD_VARIABLES)
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "OPENBLAS_NUM_THREADS" not in os.environ
    with worker_thread_limit(None):
        assert os.environ["OMP_NUM_THREADS"] == "7"


def test_pool_with_one_worker_thread_matches_sequential(records):
    paths = [r.path for r in records["Training"][:4]]
    sequential = run(part_b.image_hashes, paths, strategy="sequential")
    pooled = run(part_b.image_hashes, paths, processes=2, worker_threads=1)
    assert pooled == sequential


def test_run_scaling_on_four_classes(split_dir: Path, tmp_path: Path):
    out = run_scaling(
        split_dir / "Training",
        tmp_path,
        n_images=4,
        process_counts=(1, 2),
        repeats=1,
        compare_strategies=False,
        resize=32,
        classes=part_b.class_labels(),
        worker_threads=1,
    )
    meta = out["meta"]
    assert meta["parallel_equals_sequential"]
    assert meta["worker_threads"] == 1
    assert {name.split("/")[0] for name in meta["images"]} == set(part_b.CLASSES)
    assert "physical_cores" in meta["machine"]
    table = report.scaling_table(tmp_path, where="results/part_b")
    assert "`results/part_b/scaling_summary.csv`" in table


# ------------------------------------------------------------ README tables


def write_part_b_results(folder: Path) -> Path:
    """Minimal results/part_b files in the format the notebook writes."""
    folder.mkdir(parents=True, exist_ok=True)
    counts = {
        "found": {"Training": {"a": 2, "b": 3}, "Testing": {"a": 1, "b": 1}},
        "expected": {"Training": {"a": 2, "b": 2}, "Testing": {"a": 1, "b": 1}},
        "total_found": 7,
        "total_expected": 6,
        "differences": [{"split": "Training", "class": "b", "found": 3, "expected": 2}],
    }
    (folder / "dataset_counts.json").write_text(json.dumps(counts), encoding="utf-8")
    scope = {"candidate_pairs": 1, "pairs_with_different_classes": 0}
    pairs = {h: {"pairs_within_4_bits": 2} for h in ("dhash", "phash")}
    dup = {
        "exact": {
            "within_Training": {"extra_copies": 5},
            "within_Testing": {"extra_copies": 1},
            "across_splits": {"testing_files_with_exact_copy_in_training": 0},
        },
        "near": {
            "candidate_max_bits": 4,
            "pairs_by_hash_and_threshold": dict.fromkeys(
                ("within_Training", "within_Testing", "across_splits"), pairs
            ),
            "candidates": {
                "within_Training": scope,
                "within_Testing": scope,
                "across_splits": {**scope, "testing_images_with_candidate_in_training": 1},
            },
        },
    }
    (folder / "duplicates.json").write_text(json.dumps(dup), encoding="utf-8")
    header = (
        "model,eval_set,class,precision_mean,precision_sd,recall_mean,recall_sd,"
        "f1_mean,f1_sd,support,seeds\n"
    )
    lines = []
    for model in ("glcm_rf", "small_cnn"):
        for name in ("glioma", "macro avg", "accuracy"):
            lines.append(f"{model},test_as_provided,{name},0.9,0.01,0.8,0.02,0.85,0.015,10,3\n")
    (folder / "per_class_summary.csv").write_text(header + "".join(lines), encoding="utf-8")
    info = {"seeds": [0, 1, 2], "step_seconds": {"count_check": 1.234, "cnn": 99.0}}
    (folder / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    return folder


def test_part_b_tables_render(tmp_path: Path):
    folder = write_part_b_results(tmp_path / "part_b")
    counts = report.part_b_counts_table(folder)
    assert "| Training | 2 | 3 (expected 2) | 5 |" in counts
    assert "7 of 6, differences: 1" in counts
    dup = report.part_b_duplicates_table(folder)
    assert "| Exact copies (SHA-256): extra copies inside the split | 5 | 1 | n/a |" in dup
    assert "| dHash pairs within 4 bits | 2 | 2 | 2 |" in dup
    models = report.part_b_models_table(folder)
    assert "| Small CNN | Testing split as provided | 10 | 0.850 +/- 0.015 |" in models
    assert "seeds 0, 1, 2" in models
    assert "without cross-split" not in models  # that evaluation set is absent here
    per_class = report.part_b_per_class_table(folder)
    assert "| GLCM + random forest | glioma | 0.900 +/- 0.010 |" in per_class
    assert "macro avg" not in per_class
    times = report.part_b_times_table(folder)
    assert "| Count check (list and SHA-256 every file) | 1.2 |" in times
    assert "Random forest" not in times
    rendered = report.render_all(tmp_path)
    assert {"part_b_counts", "part_b_duplicates", "part_b_models"} <= set(rendered)
    assert "part_b_scaling" not in rendered  # no scaling files written


def test_part_b_figures_from_results(tmp_path: Path, split_dir: Path):
    from mri_pipeline import part_b_figures

    folder = write_part_b_results(tmp_path / "part_b")
    run_scaling(
        split_dir / "Training",
        folder,
        n_images=4,
        process_counts=(1, 2),
        repeats=2,
        compare_strategies=False,
        resize=32,
        classes=part_b.class_labels(),
    )
    rows = [
        {"model": m, "eval_set": "test_as_provided", "seed": 0, **row}
        for m in ("glcm_rf", "small_cnn")
        for row in part_b.per_class_rows([0, 1, 2, 3], [0, 1, 2, 2])
    ]
    from mri_pipeline.experiments import write_csv, write_json

    write_csv(folder / "per_class_summary.csv", part_b.summarise_runs(rows))
    matrix = part_b.confusion([0, 1, 2, 3], [0, 1, 2, 2])
    write_json(
        folder / "confusion_matrices.json",
        {
            "classes": list(part_b.CLASSES),
            "matrices": {m: {"test_as_provided": {"0": matrix}} for m in ("glcm_rf", "small_cnn")},
        },
    )
    write_csv(
        folder / "cnn_history.csv",
        [{"seed": s, "epoch": e, "train_loss": 1.0 / e} for s in (0, 1) for e in (1, 2)],
    )
    paths = part_b_figures.make_all(folder, tmp_path / "figures")
    assert [p.name for p in paths] == [
        "scaling_speedup.png",
        "confusion_matrices.png",
        "per_class_f1.png",
        "cnn_training_loss.png",
    ]
    assert all(p.stat().st_size > 1000 for p in paths)


def test_pixel_check_of_candidates(records, split_dir: Path):
    hashes = {s: [part_b.image_hashes(r.path) for r in records[s]] for s in part_b.SPLITS}
    _, candidates = part_b.near_duplicates(records, hashes)
    summary, rows = part_b.check_candidates(split_dir, candidates, size=32, processes=2)
    copy = next(r for r in rows if r["file_a"] == "Te-copy_000.jpg")
    assert copy["pearson_r"] == pytest.approx(1.0) and copy["mean_abs_diff"] == 0.0
    near = next(r for r in rows if r["file_a"] == "Te-near_000.png")
    assert near["pearson_r"] > 0.99 and 0 < near["mean_abs_diff"] < 5
    assert summary["across_splits"]["pairs"] == len(
        [c for c in candidates if c["scope"] == "across_splits"]
    )
    assert summary["testing_images_with_training_pair_r_at_least_0.99"] >= 2
    assert part_b.check_candidates(split_dir, [])[0]["across_splits"] == {"pairs": 0}


def test_pixel_similarity_of_flat_images(tmp_path: Path):
    from mri_pipeline.synthetic import write_image

    flat = np.full((8, 8), 7, np.uint8)
    write_image(tmp_path / "a.png", flat)
    write_image(tmp_path / "b.png", flat)
    write_image(tmp_path / "c.png", flat + 1)
    assert part_b.pixel_similarity(tmp_path / "a.png", tmp_path / "b.png", 8)["pearson_r"] == 1.0
    other = part_b.pixel_similarity(tmp_path / "a.png", tmp_path / "c.png", 8)
    assert other == {"pearson_r": 0.0, "mean_abs_diff": 1.0}
