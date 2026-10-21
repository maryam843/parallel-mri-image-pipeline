"""Part B figures, drawn only from the files in results/part_b/.

Charts only: speedup, confusion matrices, per-class F1 and the CNN training
loss. No MRI image is drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mri_pipeline.plotting import GRID, INK, INK_2, SERIES, SURFACE, new_figure, save
from mri_pipeline.report import read_csv

MODELS = {"glcm_rf": "GLCM + random forest", "small_cnn": "Small CNN"}


def _new(width: float, height: float, ncols: int = 1):
    return new_figure(width, height, ncols, labelsize=8, grid=False)


def scaling_figure(part_b: Path, out: Path) -> Path:
    rows = [r for r in read_csv(part_b / "scaling_summary.csv") if r["strategy"] == "map"]
    meta = json.loads((part_b / "scaling_meta.json").read_text(encoding="utf-8"))
    p = [int(r["processes"]) for r in rows]
    s = [float(r["speedup_mean"]) for r in rows]
    sd = [float(r["speedup_sd"]) for r in rows]
    fig, (ax,) = _new(7.0, 4.0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(p, p, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), label="ideal (linear)")
    ax.errorbar(
        p, s, yerr=sd, color=SERIES[0], marker="o", linewidth=2, capsize=3, label="Pool.map"
    )
    for x, y in zip(p, s, strict=True):
        ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=7)
    m = meta["machine"]
    ax.set_xticks(p)
    ax.set_xlabel("Worker processes", color=INK, fontsize=9)
    ax.set_ylabel("Speedup vs sequential loop", color=INK, fontsize=9)
    ax.set_title(
        f"Filter stage, {meta['n_images']} images at {meta['resize']}x{meta['resize']}, "
        f"{meta['repeats']} runs each, mean +/- sd\n{m['cpu_model']}, "
        f"{m['usable_cores']} logical cores",
        color=INK,
        fontsize=9,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return save(fig, out / "scaling_speedup.png")


def confusion_figure(part_b: Path, out: Path, seed: int = 0) -> Path:
    data = json.loads((part_b / "confusion_matrices.json").read_text(encoding="utf-8"))
    classes = data["classes"]
    fig, axes = _new(9.0, 4.2, ncols=len(MODELS))
    for ax, (model, label) in zip(axes, MODELS.items(), strict=True):
        matrix = np.array(data["matrices"][model]["test_as_provided"][str(seed)])
        ax.imshow(matrix, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                value = matrix[i, j]
                colour = SURFACE if value > matrix.max() / 2 else INK
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=colour)
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted class", color=INK, fontsize=9)
        ax.set_ylabel("True class", color=INK, fontsize=9)
        ax.set_title(f"{label}, seed {seed}", color=INK, fontsize=9, loc="left")
    fig.suptitle(
        "Confusion matrices on the Testing split as provided",
        color=INK,
        fontsize=10,
        x=0.01,
        ha="left",
    )
    return save(fig, out / "confusion_matrices.png")


def per_class_f1_figure(part_b: Path, out: Path) -> Path:
    rows = [
        r
        for r in read_csv(part_b / "per_class_summary.csv")
        if r["eval_set"] == "test_as_provided" and r["class"] not in ("accuracy", "macro avg")
    ]
    classes = list(dict.fromkeys(r["class"] for r in rows))
    fig, (ax,) = _new(7.0, 3.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    width = 0.38
    for k, (model, label) in enumerate(MODELS.items()):
        f1 = [float(next(r for r in rows if (r["model"], r["class"]) == (model, c))["f1_mean"])
              for c in classes]  # fmt: skip
        sd = [float(next(r for r in rows if (r["model"], r["class"]) == (model, c))["f1_sd"])
              for c in classes]  # fmt: skip
        x = np.arange(len(classes)) + (k - 0.5) * width
        ax.bar(x, f1, width, yerr=sd, color=SERIES[k], capsize=3, label=label)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1, mean +/- sd over seeds", color=INK, fontsize=9)
    ax.set_title("Per-class F1 on the Testing split as provided", color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return save(fig, out / "per_class_f1.png")


def cnn_loss_figure(part_b: Path, out: Path) -> Path:
    rows = read_csv(part_b / "cnn_history.csv")
    fig, (ax,) = _new(6.0, 3.6)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for k, seed in enumerate(dict.fromkeys(r["seed"] for r in rows)):
        mine = [r for r in rows if r["seed"] == seed]
        ax.plot(
            [int(r["epoch"]) for r in mine],
            [float(r["train_loss"]) for r in mine],
            color=SERIES[k % len(SERIES)],
            marker="o",
            markersize=3,
            label=f"seed {seed}",
        )
    ax.set_xlabel("Epoch", color=INK, fontsize=9)
    ax.set_ylabel("Training loss (cross-entropy)", color=INK, fontsize=9)
    ax.set_title("Small CNN: training loss per epoch", color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    return save(fig, out / "cnn_training_loss.png")


def make_all(part_b: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [
        make(part_b, out)
        for make in (scaling_figure, confusion_figure, per_class_f1_figure, cnn_loss_figure)
    ]
