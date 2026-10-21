"""Draw the README figures from results/ (no numbers are typed in here).

python scripts/make_figures.py [--results results] [--out figures]

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mri_pipeline.plotting import INK, INK_2, MARKERS, SERIES, SURFACE, new_figure, save
from mri_pipeline.report import read_csv

# (file suffix, label, note): the full-mode run was one run on a different VM.
SCALING_SERIES = (
    ("", "fast, native sizes", ""),
    ("_resized_256", "fast, resized 256x256", ""),
    ("_full_resized_256", "full, resized 256x256", "different VM, "),
)


def scaling_figure(results: Path, out: Path) -> Path:
    fig, (ax,) = new_figure(8.0, 4.4)
    max_p = 1
    for k, (suffix, label, note) in enumerate(SCALING_SERIES):
        path = results / f"scaling_summary{suffix}.csv"
        if not path.exists():
            continue
        meta = json.loads((results / f"scaling_meta{suffix}.json").read_text(encoding="utf-8"))
        rows = [r for r in read_csv(path) if r["strategy"] == "map"]
        p = [int(r["processes"]) for r in rows]
        s = [float(r["speedup_mean"]) for r in rows]
        sd = [float(r["speedup_sd"]) for r in rows]
        max_p = max(max_p, *p)
        ax.errorbar(
            p,
            s,
            yerr=sd,
            color=SERIES[k],
            linewidth=2,
            marker=MARKERS[k],
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            capsize=3,
            zorder=3,
        )
        runs = f"{meta['repeats']} run" + ("s" if meta["repeats"] > 1 else "")
        ax.annotate(
            f"{s[-1]:.2f}x  {label}\n{meta['n_images']} images, {note}{runs}",
            (p[-1], s[-1]),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
            va="center",
            fontsize=8,
            color=INK,
        )
    ideal = list(range(1, max_p + 1))
    ax.plot(ideal, ideal, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(max_p - 0.1, max_p + 0.1, "ideal (linear)", color=INK_2, fontsize=8, ha="right")
    ax.set_xticks(ideal)
    ax.set_xlim(0.8, max_p + 1.6)
    ax.set_ylim(0, max_p + 0.5)
    ax.set_xlabel("Worker processes (Pool.map, default chunk size)", color=INK, fontsize=10)
    ax.set_ylabel("Speedup vs sequential loop", color=INK, fontsize=10)
    ax.set_title(
        "Speedup of the filter stage, mean +/- sd (4-core VM)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    return save(fig, out / "scaling_speedup.png")


def profile_figure(results: Path, out: Path) -> Path:
    fig, axes = new_figure(9.0, 3.6, ncols=2)
    for ax, mode in zip(axes, ("full", "fast"), strict=True):
        rows = [r for r in read_csv(results / f"profile_{mode}.csv") if r["stage"] != "Total"]
        rows = [r for r in rows if not r["stage"].startswith("Other")]
        names = [r["stage"].split(" (")[0] for r in rows][::-1]
        share = [float(r["share_pct"]) for r in rows][::-1]
        ax.barh(names, share, color=SERIES[0], height=0.6)
        for y, v in enumerate(share):
            ax.text(v + 1.5, y, f"{v:.1f} %", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Share of sequential run time (%)", color=INK, fontsize=9)
        sig = "1-99" if mode == "full" else "1-9"
        ax.set_title(f"{mode} mode (Hessian sigmas {sig})", color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Where the time goes: cProfile of the sequential pipeline, 5 images",
        color=INK,
        fontsize=10,
        x=0.01,
        ha="left",
    )
    return save(fig, out / "profile_share.png")


def ml_figure(results: Path, out: Path) -> Path:
    rows = [r for r in read_csv(results / "ml_cv_summary.csv") if r["cv"] == "grouped"]
    majority = next(r for r in rows if r["model"] == "majority")
    feature_sets = [
        ("raw_pixels_64", "raw pixels 64x64"),
        ("glcm_minmax", "GLCM, min-max rescale"),
        ("glcm_direct_cast", "GLCM, direct cast (ablation)"),
    ]
    models = [("logreg", "LogReg"), ("svm", "SVM (RBF)"), ("rf", "Random forest")]
    fig, (ax,) = new_figure(7.2, 4.0)
    maj = float(majority["accuracy_mean"])
    ax.axhline(maj, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(2.45, maj + 0.008, f"majority class {maj:.3f}", color=INK_2, fontsize=8, ha="right")
    offsets = (-0.22, 0.0, 0.22)
    for k, (fs, label) in enumerate(feature_sets):
        for m, (model, _) in enumerate(models):
            r = next(x for x in rows if x["features"] == fs and x["model"] == model)
            ax.errorbar(
                m + offsets[k],
                float(r["accuracy_mean"]),
                yerr=float(r["accuracy_sd"]),
                color=SERIES[k],
                marker=MARKERS[k],
                markersize=7,
                capsize=3,
                linewidth=1.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label if m == 0 else None,
            )
    ax.set_xticks(range(len(models)), [name for _, name in models])
    ax.set_ylabel("Accuracy (mean +/- sd over folds)", color=INK, fontsize=10)
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=3, labelcolor=INK)
    ax.set_title(
        "Grouped 5-fold CV x 3 seeds, all images: GLCM vs raw-pixel baselines",
        color=INK,
        fontsize=10,
        loc="left",
    )
    return save(fig, out / "ml_accuracy.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for make in (scaling_figure, profile_figure, ml_figure):
        print(make(args.results, args.out))
    if (args.results / "part_b" / "run_info.json").exists():
        from mri_pipeline.part_b_figures import make_all

        for path in make_all(args.results / "part_b", args.out / "part_b"):
            print(path)


if __name__ == "__main__":
    main()
