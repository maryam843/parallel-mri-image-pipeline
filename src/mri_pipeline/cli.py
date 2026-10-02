"""Command line interface: ``python -m mri_pipeline <command> [options]``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mri_pipeline import io
from mri_pipeline.parallel import available_cores

DEFAULT_OUT = Path("results")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=io.DEFAULT_DATA_DIR,
        help="folder with yes/ and no/ subfolders (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="where CSV/JSON results are written (default: %(default)s)",
    )
    parser.add_argument(
        "--start-method",
        choices=["fork", "spawn", "forkserver"],
        default=None,
        help="multiprocessing start method (default: platform default, "
        "or the MRI_START_METHOD environment variable)",
    )


def _mode(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fast",
        dest="mode",
        action="store_const",
        const="fast",
        help="Hessian sigmas 1-9 (default)",
    )
    group.add_argument(
        "--full",
        dest="mode",
        action="store_const",
        const="full",
        help="Hessian sigmas 1-99 as in the course template (hours on all images)",
    )
    parser.set_defaults(mode="fast")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mri_pipeline",
        description="Parallel MRI filter + GLCM pipeline: profiling, scaling and ML benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory", help="count images by class/extension, find duplicates")
    _common(p)
    p.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="also write a checksum manifest CSV to this path",
    )

    p = sub.add_parser("verify-data", help="check a local dataset copy against the manifest")
    p.add_argument("--data-dir", type=Path, default=io.DEFAULT_DATA_DIR)
    p.add_argument("--manifest", type=Path, default=Path("docs") / "dataset_manifest.csv")

    p = sub.add_parser("profile", help="cProfile the sequential pipeline per filter")
    _common(p)
    _mode(p)
    p.add_argument("--images", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("run-scaling", help="time 1..N processes and pool strategies")
    _common(p)
    _mode(p)
    p.add_argument("--images", type=int, default=40)
    p.add_argument(
        "--processes",
        type=int,
        nargs="+",
        default=None,
        help="process counts (default: 1..number of usable cores)",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-strategies",
        action="store_true",
        help="skip the chunksize / imap_unordered comparison",
    )
    p.add_argument("--tag", default="", help="suffix for output file names")
    p.add_argument(
        "--resize",
        type=int,
        default=None,
        help="resize every image to N x N before timing (default: native sizes)",
    )
    p.add_argument(
        "--task",
        choices=["filters", "features"],
        default="filters",
        help="filters: return the 7 filtered arrays to the parent; "
        "features: filters + GLCM in the worker, return 168 floats",
    )

    p = sub.add_parser("run-ml", help="features for all images + grouped CV vs baselines")
    _common(p)
    _mode(p)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--processes", type=int, default=None)

    p = sub.add_parser("make-synthetic", help="write a small synthetic dataset (for demos and CI)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Imported here so that ``--help`` stays fast.
    from mri_pipeline import experiments

    if args.command == "inventory":
        summary = experiments.run_inventory(args.data_dir, args.out_dir, args.write_manifest)
        print(json.dumps(summary, indent=2))
    elif args.command == "verify-data":
        diff = io.verify_manifest(args.data_dir, args.manifest)
        for key, items in diff.items():
            print(f"{key}: {len(items)}")
            for item in items[:20]:
                print(f"  {item}")
        return 0 if not any(diff.values()) else 1
    elif args.command == "profile":
        rows = experiments.run_profile(
            args.data_dir, args.out_dir, args.images, args.mode, args.seed
        )
        for r in rows:
            print(f"{r['stage']:32s} {r['total_s']:9.2f} s  {r['share_pct']:5.1f} %")
    elif args.command == "run-scaling":
        processes = tuple(args.processes or range(1, available_cores() + 1))
        out = experiments.run_scaling(
            args.data_dir,
            args.out_dir,
            args.images,
            processes,
            args.repeats,
            args.mode,
            args.seed,
            not args.no_strategies,
            args.start_method,
            args.tag,
            args.task,
            args.resize,
        )
        for r in out["scaling"] + out["strategies"]:
            print(
                f"{r['config']:22s} p={r['processes']}  {r['mean_s']:8.2f} +/- {r['sd_s']:.2f} s"
                f"  speedup {r['speedup_mean']:.2f}x  eff {r['efficiency']:.2f}"
            )
        print("parallel == sequential:", out["meta"]["parallel_equals_sequential"])
    elif args.command == "run-ml":
        out = experiments.run_ml(
            args.data_dir,
            args.out_dir,
            args.mode,
            tuple(args.seeds),
            args.folds,
            args.processes,
            args.start_method,
        )
        for r in out["summary"]:
            print(
                f"{r['cv']:8s} {r['features']:17s} {r['model']:9s} acc {r['accuracy_mean']:.3f}"
                f" +/- {r['accuracy_sd']:.3f}  f1 {r['macro_f1_mean']:.3f}"
                f"  recall {r['tumour_recall_mean']:.3f}"
            )
    elif args.command == "make-synthetic":
        from mri_pipeline.synthetic import make_dataset

        print(json.dumps(make_dataset(args.out, args.per_class, args.seed)))
    return 0
