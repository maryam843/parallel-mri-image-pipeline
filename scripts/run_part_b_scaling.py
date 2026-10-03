"""Re-run only the Part B speedup benchmark, with the same settings as the notebook.

    MRI_PART_B_DATA_DIR=<data folder> python scripts/run_part_b_scaling.py

Filter stage, 200 Training images (50 per class, seed 0) resized to 256x256,
fast mode, sequential loop and Pool.map on 1, 2, 4, then every even count up
to the number of logical cores, 3 interleaved runs each, one BLAS thread per
worker. Writes scaling_summary.csv, scaling_runs.csv, scaling_meta.json and
parallel_equivalence.json to results/part_b/, the machine state at the start
(free memory, the process names using the most memory) to
scaling_environment.json, and redraws figures/part_b/scaling_speedup.png.
Exit code 1 if the parallel outputs differ from the sequential ones.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mri_pipeline import part_b
from mri_pipeline.experiments import run_scaling, utc_now, write_json
from mri_pipeline.parallel import available_cores
from mri_pipeline.part_b_figures import scaling_figure
from mri_pipeline.report import scaling_table


def process_counts(cores: int) -> list[int]:
    """The notebook's rule: 1, 2, 4 and every even count up to the cores, plus the cores."""
    return sorted({1, 2, 4, *range(6, cores + 1, 2), cores})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results") / "part_b")
    parser.add_argument("--figures", type=Path, default=Path("figures") / "part_b")
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--processes", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)
    root = part_b.data_dir_from_env()
    if root is None:
        print(f"set {part_b.DATA_ENV} to the folder that holds Training/ and Testing/")
        return 1
    counts = args.processes or process_counts(available_cores())
    environment = {
        "started_utc": utc_now(),
        "memory_at_start": part_b.memory_snapshot(),
        "note": "process names only, all processes of one name added up (working set)",
    }
    start = time.perf_counter()
    out = run_scaling(
        root / "Training",
        args.results,
        n_images=args.images,
        process_counts=tuple(counts),
        repeats=args.repeats,
        mode="fast",
        seed=0,
        compare_strategies=False,
        resize=args.resize,
        classes=part_b.class_labels(),
        worker_threads=1,
    )
    environment["benchmark_seconds"] = round(time.perf_counter() - start, 1)
    environment["finished_utc"] = utc_now()
    write_json(args.results / "scaling_environment.json", environment)
    args.figures.mkdir(parents=True, exist_ok=True)
    scaling_figure(args.results, args.figures)
    print(scaling_table(args.results, where="results/part_b"))
    equal = out["meta"]["parallel_equals_sequential"]
    print("parallel outputs identical to sequential:", equal)
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
