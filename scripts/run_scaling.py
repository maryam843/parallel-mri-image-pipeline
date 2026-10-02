"""Same as `python -m mri_pipeline run-scaling`; all options are passed through.

python scripts/run_scaling.py --help
"""

import sys

from mri_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run-scaling", *sys.argv[1:]]))
