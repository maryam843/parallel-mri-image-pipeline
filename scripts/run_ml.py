"""Same as `python -m mri_pipeline run-ml`; all options are passed through.

python scripts/run_ml.py --help
"""

import sys

from mri_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run-ml", *sys.argv[1:]]))
