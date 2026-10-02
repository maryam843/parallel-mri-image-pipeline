# Design decisions

One short entry per choice: what I chose, what else I could have done, why, and what it costs.

## Parallelise per image, not per filter
- **Choice:** one task = one image with all six filters (`Pool.map` over images).
- **Alternatives:** one task per filter per image; threads.
- **Reason:** images are independent, and profiling shows the Hessian takes most of the time for every image. Per-filter tasks would give one long Hessian task and five tiny ones per image, so workers would wait. Threads would not help much because parts of the filter code hold the GIL.
- **Trade-off:** every filtered image is pickled back to the parent. That is kept on purpose in the main scaling benchmark because it is the simplest design; a separate run measures a worker that returns only the features.

## Profile before parallelising
- **Choice:** cProfile on 5 images before any tuning.
- **Alternatives:** guess, or time only the whole run.
- **Reason:** it shows the Hessian dominates (sigmas 1-99), which explains why full mode takes over an hour on all images and why image-level parallelism is enough.
- **Trade-off:** cProfile adds a little overhead; the filters are mostly compiled code, so the shares are still meaningful.

## Fast mode (Hessian sigmas 1-9) as the default
- **Choice:** `--fast` by default, `--full` keeps the course filter setting, sigmas 1-99.
- **Alternatives:** always run full mode.
- **Reason:** full mode on all images takes hours on 4 cores; fast mode lets anyone reproduce every table in minutes.
- **Trade-off:** fast-mode features and timings are not the same as full mode. The README labels every number with its mode, and the full-mode run is a labelled subset.

## Speedup is measured against a plain loop, including pool start-up
- **Choice:** baseline = sequential loop with no pool; each parallel time includes creating the pool and returning the results.
- **Alternatives:** compare with `Pool(1)`, or time only the work inside workers.
- **Reason:** that is the time a user actually waits, and it includes every cost of using a pool.
- **Trade-off:** small inputs look worse because start-up cost is included.

## Interleaved repeats
- **Choice:** 3 repeats, run as rounds (every configuration once, then again).
- **Alternatives:** all repeats of one configuration back to back.
- **Reason:** if the machine slows down over time, every configuration is affected equally.
- **Trade-off:** none worth noting; it takes the same time.

## Chunk size and imap_unordered compared, not assumed
- **Choice:** measure `Pool.map` (auto chunk size and 1) and `imap_unordered` (1 and auto) on the same subset.
- **Alternatives:** pick one from the documentation.
- **Reason:** images have different sizes, so small chunks might balance load better, while larger chunks mean fewer messages. Measuring settles it for this workload.
- **Trade-off:** results are specific to this machine and subset.

## Measure load imbalance instead of hiding it
- **Choice:** keep the native image sizes for the main scaling table, and add two extra runs: the same images resized to 256x256 (`--resize 256`), and a worker that returns only the 168 GLCM values (`--task features`). The metadata records how many of the pixels belong to the largest image.
- **Alternatives:** resize everything and report only the nicer curve.
- **Reason:** the image sizes in this dataset vary a lot (in the 16-image full-mode subset one image has almost half of all pixels). A pool cannot finish before its biggest task, so this caps the speedup no matter how many processes there are. Showing both runs separates "the pool is slow" from "the work cannot be split evenly".
- **Trade-off:** more tables to read; resized images give different filter outputs, so resizing is a benchmark option only, not part of the ML pipeline.

## Class-balanced 40-image subset of unique files for scaling
- **Choice:** 20 "yes" + 20 "no", first copy of each file hash, seed 0; the file list is saved in `results/scaling_meta.json`.
- **Alternatives:** all 253 images; the first 40 files.
- **Reason:** 40 images x 9 configurations x 3 repeats fits in minutes while still giving each of 4 workers about 10 images.
- **Trade-off:** absolute times are for 40 images, not the full set.

## Duplicate-grouped stratified 5-fold CV, 3 seeds
- **Choice:** `StratifiedGroupKFold` with the file's SHA-256 as the group; seeds 0, 1, 2; report mean +/- sd over the 15 folds.
- **Alternatives:** a single train/test split; plain `StratifiedKFold`.
- **Reason:** 25 files are exact copies, and a random split can put a copy on each side. A single 43-image test set is too small to compare models.
- **Trade-off:** the sd across folds mixes fold and seed variation; it is a spread, not a confidence interval.

## "Leaky" CV kept only as a measurement
- **Choice:** also run plain `StratifiedKFold` and report the difference.
- **Alternatives:** leave it out.
- **Reason:** it shows how much the duplicates would have flattered the scores.
- **Trade-off:** more rows in the results; the README table uses grouped CV only.

## Baselines: majority class and raw 64x64 pixels
- **Choice:** the same three models on raw pixels, plus a majority-class dummy.
- **Alternatives:** compare GLCM models only with each other.
- **Reason:** without a trivial baseline there is no way to tell if the filters + GLCM add anything.
- **Trade-off:** none.

## Scaling inside the model pipeline
- **Choice:** `StandardScaler` inside an sklearn `Pipeline` for LogReg and SVM.
- **Alternatives:** scale once before CV.
- **Reason:** the scaler is fit on the training fold only, so no test information leaks.
- **Trade-off:** none.

## Min-max rescale per filter output
- **Choice:** each filter output is scaled from its own min and max to 0-255 before the GLCM.
- **Alternatives:** `skimage.util.img_as_ubyte`; a fixed range per filter.
- **Reason:** `img_as_ubyte` rejects float values outside -1..1 (entropy goes to about 4). A fixed range needs knowledge of each filter's range. Min-max never wraps and keeps the order of values.
- **Trade-off:** contrast is relative to each image, so absolute brightness information is lost.

## Rescale ablation: direct uint8 cast vs per-filter min-max
- **Choice:** `rescale_direct_cast` stays in the code and is run as a separate feature set.
- **Alternatives:** test only the min-max version.
- **Reason:** the choice of rescale should be shown with numbers, not only argued.
- **Trade-off:** one more feature set to explain.

## Gabor filter left as in the course code
- **Choice:** keep `gabor(image, frequency=0.9)[1]` on the uint8 image.
- **Alternatives:** convert to float first.
- **Reason:** I found that it returns uint8, so negative responses wrap. I kept the course filter settings unchanged and noted it as a limitation instead.
- **Trade-off:** one feature group is computed on a wrapped response.

## Worker functions in `src/`, not in notebooks
- **Choice:** every function sent to a pool is a module-level function (or `functools.partial` of one) in an importable module; scripts use `if __name__ == "__main__":`.
- **Alternatives:** define workers in the notebook.
- **Reason:** with the "spawn" start method (Windows, macOS) workers import the function by name. A function that only exists in a notebook cannot be found.
- **Trade-off:** none.

## Images read with `cv2.imdecode`
- **Choice:** read bytes with NumPy and decode, instead of `cv2.imread`.
- **Alternatives:** `cv2.imread`.
- **Reason:** `imread` fails on non-ASCII paths on Windows.
- **Trade-off:** none.

## Synthetic images for tests and CI
- **Choice:** tests generate small images (mixed sizes, `.jpg/.JPG/.jpeg/.png`, one exact duplicate).
- **Alternatives:** commit a few real MRI images.
- **Reason:** the dataset licence is unclear and medical images should not be re-hosted.
- **Trade-off:** tests prove the code works, not the real-data numbers; those come from `results/`.

## README numbers are generated, then checked
- **Choice:** tables are rendered from `results/` by `scripts/update_readme_tables.py`; `scripts/check_readme_claims.py` fails if any README number is not in `results/`.
- **Alternatives:** type the numbers by hand.
- **Reason:** a number typed by hand can drift away from the measurement it came from. This makes that impossible.
- **Trade-off:** prose numbers must match the precision in the results files.

## Pinned requirements, headless OpenCV
- **Choice:** exact versions; `opencv-python-headless`.
- **Alternatives:** unpinned; the GUI build of OpenCV.
- **Reason:** reproducible installs on CI (Linux and Windows); no GUI libraries needed.
- **Trade-off:** versions need updating by hand.

## Part B: the provided Training/Testing split, used as it is
- **Choice:** the main result trains on the dataset's Training folder and scores on its Testing folder, unchanged. A second score leaves out the Testing images that have a duplicate candidate in Training.
- **Alternatives:** re-split the data, or remove suspected duplicates before the main result.
- **Reason:** the split is part of the dataset as published, so the main number uses the data as given. The second score is a check on how much the near duplicates matter.
- **Trade-off:** the main number may be optimistic because of the near duplicates. The second score cannot measure that cleanly either: leaving out the flagged images also changes the class mix (most of the no-tumour test images are flagged).

## Part B: near duplicates need both perceptual hashes to agree
- **Choice:** a pair is a near-duplicate candidate when a difference hash (dHash) and a DCT hash (pHash) are both within 4 of 64 bits. Pair counts for each hash at 0, 2, 4 and 8 bits are reported too.
- **Alternatives:** one hash and one threshold; a learned image embedding.
- **Reason:** brain MRI slices look alike at 8x8 or 32x32, so one hash alone flags many unrelated pairs. Requiring both cuts those down, and the counts at several thresholds show how sensitive the result is.
- **Trade-off:** it can still flag different images that look very similar, and it can miss copies that were cropped or rotated.

## Part B: one BLAS/OpenMP thread per pool worker
- **Choice:** Part B pools start their workers with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1 (`worker_threads=1`). Part A's default is unchanged.
- **Alternatives:** library defaults.
- **Reason:** on Windows each worker with default settings reserved about 750 MB of commit memory (a BLAS buffer per hardware thread), so 12 workers ran out of commit memory on the laptop and the pool hung. With one thread a worker reserved about 42 MB (both measured once on the laptop, after importing the same libraries as a worker). N workers each starting N threads would also oversubscribe the CPU.
- **Trade-off:** none measured: the sequential filter stage took the same time per image with 1 thread and with the default.

## Part B: a small CNN sized for a laptop CPU
- **Choice:** four convolution blocks (about 98 thousand parameters), 128x128 grayscale input, 20 epochs, 3 seeds, no validation split and no early stopping.
- **Alternatives:** a pretrained network (transfer learning), a larger input, a GPU.
- **Reason:** it trains in minutes per seed on a CPU and gives an honest from-scratch baseline. A fixed number of epochs means the Testing split is never used to choose anything.
- **Trade-off:** it is a baseline, not a strong model. I did not try a pretrained network.
