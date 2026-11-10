# Credits

## Supplied code

- The six filter calls and their parameters (entropy, Gaussian, Sobel, Gabor, Hessian, Prewitt) and the `compute_glcm_features` helper were supplied with the original exercise. `src/mri_pipeline/filters.py` and the GLCM part of `src/mri_pipeline/features.py` are based on them; the header of each file lists what was changed here. Those supplied parts remain the work of their authors and are not covered by the MIT licence in `LICENSE`, which covers the rest of the code.

## Datasets

The images are not included in this repository, are not redistributed, and are not covered by the licence.

- **Part A:** "Brain MRI Images for Brain Tumor Detection" by Navoneel Chakrabarty, Kaggle: <https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection>. Kaggle lists the licence as "Data files © Original Authors". Please cite that page if you use it.
- **Part B:** "Brain Tumor MRI Dataset" by Masoud Nickparvar, Kaggle, version 2: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>. Licence: CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>). The author built it from three source datasets: figshare, SARTAJ and Br35H.

## Libraries

NumPy, SciPy, scikit-image, scikit-learn, OpenCV, matplotlib and PyTorch.
