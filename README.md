# Digital Mammography Classification

An end-to-end deep learning project for three-class BI-RADS classification of
digital mammograms from the INbreast dataset. The repository combines image
preparation, objective image-quality assessment, reproducible patient-wise data
splitting, preprocessing, transfer learning, and multiclass evaluation.

## Project scope

The original graduation project compared six ImageNet-pretrained CNN
architectures:

- VGG16
- VGG19
- ResNet50
- ResNet101
- InceptionV3
- DenseNet121

The classification targets are grouped as follows:

| Project class | BI-RADS categories |
|---|---|
| `SINIF1` | 1, 2 |
| `SINIF2` | 3, 4a |
| `SINIF3` | 4b, 4c, 5, 6 |

## Methodology

1. Convert DICOM mammograms to lossless PNG images.
2. Simulate salt-and-pepper and Gaussian noise, apply median, Gaussian, and
   Wiener filters, and compare the combinations with BRISQUE, PIQE, and NIQE.
3. Apply the selected denoising method and optional contrast enhancement.
4. Group images by patient before creating train, validation, and test folds.
5. Apply random augmentation only to the training split.
6. Train a classifier head and then fine-tune the upper backbone layers.
7. Evaluate the selected model once on the untouched test split using accuracy,
   macro F1, sensitivity, specificity, confusion matrices, and one-vs-rest ROC
   curves.

The patient-wise split is important because INbreast may contain multiple views
from the same patient. Keeping each patient in a single split reduces the risk
of overly optimistic evaluation.

## Repository structure

```text
.
├── scripts/
│   ├── convert_dicom.py
│   ├── prepare_dataset.py
│   ├── preprocess_images.py
│   ├── train.py
│   └── evaluate.py
├── matlab/
│   └── evaluate_filter_pairs.m
├── src/mammography/
│   ├── data.py
│   ├── labels.py
│   ├── models.py
│   └── preprocessing.py
├── notebooks/exploratory/
├── results/
├── tests/
└── requirements.txt
```

The `src/` package is the canonical implementation. The notebooks preserve the
original Colab experiments for reference; repeated cells and long training logs
were removed to make them easier to review.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data preparation

The INbreast images are not included in this repository. Place the downloaded
files under a local `data/raw/` directory, organized in folders whose names
contain their BI-RADS category, for example:

```text
data/raw/
├── BIRADS1/
├── BIRADS2/
├── BIRADS3/
├── BIRADS4a/
├── BIRADS4b/
├── BIRADS4c/
├── BIRADS5/
└── BIRADS6/
```

If the source files are DICOM images, convert them first:

```bash
python scripts/convert_dicom.py --input-dir data/raw_dicom --output-dir data/raw
```

### Metric-based filter selection

The original project selected a noise-filter combination by comparing median,
Gaussian, and Wiener denoising under controlled salt-and-pepper and Gaussian
noise. Place representative images in `data/metric_images/` and run:

```matlab
run('matlab/evaluate_filter_pairs.m')
```

The script calculates BRISQUE, PIQE, and NIQE for every variant. Lower values
indicate better perceived image quality for all three metrics. Per-image scores,
aggregate results, and a normalized filter-pair ranking are written to
`results/filter_selection/`.

Optional preprocessing can be applied without adding synthetic noise by
default. Replace `wiener` below if another filter ranks first in your analysis:

```bash
python scripts/preprocess_images.py \
  --input-dir data/raw \
  --output-dir data/preprocessed \
  --denoise wiener \
  --enhance clahe
```

Create leakage-aware splits. Patient IDs are inferred from the first numeric
token in each INbreast filename:

```bash
python scripts/prepare_dataset.py \
  --raw-dir data/preprocessed \
  --output-dir data/split
```

This command also writes `data/split/manifest.csv`, which records the assigned
patient and split for every image.

## Training

Choose one of `vgg16`, `vgg19`, `resnet50`, `resnet101`, `inceptionv3`, or
`densenet121`:

```bash
python scripts/train.py \
  --data-dir data/split \
  --model resnet50 \
  --output-dir artifacts/resnet50
```

## Evaluation

```bash
python scripts/evaluate.py \
  --data-dir data/split \
  --model-path artifacts/resnet50/best_finetuned.keras \
  --model resnet50 \
  --output-dir results/resnet50
```

The evaluation command creates a JSON metric summary, a class-wise CSV report,
a confusion matrix, and multiclass ROC curves.

## Results

The supplied notebooks document the original graduation-project experiments.
The public pipeline uses a stricter, shared evaluation protocol for all six
architectures. Final comparison scores should be regenerated with this
patient-wise split before being reported as benchmark results.

## Notes

- Model files (`.keras`, `.h5`) and the INbreast dataset are intentionally
  excluded from version control.
- The code is intended for research and educational use, not clinical decision
  making.
