# ECG-Age

Official code for:

**A Morphology-Aware Interpretable Hybrid Deep Learning Framework for ECG-Based Biological Age Estimation**

This repository provides the cleaned implementation of the final frozen ECG-Age model, including signal preprocessing, model architecture, training, and evaluation.

## Overview

The model estimates ECG-derived age from a standard 10-second, 12-lead electrocardiogram.

The final frozen pipeline uses:

- 12-lead ECG
- 500 Hz sampling frequency
- 10-second recordings
- 5,000 samples per lead
- WFDB physical signal (`p_signal`)
- per-lead mean centering
- first temporal derivative
- 24 input channels: 12 centered ECG leads + 12 derivative channels
- no per-lead standard-deviation normalization

## Architecture

The hybrid ECG-Age model consists of:

1. Lead-attention module
2. Multi-scale 1D convolutional feature extraction
3. Temporal convolutional network
4. Transformer encoder
5. Attention pooling and global mean pooling
6. Regression head for ECG-age prediction

The temporal convolutional blocks use dilations 1, 2, 4, 8, and 16.

The Transformer component contains six encoder blocks with four attention heads.

The final model contains **3,532,674 trainable parameters**.

## Repository structure

- `configs/ecg_age.yaml` — frozen model configuration
- `src/model.py` — ECG-Age model architecture
- `src/preprocessing.py` — frozen signal preprocessing
- `src/dataset.py` — WFDB dataset loader
- `scripts/train.py` — model training
- `scripts/evaluate.py` — model evaluation
- `requirements.txt` — Python dependencies

## Installation

Validated environment:

- Python 3.10.6
- PyTorch 2.6.0
- NumPy 1.26.4
- Pandas 2.2.3
- scikit-learn 1.1.3
- WFDB 4.3.1

Install dependencies with:

`pip install -r requirements.txt`

A CUDA-enabled PyTorch installation is recommended for training and large-scale evaluation.

## Data

ECG waveform data and patient-level cohort files are not distributed with this repository.

Input CSV files require at minimum:

- `subject_id`
- `age`
- `path`

The `path` column identifies the corresponding WFDB ECG record relative to the ECG root directory supplied to the script.

## Signal preprocessing

For each ECG:

1. Read the WFDB physical signal using `p_signal`.
2. Replace non-finite values with zero.
3. Mean-center each of the 12 ECG leads independently.
4. Compute the first temporal derivative of each centered lead.
5. Concatenate centered signals and derivatives to create 24 input channels.
6. Crop or zero-pad the signal to 5,000 samples.

No per-lead z-score normalization is used in the frozen final model.

Training augmentation includes temporal translation, additive Gaussian noise, temporal masking, and paired physical-lead dropout.

Random global amplitude scaling is not used.

## Training

The frozen model configuration is provided in `configs/ecg_age.yaml`.

Primary training settings:

- random seed: 42
- maximum epochs: 60
- early-stopping patience: 10
- batch size: 64
- learning rate: 0.0003
- weight decay: 0.01
- optimizer: AdamW
- loss: SmoothL1Loss with beta = 5
- scheduler: CosineAnnealingLR
- gradient clipping: 1.0
- mixed precision on CUDA

Chronological age is standardized using the mean and standard deviation calculated from the training set only.

Model selection is based on the lowest development-set MAE.

## Evaluation

Example:

`python scripts/evaluate.py --manifest /path/to/evaluation.csv --ecg-root /path/to/ecg/files --checkpoint /path/to/best_model.pt --output-dir /path/to/results`

The evaluation script reports:

- mean absolute error (MAE)
- root mean squared error (RMSE)
- R²
- Pearson correlation coefficient
- Lin concordance correlation coefficient (CCC)
- calibration intercept
- calibration slope
- mean ECG-age gap
- Bland-Altman bias and 95% limits of agreement

ECG-age gap is defined as predicted ECG age minus chronological age.

By default, confidence intervals are estimated using 2,000 patient-level bootstrap replicates with seed 20260819.

## Frozen model provenance

- selected epoch: 12
- development MAE: 7.3037868 years
- trainable parameters: 3,532,674

Frozen checkpoint SHA256:

`f9076fd3fb5dbda31b48069eac7b6517286f9e782ab4b8e3238821e7daf0e49a`

The trained checkpoint is not distributed in this repository.

## Reproducibility validation

The cleaned implementation was verified against the exact frozen checkpoint using strict PyTorch state-dictionary loading, with no missing or unexpected parameter keys.

The complete inference pipeline was also replayed on the frozen locked test cohort.

| Metric | Value |
|---|---:|
| N | 19,431 |
| MAE | 7.363792 |
| RMSE | 9.618699 |
| R² | 0.786941 |
| Pearson r | 0.887789 |
| CCC | 0.884864 |
| Mean ECG-age gap | -0.138814 |

## Privacy and data availability

This repository does not distribute:

- ECG waveform data
- subject identifiers
- patient-level cohort manifests
- patient-level predictions
- protected clinical data

Users must obtain the source ECG data independently and comply with the applicable data-use requirements.

## Citation

Citation information will be added when the manuscript is published.
