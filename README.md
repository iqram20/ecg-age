# ECG-Age

Official code for:

**Morphology-aware deep learning for ECG-derived age estimation and its association with postoperative outcomes**

This public repository provides the frozen ECG-Age inference checkpoint, model architecture, core preprocessing, and a simple single-record inference workflow.

## Overview

The model estimates ECG-derived age from a standard 10-second, 12-lead electrocardiogram.

The frozen inference pipeline uses:

- 12-lead ECG
- 500 Hz sampling frequency
- 10-second recordings
- 5,000 samples per lead
- WFDB physical signal (`p_signal`)
- per-lead mean centering
- first temporal derivative
- 24 input channels: 12 centered ECG leads + 12 derivative channels
- no per-lead standard-deviation normalization

## Public repository contents

- `checkpoints/ecg_age_frozen_inference.pt` — compact frozen inference checkpoint
- `configs/inference.json` — frozen inference metadata and age-normalization values
- `src/model.py` — ECG-Age model architecture
- `src/preprocessing.py` — core frozen preprocessing
- `scripts/infer_single.py` — single-record WFDB inference
- `requirements.txt` — Python dependencies

## Installation

Install dependencies with:

`pip install -r requirements.txt`

## Signal preprocessing

For each ECG:

1. Read the WFDB physical signal using `p_signal`.
2. Replace non-finite values with zero.
3. Mean-center each of the 12 ECG leads independently.
4. Compute the first temporal derivative of each centered lead.
5. Concatenate centered signals and derivatives to create 24 input channels.
6. Crop or zero-pad the signal to 5,000 samples.

No per-lead z-score normalization is used in the frozen final model.

## Single-record inference

Example:

```bash
python scripts/infer_single.py \
  --checkpoint checkpoints/ecg_age_frozen_inference.pt \
  --wfdb-record /path/to/record \
  --json
```

Frozen target normalization values are provided in `configs/inference.json`.

The compact inference package contains model weights and inference metadata only; optimizer and training-state objects are omitted.

## Frozen model provenance

- selected epoch: 12
- development MAE: 7.3037868 years
- trainable parameters: 3,532,674

Frozen source checkpoint SHA256:

`f9076fd3fb5dbda31b48069eac7b6517286f9e782ab4b8e3238821e7daf0e49a`

## Privacy and data availability

This repository does not distribute ECG waveform data, subject identifiers, patient-level cohort manifests, patient-level predictions, or protected clinical data.

Users must obtain the source ECG data independently and comply with the applicable data-use requirements.

## Research use

This model is a research model and is not intended for clinical diagnosis or clinical decision-making.

## Citation

Citation information will be added when the manuscript is published.
