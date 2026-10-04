# ECG-Age frozen inference checkpoint

The validated manuscript checkpoint is converted to a compact inference-only package for
application demos with:

```bash
python scripts/export_inference_checkpoint.py \
  --source /path/to/validated/best_model.pt \
  --output checkpoints/ecg_age_frozen_inference.pt
```

Validated source checkpoint SHA256:

```text
f9076fd3fb5dbda31b48069eac7b6517286f9e782ab4b8e3238821e7daf0e49a
```

Frozen target normalization:

- age mean: `55.045702521873395`
- age standard deviation: `20.83683569742911`
- inverse transform: `predicted_age = normalized_output * age_std + age_mean`

The compact package contains model weights and inference metadata only; optimizer and
training-state objects are omitted.

For a single WFDB record:

```bash
python scripts/infer_single.py \
  --checkpoint checkpoints/ecg_age_frozen_inference.pt \
  --wfdb-record /path/to/record \
  --json
```

This model is a research model and is not intended for clinical diagnosis or clinical
decision-making.
