#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import wfdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import ECGAgeRegressor, count_trainable_parameters
from src.preprocessing import prepare_frozen_channels, fix_length

EXPECTED_PARAMETERS = 3_532_674
EXPECTED_SOURCE_SHA256 = (
    "f9076fd3fb5dbda31b48069eac7b6517286f9e782ab4b8e3238821e7daf0e49a"
)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required = {"model_state_dict", "age_mean", "age_std"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {sorted(missing)}")

    source_sha = checkpoint.get("source_checkpoint_sha256")
    if source_sha is not None and source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Inference package does not originate from the validated frozen checkpoint."
        )

    model = ECGAgeRegressor(in_ch=24, d_model=192, depth=6, nhead=4)
    n_params = count_trainable_parameters(model)
    if n_params != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Architecture mismatch: expected {EXPECTED_PARAMETERS:,}, observed {n_params:,}"
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return (
        model,
        float(checkpoint["age_mean"]),
        float(checkpoint["age_std"]),
    )


def read_wfdb_record(record_path):
    record_path = str(record_path)
    if record_path.endswith(".dat") or record_path.endswith(".hea"):
        record_path = record_path[:-4]

    record = wfdb.rdrecord(record_path)

    x = torch.tensor(record.p_signal.T, dtype=torch.float32)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.ndim != 2 or x.shape[0] != 12:
        raise RuntimeError(
            f"Expected 12-lead ECG [12,T]; observed {tuple(x.shape)}"
        )

    # Same frozen preprocessing used in model development.
    x = prepare_frozen_channels(x)
    x = fix_length(x, target_len=5000, augment=False)

    if tuple(x.shape) != (24, 5000):
        raise RuntimeError(f"Prepared input has unexpected shape {tuple(x.shape)}")

    return x.unsqueeze(0)


@torch.no_grad()
def predict_age(model, x, age_mean, age_std, device):
    x = x.to(device)
    pred_norm = model(x)
    pred_age = pred_norm * age_std + age_mean
    return float(pred_age.item()), float(pred_norm.item())


def main():
    ap = argparse.ArgumentParser(
        description="Run ECG-Age inference on one 10-second 12-lead WFDB record."
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument(
        "--wfdb-record",
        required=True,
        help="WFDB record path without .hea/.dat extension (extensions are also accepted).",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, age_mean, age_std = load_model(args.checkpoint, device)
    x = read_wfdb_record(args.wfdb_record)
    predicted_age, normalized_output = predict_age(
        model, x, age_mean, age_std, device
    )

    result = {
        "predicted_ecg_age_years": predicted_age,
        "normalized_model_output": normalized_output,
        "age_mean": age_mean,
        "age_std": age_std,
        "device": str(device),
        "checkpoint_sha256": sha256_file(args.checkpoint),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Predicted ECG age: {predicted_age:.2f} years")
        print(f"Age normalization: mean={age_mean:.10f}, std={age_std:.10f}")


if __name__ == "__main__":
    main()
