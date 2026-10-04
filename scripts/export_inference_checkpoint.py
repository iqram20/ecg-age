#!/usr/bin/env python3

import argparse
import hashlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import ECGAgeRegressor, count_trainable_parameters

FROZEN_SOURCE_SHA256 = (
    "f9076fd3fb5dbda31b48069eac7b6517286f9e782ab4b8e3238821e7daf0e49a"
)
EXPECTED_PARAMETERS = 3_532_674
AGE_MEAN = 55.045702521873395
AGE_STD = 20.83683569742911


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description="Export a compact inference-only checkpoint from the validated ECG-Age checkpoint."
    )
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "ecg_age_frozen_inference.pt",
    )
    args = ap.parse_args()

    observed_sha = sha256_file(args.source)
    if observed_sha != FROZEN_SOURCE_SHA256:
        raise RuntimeError(
            "Source checkpoint SHA256 mismatch.\n"
            f"Expected: {FROZEN_SOURCE_SHA256}\n"
            f"Observed: {observed_sha}"
        )

    ckpt = torch.load(args.source, map_location="cpu", weights_only=False)

    required = {"model_state_dict", "age_mean", "age_std"}
    missing = required - set(ckpt)
    if missing:
        raise RuntimeError(f"Source checkpoint missing keys: {sorted(missing)}")

    age_mean = float(ckpt["age_mean"])
    age_std = float(ckpt["age_std"])

    if abs(age_mean - AGE_MEAN) > 1e-10 or abs(age_std - AGE_STD) > 1e-10:
        raise RuntimeError(
            "Age-normalization values do not match the frozen configuration.\n"
            f"age_mean={age_mean}\n"
            f"age_std={age_std}"
        )

    model = ECGAgeRegressor(in_ch=24, d_model=192, depth=6, nhead=4)
    n_params = count_trainable_parameters(model)
    if n_params != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Architecture mismatch: expected {EXPECTED_PARAMETERS:,}, observed {n_params:,}"
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    package = {
        "format": "ecg-age-inference-v1",
        "model_state_dict": ckpt["model_state_dict"],
        "age_mean": age_mean,
        "age_std": age_std,
        "input_channels": 24,
        "epoch": int(ckpt.get("epoch", 12)),
        "variant": ckpt.get(
            "variant",
            "raw_derivative_centered_physical_no_scale_aug",
        ),
        "source_checkpoint_sha256": FROZEN_SOURCE_SHA256,
        "expected_trainable_parameters": EXPECTED_PARAMETERS,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.output)

    print("Saved:", args.output)
    print("Source checkpoint SHA256:", FROZEN_SOURCE_SHA256)
    print("Inference package SHA256:", sha256_file(args.output))
    print("Age mean:", age_mean)
    print("Age std :", age_std)
    print("Parameters:", f"{n_params:,}")


if __name__ == "__main__":
    main()
