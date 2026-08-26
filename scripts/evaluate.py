#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import MIMICECGAgeDataset
from src.model import ECGAgeRegressor, count_trainable_parameters


# ---------------------------------------------------------------------
# Frozen primary model constants
# ---------------------------------------------------------------------

EXPECTED_PARAMETERS = 3_532_674
EXPECTED_INPUT_CHANNELS = 24

FROZEN_PRIMARY_SHA256 = (
    "f9076fd3fb5dbda31b48069eac7b6517286f9e782ab4b8e3238821e7daf0e49a"
)

DEFAULT_N_BOOT = 2000
DEFAULT_BOOT_SEED = 20260819


METRIC_NAMES = [
    "MAE",
    "RMSE",
    "R2",
    "Pearson_r",
    "CCC",
    "calibration_intercept_true_on_pred",
    "calibration_slope_true_on_pred",
    "mean_AgeGap",
    "BlandAltman_bias",
    "BlandAltman_LOA_low",
    "BlandAltman_LOA_high",
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen full-12-lead ECG-Age ECG-age model."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            "Evaluation manifest containing subject_id, age, and path."
        ),
    )

    parser.add_argument(
        "--ecg-root",
        required=True,
        type=Path,
        help=(
            "Path to the MIMIC-IV-ECG WFDB 'files' directory."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="ECG-Age checkpoint (.pt).",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for evaluation outputs.",
    )

    parser.add_argument(
        "--expected-checkpoint-sha256",
        type=str,
        default=None,
        help=(
            "Optional checkpoint SHA256 to enforce before evaluation."
        ),
    )

    parser.add_argument(
        "--expected-n",
        type=int,
        default=None,
        help=(
            "Optional expected evaluation cohort size."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_N_BOOT,
        help=(
            "Number of patient-level bootstrap replicates. "
            "Use 0 to disable bootstrap."
        ),
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOT_SEED,
    )

    parser.add_argument(
        "--verify-frozen-primary",
        action="store_true",
        help=(
            "Require the checkpoint SHA256 to equal the frozen "
            "manuscript primary checkpoint."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sha256_file(
    path,
    chunk_size=1024 * 1024,
):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def save_json(
    obj,
    path,
):
    with open(path, "w") as f:
        json.dump(
            obj,
            f,
            indent=2,
        )


# ---------------------------------------------------------------------
# Exact STEP03B statistical definitions
# ---------------------------------------------------------------------

def pearson_r(
    y: np.ndarray,
    p: np.ndarray,
) -> float:
    y0 = (
        y - np.mean(y)
    )

    p0 = (
        p - np.mean(p)
    )

    denom = (
        np.sqrt(
            np.sum(
                y0 ** 2
            )
        )
        *
        np.sqrt(
            np.sum(
                p0 ** 2
            )
        )
    )

    if denom <= 0:
        return float("nan")

    return float(
        np.sum(
            y0 * p0
        )
        / denom
    )


def ccc(
    y: np.ndarray,
    p: np.ndarray,
) -> float:
    mean_y = np.mean(
        y
    )

    mean_p = np.mean(
        p
    )

    var_y = np.mean(
        (y - mean_y) ** 2
    )

    var_p = np.mean(
        (p - mean_p) ** 2
    )

    cov = np.mean(
        (y - mean_y)
        * (p - mean_p)
    )

    denom = (
        var_y
        + var_p
        + (
            mean_y
            - mean_p
        ) ** 2
    )

    if denom <= 0:
        return float("nan")

    return float(
        2.0
        * cov
        / denom
    )


def calibration_true_on_pred(
    y: np.ndarray,
    p: np.ndarray,
):
    """
    Continuous calibration:

        true_age = intercept + slope * predicted_age
    """

    mean_p = np.mean(
        p
    )

    mean_y = np.mean(
        y
    )

    denom = np.sum(
        (
            p
            - mean_p
        ) ** 2
    )

    if denom <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    slope = (
        np.sum(
            (
                p
                - mean_p
            )
            *
            (
                y
                - mean_y
            )
        )
        / denom
    )

    intercept = (
        mean_y
        - slope
        * mean_p
    )

    return (
        float(
            intercept
        ),
        float(
            slope
        ),
    )


def regression_metrics(
    y: np.ndarray,
    p: np.ndarray,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    err = (
        p - y
    )

    mae = np.mean(
        np.abs(
            err
        )
    )

    mse = np.mean(
        err ** 2
    )

    rmse = np.sqrt(
        mse
    )

    sst = np.sum(
        (
            y
            - np.mean(y)
        ) ** 2
    )

    r2 = (
        1.0
        - np.sum(
            err ** 2
        )
        / sst
        if sst > 0
        else float("nan")
    )

    (
        cal_intercept,
        cal_slope,
    ) = calibration_true_on_pred(
        y,
        p,
    )

    bias = np.mean(
        err
    )

    # Exact STEP03B convention.
    sd_diff = np.std(
        err,
        ddof=1,
    )

    return {
        "MAE": float(
            mae
        ),

        "RMSE": float(
            rmse
        ),

        "R2": float(
            r2
        ),

        "Pearson_r": (
            pearson_r(
                y,
                p,
            )
        ),

        "CCC": (
            ccc(
                y,
                p,
            )
        ),

        "calibration_intercept_true_on_pred": (
            cal_intercept
        ),

        "calibration_slope_true_on_pred": (
            cal_slope
        ),

        "mean_AgeGap": float(
            bias
        ),

        "BlandAltman_bias": float(
            bias
        ),

        "BlandAltman_LOA_low": float(
            bias
            - 1.96
            * sd_diff
        ),

        "BlandAltman_LOA_high": float(
            bias
            + 1.96
            * sd_diff
        ),
    }


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

def ci95(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return (
            float("nan"),
            float("nan"),
        )

    low, high = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(low),
        float(high),
    )


def bootstrap_metrics(
    y,
    p,
    n_boot,
    seed,
):
    """
    STEP03B patient-level bootstrap.

    For a cohort with one ECG observation per patient,
    rows are sampled with replacement using NumPy's
    default_rng and an index vector of length N.
    """

    rng = np.random.default_rng(
        seed
    )

    n = len(
        y
    )

    rows = []

    for b in range(
        n_boot
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        metrics = regression_metrics(
            y[idx],
            p[idx],
        )

        row = {
            "bootstrap_id": b,
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

        if (
            (b + 1) % 100 == 0
            or b == 0
        ):
            print(
                f"Bootstrap "
                f"{b + 1:4d}/"
                f"{n_boot}",
                flush=True,
            )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------

@torch.no_grad()
def predict(
    model,
    loader,
    device,
    age_mean,
    age_std,
):
    model.eval()

    ys = []
    ps = []
    subject_ids = []

    for (
        xb,
        yb,
        subject_batch,
    ) in loader:

        xb = xb.to(
            device,
            non_blocking=True,
        )

        pred_norm = model(
            xb
        )

        # Exact STEP03B inverse target transformation.
        pred = (
            pred_norm
            * age_std
            + age_mean
        )

        ys.append(
            yb.numpy()
        )

        ps.append(
            pred
            .detach()
            .cpu()
            .numpy()
        )

        subject_ids.append(
            subject_batch
            .detach()
            .cpu()
            .numpy()
        )

    y = np.concatenate(
        ys
    )

    p = np.concatenate(
        ps
    )

    subject_ids = np.concatenate(
        subject_ids
    )

    return (
        y,
        p,
        subject_ids,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    # --------------------------------------------------------------
    # Required files
    # --------------------------------------------------------------

    for path, description in [
        (
            args.manifest,
            "evaluation manifest",
        ),
        (
            args.checkpoint,
            "checkpoint",
        ),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{description}: {path}"
            )

    if not args.ecg_root.exists():
        raise FileNotFoundError(
            f"ECG root: {args.ecg_root}"
        )

    # --------------------------------------------------------------
    # Checkpoint preflight BEFORE waveform evaluation
    # --------------------------------------------------------------

    checkpoint_sha256 = sha256_file(
        args.checkpoint
    )

    print("=" * 80)
    print("ECG-Age ECG-age evaluation")
    print("=" * 80)

    print(
        "Checkpoint SHA256:",
        checkpoint_sha256,
    )

    expected_hash = (
        args.expected_checkpoint_sha256
    )

    if args.verify_frozen_primary:
        if (
            expected_hash is not None
            and expected_hash
            != FROZEN_PRIMARY_SHA256
        ):
            raise RuntimeError(
                "--verify-frozen-primary conflicts with "
                "--expected-checkpoint-sha256."
            )

        expected_hash = (
            FROZEN_PRIMARY_SHA256
        )

    if expected_hash is not None:
        if (
            checkpoint_sha256
            != expected_hash
        ):
            raise RuntimeError(
                "Checkpoint SHA256 mismatch.\n"
                f"Expected: {expected_hash}\n"
                f"Observed: {checkpoint_sha256}"
            )

        print(
            "Checkpoint hash : PASS"
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    if (
        "age_mean"
        not in checkpoint
        or "age_std"
        not in checkpoint
    ):
        raise RuntimeError(
            "Checkpoint does not contain age_mean/age_std."
        )

    age_mean = float(
        checkpoint[
            "age_mean"
        ]
    )

    age_std = float(
        checkpoint[
            "age_std"
        ]
    )

    input_channels = int(
        checkpoint.get(
            "input_channels",
            EXPECTED_INPUT_CHANNELS,
        )
    )

    if (
        input_channels
        != EXPECTED_INPUT_CHANNELS
    ):
        raise RuntimeError(
            "Frozen full-12 ECG-Age requires "
            f"{EXPECTED_INPUT_CHANNELS} channels; "
            f"checkpoint reports {input_channels}."
        )

    model = ECGAgeRegressor(
        in_ch=24,
        d_model=192,
        depth=6,
        nhead=4,
    )

    n_params = (
        count_trainable_parameters(
            model
        )
    )

    if (
        n_params
        != EXPECTED_PARAMETERS
    ):
        raise RuntimeError(
            "ECG-Age architecture parameter mismatch.\n"
            f"Expected: {EXPECTED_PARAMETERS:,}\n"
            f"Observed: {n_params:,}"
        )

    load_result = (
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            "strict=True checkpoint loading failed."
        )

    print(
        "Architecture     : PASS"
    )
    print(
        f"Parameters       : {n_params:,}"
    )
    print(
        f"Age mean         : {age_mean:.10f}"
    )
    print(
        f"Age std          : {age_std:.10f}"
    )
    print(
        "Checkpoint epoch :",
        checkpoint.get(
            "epoch",
            "NA",
        ),
    )
    print(
        "Checkpoint variant:",
        checkpoint.get(
            "variant",
            "NA",
        ),
    )

    # --------------------------------------------------------------
    # Manifest
    # --------------------------------------------------------------

    df = pd.read_csv(
        args.manifest
    )

    required_columns = {
        "subject_id",
        "age",
        "path",
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Evaluation manifest missing columns: "
            f"{sorted(missing)}"
        )

    if (
        args.expected_n is not None
        and len(df)
        != args.expected_n
    ):
        raise RuntimeError(
            "Evaluation cohort size mismatch.\n"
            f"Expected: {args.expected_n:,}\n"
            f"Observed: {len(df):,}"
        )

    print(
        f"Evaluation N     : {len(df):,}"
    )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    dataset = MIMICECGAgeDataset(
        df=df,
        base_path=args.ecg_root,
        target_len=5000,
        augment=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    model = model.to(
        device
    )

    print(
        f"Device           : {device}"
    )

    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------

    (
        y,
        p,
        subject_ids,
    ) = predict(
        model=model,
        loader=loader,
        device=device,
        age_mean=age_mean,
        age_std=age_std,
    )

    if len(y) != len(df):
        raise RuntimeError(
            "Prediction count does not match manifest."
        )

    if not np.all(
        np.isfinite(
            p
        )
    ):
        raise RuntimeError(
            "Non-finite predictions detected."
        )

    # --------------------------------------------------------------
    # Point estimates
    # --------------------------------------------------------------

    point = regression_metrics(
        y,
        p,
    )

    print("\nPoint estimates")
    print("-" * 80)

    for metric in METRIC_NAMES:
        print(
            f"{metric:45s} "
            f"{point[metric]:.8f}"
        )

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.DataFrame({
        "subject_id": (
            subject_ids
        ),
        "age_true": (
            y
        ),
        "age_pred": (
            p
        ),
    })

    predictions[
        "age_gap"
    ] = (
        predictions[
            "age_pred"
        ]
        - predictions[
            "age_true"
        ]
    )

    predictions[
        "abs_error"
    ] = np.abs(
        predictions[
            "age_gap"
        ]
    )

    predictions.to_csv(
        args.output_dir
        / "predictions.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------------

    bootstrap_df = None
    metric_output = {}

    if args.bootstrap > 0:
        print(
            "\nPatient-level bootstrap: "
            f"{args.bootstrap:,} replicates | "
            f"seed={args.bootstrap_seed}"
        )

        bootstrap_df = bootstrap_metrics(
            y=y,
            p=p,
            n_boot=args.bootstrap,
            seed=args.bootstrap_seed,
        )

        bootstrap_df.to_csv(
            args.output_dir
            / "bootstrap_replicates.csv.gz",
            index=False,
            compression="gzip",
        )

        for metric in METRIC_NAMES:
            low, high = ci95(
                bootstrap_df[
                    metric
                ].to_numpy()
            )

            metric_output[
                metric
            ] = {
                "estimate": float(
                    point[
                        metric
                    ]
                ),
                "ci95_low": low,
                "ci95_high": high,
            }

    else:
        for metric in METRIC_NAMES:
            metric_output[
                metric
            ] = {
                "estimate": float(
                    point[
                        metric
                    ]
                ),
                "ci95_low": None,
                "ci95_high": None,
            }

    # --------------------------------------------------------------
    # Results tables
    # --------------------------------------------------------------

    metric_rows = []

    for metric in METRIC_NAMES:
        result = (
            metric_output[
                metric
            ]
        )

        metric_rows.append({
            "metric": metric,
            "estimate": (
                result[
                    "estimate"
                ]
            ),
            "CI95_low": (
                result[
                    "ci95_low"
                ]
            ),
            "CI95_high": (
                result[
                    "ci95_high"
                ]
            ),
        })

    pd.DataFrame(
        metric_rows
    ).to_csv(
        args.output_dir
        / "metrics.csv",
        index=False,
    )

    result_json = {
        "model": (
            "full12_ECG-Age"
        ),

        "n": int(
            len(y)
        ),

        "checkpoint_sha256": (
            checkpoint_sha256
        ),

        "trainable_parameters": (
            n_params
        ),

        "age_mean": (
            age_mean
        ),

        "age_std": (
            age_std
        ),

        "checkpoint_epoch": (
            checkpoint.get(
                "epoch"
            )
        ),

        "checkpoint_variant": (
            checkpoint.get(
                "variant"
            )
        ),

        "metrics": (
            metric_output
        ),

        "bootstrap": {
            "replicates": int(
                args.bootstrap
            ),
            "seed": int(
                args.bootstrap_seed
            ),
            "ci_method": (
                "2.5th and 97.5th percentile"
            ),
        },

        "calibration_definition": (
            "true_age = intercept + slope * predicted_age"
        ),

        "age_gap_definition": (
            "predicted_age - chronological_age"
        ),

        "bland_altman": {
            "bias": (
                "mean(predicted_age - chronological_age)"
            ),
            "limits": (
                "bias +/- 1.96 * sample SD of differences (ddof=1)"
            ),
        },

        "created_at": (
            datetime.now()
            .isoformat()
        ),
    }

    save_json(
        result_json,
        args.output_dir
        / "metrics.json",
    )

    provenance = {
        "evaluation_script": (
            str(
                Path(
                    __file__
                ).name
            )
        ),

        "evaluation_script_sha256": (
            sha256_file(
                Path(
                    __file__
                )
            )
        ),

        "manifest_sha256": (
            sha256_file(
                args.manifest
            )
        ),

        "checkpoint_sha256": (
            checkpoint_sha256
        ),

        "preprocessing": {
            "wfdb_signal": (
                "p_signal"
            ),
            "normalization": (
                "per-lead mean_center_only"
            ),
            "include_raw": True,
            "use_first_derivative": True,
            "scale_augmentation": False,
            "evaluation_augmentation": False,
            "target_length": 5000,
            "input_channels": 24,
        },

        "checkpoint_loaded_strict": True,
    }

    save_json(
        provenance,
        args.output_dir
        / "provenance.json",
    )

    # --------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"N       : {len(y):,}"
    )

    print(
        f"MAE     : {point['MAE']:.6f}"
    )

    print(
        f"RMSE    : {point['RMSE']:.6f}"
    )

    print(
        f"R2      : {point['R2']:.6f}"
    )

    print(
        f"Pearson : {point['Pearson_r']:.6f}"
    )

    print(
        f"CCC     : {point['CCC']:.6f}"
    )

    print(
        f"AgeGap  : {point['mean_AgeGap']:.6f}"
    )

    print(
        "Saved to:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
