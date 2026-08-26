#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader


# Allow: python scripts/train.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import MIMICECGAgeDataset
from src.model import ECGAgeRegressor, count_trainable_parameters


# Frozen manuscript constants
FROZEN_SPLIT_SHA256 = (
    "6b87510fb8869ac85e33997ad7c124369f68527cb16f598390c14a4daf5c8b5c"
)

EXPECTED_TRAIN_N = 68005
EXPECTED_DEV_N = 9716
EXPECTED_PARAMETERS = 3_532_674


@dataclass
class Config:
    seed: int = 42

    epochs: int = 60
    patience: int = 10
    min_delta: float = 0.0

    batch_size: int = 64
    num_workers: int = 8

    lr: float = 3e-4
    weight_decay: float = 1e-2

    target_len: int = 5000
    input_channels: int = 24
    d_model: int = 192
    depth: int = 6
    nhead: int = 4

    smooth_l1_beta: float = 5.0
    grad_clip: float = 1.0

    train_augment: bool = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen full-12-lead ECG-Age ECG-age model."
        )
    )

    parser.add_argument(
        "--train-csv",
        required=True,
        type=Path,
        help="Training manifest CSV.",
    )

    parser.add_argument(
        "--dev-csv",
        required=True,
        type=Path,
        help="Development/validation manifest CSV.",
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
        "--output-dir",
        type=Path,
        default=Path("runs/ecg_age"),
        help="Directory in which training runs are written.",
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "Optional frozen split manifest. If supplied, its "
            "SHA256 is checked against the published split hash."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


def run_epoch(
    model,
    loader,
    device,
    criterion,
    age_mean,
    age_std,
    optimizer=None,
    scaler=None,
    grad_clip=1.0,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0

    y_true_all = []
    y_pred_all = []
    subject_all = []

    for xb, yb, sid in loader:
        xb = xb.to(
            device,
            non_blocking=True,
        )

        yb = yb.to(
            device,
            non_blocking=True,
        )

        # Frozen model normalizes target age using
        # TRAIN-set statistics only.
        yb_norm = (
            yb - age_mean
        ) / (
            age_std + 1e-8
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            if (
                training
                and scaler is not None
            ):
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    pred_norm = model(xb)

                    loss = criterion(
                        pred_norm,
                        yb_norm,
                    )
            else:
                pred_norm = model(xb)

                loss = criterion(
                    pred_norm,
                    yb_norm,
                )

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

                optimizer.step()

        pred_age = (
            pred_norm
            * (age_std + 1e-8)
            + age_mean
        )

        total_loss += (
            float(loss.item())
            * yb.size(0)
        )

        y_true_all.append(
            yb.detach().cpu().numpy()
        )

        y_pred_all.append(
            pred_age.detach().cpu().numpy()
        )

        subject_all.append(
            sid.detach().cpu().numpy()
        )

    y_true = np.concatenate(
        y_true_all
    )

    y_pred = np.nan_to_num(
        np.concatenate(
            y_pred_all
        )
    )

    subject_ids = np.concatenate(
        subject_all
    )

    metrics = regression_metrics(
        y_true,
        y_pred,
    )

    metrics["loss"] = float(
        total_loss
        / len(loader.dataset)
    )

    return (
        metrics,
        y_true,
        y_pred,
        subject_ids,
    )


def main():
    args = parse_args()

    cfg = Config(
        seed=args.seed
    )

    set_seed(
        cfg.seed
    )

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------

    if not args.train_csv.exists():
        raise FileNotFoundError(
            args.train_csv
        )

    if not args.dev_csv.exists():
        raise FileNotFoundError(
            args.dev_csv
        )

    if not args.ecg_root.exists():
        raise FileNotFoundError(
            args.ecg_root
        )

    split_hash = None

    if args.split_manifest is not None:
        if not args.split_manifest.exists():
            raise FileNotFoundError(
                args.split_manifest
            )

        split_hash = sha256_file(
            args.split_manifest
        )

        if (
            split_hash
            != FROZEN_SPLIT_SHA256
        ):
            raise RuntimeError(
                "Frozen split manifest SHA256 mismatch.\n"
                f"Expected: {FROZEN_SPLIT_SHA256}\n"
                f"Observed: {split_hash}"
            )

    # ------------------------------------------------------------------
    # TRAIN + DEV only
    # ------------------------------------------------------------------

    df_train = pd.read_csv(
        args.train_csv
    )

    df_dev = pd.read_csv(
        args.dev_csv
    )

    required_columns = {
        "subject_id",
        "age",
        "path",
    }

    for name, df in [
        ("train", df_train),
        ("dev", df_dev),
    ]:
        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise RuntimeError(
                f"{name} manifest missing columns: "
                f"{sorted(missing)}"
            )

    if len(df_train) != EXPECTED_TRAIN_N:
        raise RuntimeError(
            "Frozen TRAIN size mismatch: "
            f"{len(df_train):,} != "
            f"{EXPECTED_TRAIN_N:,}"
        )

    if len(df_dev) != EXPECTED_DEV_N:
        raise RuntimeError(
            "Frozen DEV size mismatch: "
            f"{len(df_dev):,} != "
            f"{EXPECTED_DEV_N:,}"
        )

    train_ids = set(
        df_train[
            "subject_id"
        ].astype(int)
    )

    dev_ids = set(
        df_dev[
            "subject_id"
        ].astype(int)
    )

    overlap = (
        train_ids
        .intersection(dev_ids)
    )

    if overlap:
        raise RuntimeError(
            "Train/DEV subject overlap detected: "
            f"{len(overlap):,}"
        )

    # TRAIN-only target normalization.
    train_ages = pd.to_numeric(
        df_train["age"],
        errors="raise",
    ).astype(float)

    age_mean = float(
        train_ages.mean()
    )

    age_std = float(
        train_ages.std(
            ddof=0
        )
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        args.output_dir
        / f"{timestamp}_ECG-Age_seed{cfg.seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    df_train.to_csv(
        run_dir
        / "train_manifest_used.csv",
        index=False,
    )

    df_dev.to_csv(
        run_dir
        / "dev_manifest_used.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    train_ds = MIMICECGAgeDataset(
        df=df_train,
        base_path=args.ecg_root,
        target_len=cfg.target_len,
        augment=True,
    )

    dev_ds = MIMICECGAgeDataset(
        df=df_dev,
        base_path=args.ecg_root,
        target_len=cfg.target_len,
        augment=False,
    )

    pin_memory = (
        torch.cuda.is_available()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            cfg.num_workers > 0
        ),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            cfg.num_workers > 0
        ),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ECGAgeRegressor(
        in_ch=cfg.input_channels,
        d_model=cfg.d_model,
        depth=cfg.depth,
        nhead=cfg.nhead,
    ).to(device)

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
            "ECG-Age parameter count mismatch.\n"
            f"Expected: {EXPECTED_PARAMETERS:,}\n"
            f"Observed: {n_params:,}"
        )

    current_script = Path(
        __file__
    ).resolve()

    provenance = {
        "created_at": (
            datetime.now()
            .isoformat()
        ),
        "model": "ECG-Age",
        "seed": cfg.seed,

        "preprocessing": {
            "wfdb_signal": "p_signal",
            "normalization": (
                "per-lead mean_center_only"
            ),
            "include_raw": True,
            "use_first_derivative": True,
            "input_channels": 24,
            "scale_augmentation": False,
        },

        "train_n": int(
            len(df_train)
        ),

        "dev_n": int(
            len(df_dev)
        ),

        "locked_test_read": False,
        "surgical_cohort_read": False,

        "target_age_normalization_source": (
            "train_only"
        ),

        "age_mean": age_mean,
        "age_std": age_std,

        "trainable_parameters": (
            n_params
        ),

        "training_script": str(
            current_script
        ),

        "training_script_sha256": (
            sha256_file(
                current_script
            )
        ),

        "train_csv_sha256": (
            sha256_file(
                args.train_csv
            )
        ),

        "dev_csv_sha256": (
            sha256_file(
                args.dev_csv
            )
        ),

        "split_manifest_sha256": (
            split_hash
        ),

        "config": asdict(
            cfg
        ),
    }

    save_json(
        provenance,
        run_dir
        / "provenance.json",
    )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    criterion = nn.SmoothL1Loss(
        beta=cfg.smooth_l1_beta
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
        )
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda"
        )
        if device.type == "cuda"
        else None
    )

    best_dev_mae = float("inf")
    best_epoch = None
    patience_count = 0

    history = []

    best_model_path = (
        run_dir
        / "best_model.pt"
    )

    training_start = (
        datetime.now()
    )

    print("=" * 80)
    print("Frozen ECG-Age ECG-age training")
    print("=" * 80)
    print(f"Device          : {device}")
    print(f"Train N         : {len(df_train):,}")
    print(f"DEV N           : {len(df_dev):,}")
    print(f"Age mean        : {age_mean:.6f}")
    print(f"Age std         : {age_std:.6f}")
    print(f"Parameters      : {n_params:,}")
    print(
        "Locked test     : NOT READ"
    )
    print(
        "Surgical cohort : NOT READ"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    for epoch in range(
        1,
        cfg.epochs + 1,
    ):
        epoch_start = (
            datetime.now()
        )

        (
            train_metrics,
            _,
            _,
            _,
        ) = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            age_mean=age_mean,
            age_std=age_std,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=cfg.grad_clip,
        )

        (
            dev_metrics,
            _,
            _,
            _,
        ) = run_epoch(
            model=model,
            loader=dev_loader,
            device=device,
            criterion=criterion,
            age_mean=age_mean,
            age_std=age_std,
            optimizer=None,
            scaler=None,
            grad_clip=cfg.grad_clip,
        )

        lr = float(
            optimizer
            .param_groups[0]["lr"]
        )

        history.append({
            "epoch": epoch,
            "lr": lr,

            "train_loss": (
                train_metrics["loss"]
            ),
            "train_mae": (
                train_metrics["mae"]
            ),
            "train_rmse": (
                train_metrics["rmse"]
            ),
            "train_r2": (
                train_metrics["r2"]
            ),

            "dev_loss": (
                dev_metrics["loss"]
            ),
            "dev_mae": (
                dev_metrics["mae"]
            ),
            "dev_rmse": (
                dev_metrics["rmse"]
            ),
            "dev_r2": (
                dev_metrics["r2"]
            ),
        })

        improved = (
            dev_metrics["mae"]
            < (
                best_dev_mae
                - cfg.min_delta
            )
        )

        if improved:
            best_dev_mae = float(
                dev_metrics["mae"]
            )

            best_epoch = int(
                epoch
            )

            patience_count = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "model_name": (
                        "ECG-Age"
                    ),

                    "epoch": (
                        best_epoch
                    ),

                    "best_dev_mae": (
                        best_dev_mae
                    ),

                    "age_mean": (
                        age_mean
                    ),

                    "age_std": (
                        age_std
                    ),

                    "input_channels": (
                        cfg.input_channels
                    ),

                    "trainable_parameters": (
                        n_params
                    ),

                    "preprocessing": {
                        "wfdb_signal": (
                            "p_signal"
                        ),
                        "normalization": (
                            "mean_center_only"
                        ),
                        "include_raw": True,
                        "use_derivative": True,
                        "scale_augmentation": False,
                    },

                    "config": (
                        asdict(cfg)
                    ),

                    "split_manifest_sha256": (
                        split_hash
                    ),

                    "training_script_sha256": (
                        sha256_file(
                            current_script
                        )
                    ),
                },
                best_model_path,
            )

        else:
            patience_count += 1

        scheduler.step()

        pd.DataFrame(
            history
        ).to_csv(
            run_dir
            / "history.csv",
            index=False,
        )

        elapsed = (
            datetime.now()
            - epoch_start
        ).total_seconds()

        print(
            f"\nEpoch {epoch:03d}/{cfg.epochs}\n"
            f"----------------------------------------\n"
            f"Train | "
            f"Loss {train_metrics['loss']:.4f} | "
            f"MAE {train_metrics['mae']:.3f} | "
            f"RMSE {train_metrics['rmse']:.3f} | "
            f"R2 {train_metrics['r2']:.3f}\n"
            f"Dev   | "
            f"Loss {dev_metrics['loss']:.4f} | "
            f"MAE {dev_metrics['mae']:.3f} | "
            f"RMSE {dev_metrics['rmse']:.3f} | "
            f"R2 {dev_metrics['r2']:.3f}\n"
            f"LR {lr:.8f} | "
            f"Best Dev MAE {best_dev_mae:.3f} | "
            f"Patience "
            f"{patience_count}/{cfg.patience} | "
            f"Time {elapsed:.1f}s",
            flush=True,
        )

        if (
            patience_count
            >= cfg.patience
        ):
            print(
                f"\nEarly stopping at "
                f"epoch {epoch}; "
                f"best epoch="
                f"{best_epoch}"
            )
            break

    if (
        best_epoch is None
        or not best_model_path.exists()
    ):
        raise RuntimeError(
            "No valid DEV checkpoint saved."
        )

    # ------------------------------------------------------------------
    # Reload selected DEV checkpoint
    # ------------------------------------------------------------------

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    (
        final_dev,
        y_dev,
        p_dev,
        sid_dev,
    ) = run_epoch(
        model=model,
        loader=dev_loader,
        device=device,
        criterion=criterion,
        age_mean=age_mean,
        age_std=age_std,
        optimizer=None,
        scaler=None,
        grad_clip=cfg.grad_clip,
    )

    predictions = pd.DataFrame({
        "subject_id": sid_dev,
        "age_true": y_dev,
        "age_pred": p_dev,
    })

    predictions["age_gap"] = (
        predictions["age_pred"]
        - predictions["age_true"]
    )

    predictions["abs_error"] = (
        np.abs(
            predictions["age_gap"]
        )
    )

    predictions.to_csv(
        run_dir
        / "dev_predictions.csv",
        index=False,
    )

    runtime_seconds = (
        datetime.now()
        - training_start
    ).total_seconds()

    summary = {
        "status": (
            "DEVELOPMENT_TRAINING_COMPLETE"
        ),

        "model": "ECG-Age",
        "seed": cfg.seed,

        "best_epoch": (
            best_epoch
        ),

        "best_dev_mae": (
            best_dev_mae
        ),

        "best_checkpoint_dev_metrics": (
            final_dev
        ),

        "train_n": int(
            len(df_train)
        ),

        "dev_n": int(
            len(df_dev)
        ),

        "runtime_seconds": (
            runtime_seconds
        ),

        "locked_test_read": False,
        "surgical_cohort_read": False,

        "best_model": str(
            best_model_path
        ),
    }

    save_json(
        summary,
        run_dir
        / "summary.json",
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best epoch    : {best_epoch}")
    print(
        f"DEV MAE       : "
        f"{final_dev['mae']:.3f}"
    )
    print(
        f"DEV RMSE      : "
        f"{final_dev['rmse']:.3f}"
    )
    print(
        f"DEV R2        : "
        f"{final_dev['r2']:.3f}"
    )
    print(f"Saved to      : {run_dir}")


if __name__ == "__main__":
    main()
