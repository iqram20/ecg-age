from pathlib import Path

import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from .preprocessing import (
    TARGET_LENGTH,
    prepare_frozen_channels,
    fix_length,
    augment_frozen_ecg,
)


class MIMICECGAgeDataset(Dataset):
    """
    Dataset used for the frozen final ECG-age ECG-Age model.

    Expected dataframe columns:
        path
        age
        subject_id

    ECG waveforms are read from WFDB p_signal.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_path,
        target_len: int = TARGET_LENGTH,
        augment: bool = False,
    ):
        self.df = (
            df.reset_index(drop=True)
            .copy()
        )

        self.base_path = Path(base_path)
        self.target_len = int(target_len)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _normalize_record_path(record_path):
        record_path = str(record_path).strip()

        if (
            record_path.endswith(".dat")
            or record_path.endswith(".hea")
        ):
            record_path = record_path[:-4]

        record_path = record_path.lstrip("./")

        if record_path.startswith("files/"):
            record_path = record_path[
                len("files/"):
            ]

        return record_path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        record_path = self._normalize_record_path(
            row["path"]
        )

        full_path = (
            self.base_path
            / record_path
        )

        record = wfdb.rdrecord(
            str(full_path)
        )

        # Frozen pipeline uses WFDB physical signal.
        raw_signal = torch.tensor(
            record.p_signal.T,
            dtype=torch.float32,
        )

        raw_signal = torch.nan_to_num(
            raw_signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if raw_signal.ndim != 2:
            raise RuntimeError(
                f"Unexpected signal ndim="
                f"{raw_signal.ndim} for {full_path}"
            )

        if raw_signal.shape[0] != 12:
            raise RuntimeError(
                f"Expected 12 leads, found "
                f"{raw_signal.shape[0]} "
                f"for {full_path}"
            )

        signal = prepare_frozen_channels(
            raw_signal
        )

        signal = fix_length(
            signal,
            target_len=self.target_len,
            augment=self.augment,
        )

        if self.augment:
            signal = augment_frozen_ecg(
                signal
            )

        age = torch.tensor(
            float(row["age"]),
            dtype=torch.float32,
        )

        subject_id = torch.tensor(
            int(row["subject_id"]),
            dtype=torch.long,
        )

        return signal, age, subject_id
