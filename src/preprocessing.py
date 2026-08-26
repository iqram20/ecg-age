import numpy as np
import torch
import torch.nn.functional as F


TARGET_LENGTH = 5000
N_PHYSICAL_LEADS = 12
N_INPUT_CHANNELS = 24


def mean_center_per_lead(signal: torch.Tensor) -> torch.Tensor:
    """
    Mean-center each physical ECG lead independently.

    The frozen final preprocessing preserves physical amplitude:
    no division by per-lead standard deviation is performed.
    """
    return signal - signal.mean(dim=1, keepdim=True)


def prepare_frozen_channels(raw_signal: torch.Tensor) -> torch.Tensor:
    """
    Frozen manuscript preprocessing.

    Input
    -----
    raw_signal : Tensor [12, T]
        WFDB physical ECG signal (p_signal).

    Output
    ------
    Tensor [24, T]
        12 mean-centered physical leads followed by their
        first temporal derivatives.
    """
    if raw_signal.ndim != 2:
        raise ValueError(
            f"Expected [leads, time], got shape {tuple(raw_signal.shape)}"
        )

    if raw_signal.shape[0] != N_PHYSICAL_LEADS:
        raise ValueError(
            f"Expected 12 physical leads, got {raw_signal.shape[0]}"
        )

    processed = mean_center_per_lead(raw_signal)

    derivative = torch.diff(
        processed,
        dim=1,
        prepend=processed[:, :1],
    )

    signal = torch.cat(
        [processed, derivative],
        dim=0,
    )

    if signal.shape[0] != N_INPUT_CHANNELS:
        raise RuntimeError(
            f"Prepared {signal.shape[0]} channels; expected 24."
        )

    return signal


def fix_length(
    signal: torch.Tensor,
    target_len: int = TARGET_LENGTH,
    augment: bool = False,
) -> torch.Tensor:
    """
    Crop or pad ECG to the fixed 5000-sample input length.

    Training:
        random crop when longer than target length.

    Evaluation:
        centered crop when longer than target length.

    Short signals are zero-padded on the right.
    """
    T = signal.shape[-1]

    if T > target_len:
        max_start = T - target_len

        if augment:
            crop_start = np.random.randint(
                0,
                max_start + 1,
            )
        else:
            crop_start = max_start // 2

        signal = signal[
            :,
            crop_start:crop_start + target_len
        ]

    elif T < target_len:
        signal = F.pad(
            signal,
            (0, target_len - T),
        )

    return signal


def augment_frozen_ecg(signal: torch.Tensor) -> torch.Tensor:
    """
    Training augmentation used by the frozen final ECG-Age model.

    - temporal roll: p=0.5, -40..+40 samples
    - additive Gaussian noise: p=0.5, sigma=0.01
    - temporal masking: p=0.3, 100..400 samples
    - paired physical-lead dropout: p=0.3

    Global amplitude scaling is intentionally NOT used.
    """

    # Temporal translation.
    if np.random.rand() < 0.5:
        signal = torch.roll(
            signal,
            shifts=np.random.randint(-40, 41),
            dims=-1,
        )

    # No global amplitude-scale augmentation.

    # Additive Gaussian noise.
    if np.random.rand() < 0.5:
        signal = (
            signal
            + torch.randn_like(signal) * 0.01
        )

    # Temporal masking.
    if np.random.rand() < 0.3:
        mask_len = np.random.randint(100, 401)

        if signal.shape[-1] > mask_len:
            start = np.random.randint(
                0,
                signal.shape[-1] - mask_len + 1,
            )

            signal[
                :,
                start:start + mask_len
            ] = 0

    # Physical-lead dropout:
    # paired raw lead + corresponding derivative.
    if np.random.rand() < 0.3:
        lead_idx = np.random.randint(
            0,
            N_PHYSICAL_LEADS,
        )

        signal[lead_idx] = 0
        signal[
            lead_idx + N_PHYSICAL_LEADS
        ] = 0

    return signal
