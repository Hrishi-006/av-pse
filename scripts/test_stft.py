"""STFT helpers for AV-PSE waveform and spectrogram conversion.

Usage:
    >>> import torch
    >>> from models.stft import compute_istft, compute_stft
    >>> wav = torch.randn(2, 48000)
    >>> spec = compute_stft(wav)
    >>> recon = compute_istft(spec, length=wav.shape[-1])
"""

from __future__ import annotations

import torch


def compute_stft(
    waveform: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
) -> torch.Tensor:
    """
    Convert real waveform to complex spectrogram.

    Args:
        waveform: shape [B, num_samples] or [num_samples], real tensor.
                  Will be unsqueezed to [B, num_samples] if needed.

    Returns:
        Complex spectrogram of shape [B, F, T] where F = n_fft // 2 + 1.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )


def compute_istft(
    spectrogram: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    length: int | None = None,
) -> torch.Tensor:
    """
    Convert complex spectrogram back to real waveform.

    Args:
        spectrogram: shape [B, F, T] complex.
        length: if given, crop output to exactly this many samples.

    Returns:
        Real waveform of shape [B, num_samples].
    """
    window_dtype = torch.float64 if spectrogram.dtype == torch.complex128 else torch.float32
    window = torch.hann_window(win_length, device=spectrogram.device, dtype=window_dtype)
    return torch.istft(
        spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )


def get_stft_frame_count(num_samples: int, hop_length: int = 128) -> int:
    """
    Compute expected number of STFT frames for center=True padding.

    For center=True: frame_count = num_samples // hop_length + 1.
    """
    return num_samples // hop_length + 1



wav = torch.randn(1, 48000)
spec = compute_stft(wav)
print(spec.shape)  # should be torch.Size([1, 257, 376])
print(spec.dtype)  # should be torch.complex64