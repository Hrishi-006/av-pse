"""Multi-resolution spectral loss for AV-PSE (paper Section 3).

Two terms are averaged over R STFT resolutions:
  1. Power-compressed magnitude L1:  E[ ||Ŝ|^p - |S|^p| ],  p=0.3
  2. Complex L1:                     E[ |Ŝ_r - S_r| ]       (mean over real+imag)

Usage example:
    >>> import torch
    >>> from losses.multi_res_loss import MultiResolutionLoss
    >>> loss_fn = MultiResolutionLoss(sample_rate=16000)
    >>> enhanced = torch.randn(2, 47648)
    >>> clean    = torch.randn(2, 47648)
    >>> loss = loss_fn(enhanced, clean)
    >>> loss.shape
    torch.Size([])
"""

from __future__ import annotations

import torch
from torch import nn


class MultiResolutionLoss(nn.Module):
    """Average of magnitude + complex L1 losses across R STFT resolutions.

    Args:
        sample_rate: Audio sample rate in Hz (used to convert ms → samples).
        window_ms: STFT window lengths in milliseconds. Default: [10, 20, 30, 40].
        power: Exponent for magnitude compression. Default: 0.3.
        hop_divisor: ``hop = n_fft // hop_divisor`` for each resolution. Default: 4.
        eps: Small constant added before power compression for numerical stability.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_ms: list[int] | None = None,
        power: float = 0.3,
        hop_divisor: int = 4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if window_ms is None:
            window_ms = [10, 20, 30, 40]
        if not window_ms:
            raise ValueError("window_ms must be non-empty")
        if not (0.0 < power <= 1.0):
            raise ValueError(f"power must be in (0, 1], got {power}")
        if hop_divisor < 1:
            raise ValueError(f"hop_divisor must be >= 1, got {hop_divisor}")

        self.sample_rate = sample_rate
        self.power = power
        self.hop_divisor = hop_divisor
        self.eps = eps

        # Convert ms → integer sample counts
        self.n_ffts: list[int] = [round(ms * sample_rate / 1000) for ms in window_ms]

    def _stft(self, waveform: torch.Tensor, n_fft: int) -> torch.Tensor:
        """Compute a complex STFT for one resolution.

        Args:
            waveform: ``[B, S]`` real tensor.
            n_fft: Window length in samples.

        Returns:
            Complex spectrogram ``[B, F, T]``.
        """
        hop = n_fft // self.hop_divisor
        window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

    def _loss_one_resolution(
        self,
        enhanced_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        n_fft: int,
    ) -> torch.Tensor:
        """Compute the combined loss at a single STFT resolution.

        Args:
            enhanced_wav: ``[B, S]`` real tensor.
            clean_wav: ``[B, S]`` real tensor.
            n_fft: STFT window length in samples.

        Returns:
            Scalar loss tensor.
        """
        s_hat = self._stft(enhanced_wav, n_fft)  # [B, F, T] complex
        s_ref = self._stft(clean_wav, n_fft)      # [B, F, T] complex

        # Power-compressed magnitude L1
        mag_hat = s_hat.abs().clamp(min=self.eps).pow(self.power)
        mag_ref = s_ref.abs().clamp(min=self.eps).pow(self.power)
        mag_loss = (mag_hat - mag_ref).abs().mean()

        # Complex L1 — treat real and imag as two channels
        complex_diff = s_hat - s_ref
        complex_loss = (complex_diff.real.abs() + complex_diff.imag.abs()).mean()

        return mag_loss + complex_loss

    def forward(self, enhanced_wav: torch.Tensor, clean_wav: torch.Tensor) -> torch.Tensor:
        """Compute the multi-resolution loss.

        Args:
            enhanced_wav: Model output waveform ``[B, S]`` real tensor.
            clean_wav: Ground-truth clean waveform ``[B, S]`` real tensor.

        Returns:
            Scalar loss tensor averaged over all resolutions.
        """
        if enhanced_wav.shape != clean_wav.shape:
            raise ValueError(
                f"Shape mismatch: enhanced {tuple(enhanced_wav.shape)} "
                f"vs clean {tuple(clean_wav.shape)}"
            )
        if enhanced_wav.dim() != 2:
            raise ValueError(f"Expected [B, S] input, got {tuple(enhanced_wav.shape)}")

        total = torch.zeros((), device=enhanced_wav.device, dtype=enhanced_wav.dtype)
        for n_fft in self.n_ffts:
            total = total + self._loss_one_resolution(enhanced_wav, clean_wav, n_fft)

        return total / len(self.n_ffts)
