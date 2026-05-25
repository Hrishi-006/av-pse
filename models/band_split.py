"""Band-split projection module for AV-PSE.

Usage example:
    >>> import torch
    >>> from models.band_split import BandSplit
    >>> spec = torch.randn(2, 257, 376, dtype=torch.complex64)
    >>> module = BandSplit(feat_dim=128)
    >>> out = module(spec)
    >>> out.shape
    torch.Size([2, 32, 376, 128])
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn


class BandSplit(nn.Module):
    """Split a complex spectrogram into subbands and project each band.

    Input shape:
        ``spec``: ``[B, F=257, T]`` complex tensor.

    Output shape:
        ``[B, K=32, T, N=feat_dim]`` real tensor.
    """

    BAND_WIDTHS: ClassVar[tuple[int, ...]] = tuple([4] * 20 + [12] * 10 + [28, 29])

    def __init__(self, feat_dim: int = 128) -> None:
        """Initialize per-band normalization and projection layers.

        Args:
            feat_dim: Output feature dimension ``N`` for each band.
        """
        super().__init__()
        if sum(self.BAND_WIDTHS) != 257:
            raise ValueError(f"Band widths must sum to 257, got {sum(self.BAND_WIDTHS)}")

        self.num_bands = len(self.BAND_WIDTHS)
        self.feat_dim = feat_dim

        boundaries = [0]
        for width in self.BAND_WIDTHS:
            boundaries.append(boundaries[-1] + width)
        self.band_boundaries = boundaries

        self.norms = nn.ModuleList([nn.LayerNorm(2 * width) for width in self.BAND_WIDTHS])
        self.projections = nn.ModuleList([nn.Linear(2 * width, feat_dim) for width in self.BAND_WIDTHS])

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Project a complex spectrogram into band features.

        Args:
            spec: Complex spectrogram with shape ``[B, F=257, T]``.

        Returns:
            Real band features with shape ``[B, K=32, T, N=feat_dim]``.
        """
        if spec.dim() != 3:
            raise AssertionError(f"Expected spec shape [B, F, T], got {tuple(spec.shape)}")
        batch_size, freq_bins, time_frames = spec.shape
        if freq_bins != 257:
            raise AssertionError(f"Expected F=257, got F={freq_bins}")
        if not spec.is_complex():
            raise AssertionError("Input must be complex")

        outputs: list[torch.Tensor] = []
        for band_idx, (norm, projection) in enumerate(zip(self.norms, self.projections)):
            start = self.band_boundaries[band_idx]
            end = self.band_boundaries[band_idx + 1]

            # [B, band_width, T] complex
            band = spec[:, start:end, :]

            # [B, band_width, T] complex -> [B, 2 * band_width, T] real
            band_real = torch.cat([band.real, band.imag], dim=1)

            # [B, 2 * band_width, T] -> [B, T, 2 * band_width]
            band_real = band_real.transpose(1, 2)

            # [B, T, 2 * band_width] -> [B, T, N]
            band_feat = projection(norm(band_real))
            outputs.append(band_feat)

        out = torch.stack(outputs, dim=1)
        expected_shape = (batch_size, self.num_bands, time_frames, self.feat_dim)
        if out.shape != expected_shape:
            raise RuntimeError(f"Expected output shape {expected_shape}, got {tuple(out.shape)}")
        return out

