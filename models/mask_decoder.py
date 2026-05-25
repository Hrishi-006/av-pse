"""Complex mask and residual decoder for AV-PSE.

Usage example:
    >>> import torch
    >>> from models.mask_decoder import MaskDecoder
    >>> features = torch.randn(2, 32, 376, 128)
    >>> noisy_spec = torch.randn(2, 257, 376, dtype=torch.complex64)
    >>> decoder = MaskDecoder(feat_dim=128)
    >>> enhanced = decoder(features, noisy_spec)
    >>> enhanced.shape
    torch.Size([2, 257, 376])
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn


BAND_WIDTHS: tuple[int, ...] = tuple([4] * 20 + [12] * 10 + [28, 29])


class BandMaskMLP(nn.Module):
    """Per-band MLP that predicts a 1-tap complex mask for one band.

    Input shape:
        ``[B, T, N]``

    Output shape:
        ``[B, T, band_width, 2]`` where the final dimension is
        ``[real, imag]``.
    """

    def __init__(self, feat_dim: int, band_width: int) -> None:
        """Initialize one band-specific mask MLP.

        Args:
            feat_dim: Feature dimension ``N``.
            band_width: Number of frequency bins in this band.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.band_width = band_width
        self.linear_in = nn.Linear(feat_dim, 4 * feat_dim)
        self.linear_out = nn.Linear(4 * feat_dim, 4 * band_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict real and imaginary mask values for one band.

        Args:
            x: Real band features with shape ``[B, T, N]``.

        Returns:
            Real tensor with shape ``[B, T, band_width, 2]``.
        """
        if x.dim() != 3:
            raise AssertionError(f"Expected [B, T, N], got {tuple(x.shape)}")
        batch_size, time_frames, feat_dim = x.shape
        if feat_dim != self.feat_dim:
            raise AssertionError(f"Expected N={self.feat_dim}, got N={feat_dim}")

        out = self.linear_in(x)
        out = torch.tanh(out)
        out = self.linear_out(out)
        out = F.glu(out, dim=-1)
        return out.reshape(batch_size, time_frames, self.band_width, 2)


class ResidualMLP(nn.Module):
    """Global MLP that predicts a complex residual spectrogram.

    Input shape:
        ``[B, K, T, N]``

    Output shape:
        ``[B, F, T]`` complex.
    """

    def __init__(self, num_bands: int, feat_dim: int, num_freq: int) -> None:
        """Initialize the residual prediction MLP.

        Args:
            num_bands: Number of frequency bands ``K``.
            feat_dim: Feature dimension ``N``.
            num_freq: Number of one-sided STFT frequency bins ``F``.
        """
        super().__init__()
        self.num_bands = num_bands
        self.feat_dim = feat_dim
        self.num_freq = num_freq
        input_dim = num_bands * feat_dim
        self.linear_in = nn.Linear(input_dim, 4 * num_freq)
        self.linear_out = nn.Linear(4 * num_freq, 4 * num_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a complex residual spectrogram.

        Args:
            x: Real features with shape ``[B, K, T, N]``.

        Returns:
            Complex residual with shape ``[B, F, T]``.
        """
        if x.dim() != 4:
            raise AssertionError(f"Expected [B, K, T, N], got {tuple(x.shape)}")
        batch_size, num_bands, time_frames, feat_dim = x.shape
        if num_bands != self.num_bands:
            raise AssertionError(f"Expected K={self.num_bands}, got K={num_bands}")
        if feat_dim != self.feat_dim:
            raise AssertionError(f"Expected N={self.feat_dim}, got N={feat_dim}")

        out = x.permute(0, 2, 1, 3).reshape(batch_size, time_frames, num_bands * feat_dim)
        out = self.linear_in(out)
        out = torch.tanh(out)
        out = self.linear_out(out)
        out = F.glu(out, dim=-1)
        out = out.reshape(batch_size, time_frames, self.num_freq, 2)
        out_complex = torch.complex(out[..., 0], out[..., 1])
        return out_complex.permute(0, 2, 1)


class MaskDecoder(nn.Module):
    """Module C: complex mask plus residual spectrogram decoder.

    The enhanced spectrogram is computed as ``enhanced = M * X + R``.

    Input shapes:
        ``features``: ``[B, K=32, T, N]``
        ``noisy_spec``: ``[B, F=257, T]`` complex

    Output shape:
        ``[B, F=257, T]`` complex
    """

    BAND_WIDTHS: ClassVar[tuple[int, ...]] = BAND_WIDTHS
    NUM_FREQ: ClassVar[int] = 257

    def __init__(self, feat_dim: int = 128) -> None:
        """Initialize per-band mask MLPs and global residual MLP."""
        super().__init__()
        if sum(self.BAND_WIDTHS) != self.NUM_FREQ:
            raise ValueError(f"Band widths must sum to {self.NUM_FREQ}, got {sum(self.BAND_WIDTHS)}")

        self.num_bands = len(self.BAND_WIDTHS)
        self.feat_dim = feat_dim

        self.mask_mlps = nn.ModuleList(
            [BandMaskMLP(feat_dim=feat_dim, band_width=width) for width in self.BAND_WIDTHS]
        )
        self.residual_mlp = ResidualMLP(
            num_bands=self.num_bands,
            feat_dim=feat_dim,
            num_freq=self.NUM_FREQ,
        )

    def forward(self, features: torch.Tensor, noisy_spec: torch.Tensor) -> torch.Tensor:
        """Decode processed band features into an enhanced spectrogram.

        Args:
            features: Real features with shape ``[B, K, T, N]``.
            noisy_spec: Complex noisy spectrogram with shape ``[B, F, T]``.

        Returns:
            Complex enhanced spectrogram with shape ``[B, F, T]``.
        """
        if features.dim() != 4:
            raise AssertionError(f"Expected features [B, K, T, N], got {tuple(features.shape)}")
        batch_size, num_bands, time_frames, feat_dim = features.shape
        if num_bands != self.num_bands:
            raise AssertionError(f"K={num_bands} != {self.num_bands}")
        if feat_dim != self.feat_dim:
            raise AssertionError(f"N={feat_dim} != {self.feat_dim}")
        if noisy_spec.dim() != 3:
            raise AssertionError(f"Expected noisy_spec [B, F, T], got {tuple(noisy_spec.shape)}")
        if not noisy_spec.is_complex():
            raise AssertionError("noisy_spec must be complex")
        if noisy_spec.shape != (batch_size, self.NUM_FREQ, time_frames):
            raise AssertionError(
                "Expected noisy_spec shape "
                f"{(batch_size, self.NUM_FREQ, time_frames)}, got {tuple(noisy_spec.shape)}"
            )

        mask_band_outputs: list[torch.Tensor] = []
        for band_idx, mlp in enumerate(self.mask_mlps):
            band_feat = features[:, band_idx, :, :]
            band_mask = mlp(band_feat)
            mask_band_outputs.append(band_mask)

        mask = torch.cat(mask_band_outputs, dim=2)
        mask = torch.complex(mask[..., 0], mask[..., 1])
        mask = mask.permute(0, 2, 1)

        residual = self.residual_mlp(features)
        enhanced = mask * noisy_spec + residual
        return enhanced

