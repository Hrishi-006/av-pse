"""Band and sequence recurrent modeling for AV-PSE.

Usage example:
    >>> import torch
    >>> from models.band_sequence_rnn import BandSequenceRNN
    >>> x = torch.randn(2, 32, 376, 128)
    >>> model = BandSequenceRNN(feat_dim=128, num_low_bands=30, num_high_bands=2)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 32, 376, 128])
"""

from __future__ import annotations

import torch
from torch import nn


class TimeBlock(nn.Module):
    """Residual time-axis BLSTM block.

    Each band is processed independently along the time axis.

    Input shape:
        ``[B, K, T, N]``

    Output shape:
        ``[B, K, T, N]``
    """

    def __init__(self, feat_dim: int) -> None:
        """Initialize the time-axis recurrent block.

        Args:
            feat_dim: Feature dimension ``N``.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.norm = nn.LayerNorm(feat_dim)
        self.rnn = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * feat_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual time-axis BLSTM modeling.

        Args:
            x: Tensor with shape ``[B, K, T, N]``.

        Returns:
            Tensor with shape ``[B, K, T, N]``.
        """
        if x.dim() != 4:
            raise AssertionError(f"Expected [B, K, T, N], got {tuple(x.shape)}")
        batch_size, num_bands, time_frames, feat_dim = x.shape
        if feat_dim != self.feat_dim:
            raise AssertionError(f"Expected N={self.feat_dim}, got N={feat_dim}")

        residual = x
        out = x.reshape(batch_size * num_bands, time_frames, feat_dim)
        out = self.norm(out)
        out, _ = self.rnn(out)
        out = self.proj(out)
        out = out.reshape(batch_size, num_bands, time_frames, feat_dim)
        return out + residual


class BandBlockBSRNNS(nn.Module):
    """Residual band-axis block with BSRNN-S directional asymmetry.

    Low-frequency bands are processed with a BLSTM. High-frequency bands are
    processed with a unidirectional LSTM initialized from the low-frequency
    BLSTM's forward final hidden and cell state.

    Input shape:
        ``[B, K, T, N]`` where ``K = num_low_bands + num_high_bands``.

    Output shape:
        ``[B, K, T, N]``.
    """

    def __init__(self, feat_dim: int, num_low_bands: int, num_high_bands: int) -> None:
        """Initialize the BSRNN-S band-axis block.

        Args:
            feat_dim: Feature dimension ``N``.
            num_low_bands: Number of low-frequency bands handled by BLSTM.
            num_high_bands: Number of high-frequency bands handled by LSTM.
        """
        super().__init__()
        if num_low_bands <= 0:
            raise ValueError("num_low_bands must be positive")
        if num_high_bands <= 0:
            raise ValueError("num_high_bands must be positive")

        self.num_low_bands = num_low_bands
        self.num_high_bands = num_high_bands
        self.num_bands = num_low_bands + num_high_bands
        self.feat_dim = feat_dim

        self.norm = nn.LayerNorm(feat_dim)
        self.blstm_low = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj_low = nn.Linear(2 * feat_dim, feat_dim)
        self.lstm_high = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual BSRNN-S band-axis modeling.

        Args:
            x: Tensor with shape ``[B, K, T, N]``.

        Returns:
            Tensor with shape ``[B, K, T, N]``.
        """
        if x.dim() != 4:
            raise AssertionError(f"Expected [B, K, T, N], got {tuple(x.shape)}")
        batch_size, num_bands, time_frames, feat_dim = x.shape
        if num_bands != self.num_bands:
            raise AssertionError(f"K={num_bands} but expected {self.num_bands}")
        if feat_dim != self.feat_dim:
            raise AssertionError(f"Expected N={self.feat_dim}, got N={feat_dim}")

        residual = x

        # [B, K, T, N] -> [B, T, K, N] -> [B*T, K, N]
        out = x.permute(0, 2, 1, 3).reshape(batch_size * time_frames, num_bands, feat_dim)
        out = self.norm(out)

        low = out[:, : self.num_low_bands, :]
        high = out[:, self.num_low_bands :, :]

        low_out, (h_n, c_n) = self.blstm_low(low)

        # h_n/c_n: [2, B*T, N]. Index 0 is the forward direction.
        h_fwd = h_n[0:1, :, :].contiguous()
        c_fwd = c_n[0:1, :, :].contiguous()

        high_out, _ = self.lstm_high(high, (h_fwd, c_fwd))
        low_out = self.proj_low(low_out)

        combined = torch.cat([low_out, high_out], dim=1)
        combined = combined.reshape(batch_size, time_frames, num_bands, feat_dim).permute(0, 2, 1, 3)
        return combined + residual


class BandSequenceLayer(nn.Module):
    """One Module B layer: time-axis block followed by BSRNN-S band block."""

    def __init__(self, feat_dim: int, num_low_bands: int, num_high_bands: int) -> None:
        """Initialize one interleaved time/band modeling layer."""
        super().__init__()
        self.time_block = TimeBlock(feat_dim)
        self.band_block = BandBlockBSRNNS(
            feat_dim=feat_dim,
            num_low_bands=num_low_bands,
            num_high_bands=num_high_bands,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time-axis then band-axis residual modeling.

        Args:
            x: Tensor with shape ``[B, K, T, N]``.

        Returns:
            Tensor with shape ``[B, K, T, N]``.
        """
        out = self.time_block(x)
        out = self.band_block(out)
        return out


class BandSequenceRNN(nn.Module):
    """Stacked Module B with BSRNN-S asymmetric band-axis split.

    Input shape:
        ``[B, K, T, N]``

    Output shape:
        ``[B, K, T, N]``

    For visual conditioning, the caller concatenates audio and visual features
    before this module, so ``feat_dim`` can be ``2N``.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_low_bands: int = 30,
        num_high_bands: int = 2,
        num_layers: int = 6,
    ) -> None:
        """Initialize stacked band/sequence recurrent layers.

        Args:
            feat_dim: Feature dimension ``N``.
            num_low_bands: Number of low-frequency bands.
            num_high_bands: Number of high-frequency bands.
            num_layers: Number of stacked time/band layers.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.feat_dim = feat_dim
        self.num_low_bands = num_low_bands
        self.num_high_bands = num_high_bands
        self.num_bands = num_low_bands + num_high_bands
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                BandSequenceLayer(
                    feat_dim=feat_dim,
                    num_low_bands=num_low_bands,
                    num_high_bands=num_high_bands,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all interleaved time/band layers.

        Args:
            x: Tensor with shape ``[B, K, T, N]``.

        Returns:
            Tensor with shape ``[B, K, T, N]``.
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

