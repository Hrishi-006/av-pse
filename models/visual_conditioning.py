"""Visual conditioning module for AV-PSE.

Usage example:
    >>> import torch
    >>> from models.visual_conditioning import VisualConditioningModule
    >>> landmarks = torch.randn(2, 75, 40, 3)
    >>> module = VisualConditioningModule()
    >>> out = module(landmarks)
    >>> out.shape
    torch.Size([2, 32, 373, 128])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VisualConditioningModule(nn.Module):
    """Produce per-band conditioning features from lip landmarks.

    This module replaces the paper's audio-based speaker enrollment module.

    Input shape:
        ``[B, T_video, num_landmarks, coord_dim]``

    Output shape:
        ``[B, K=num_bands, T_audio=target_audio_frames, N=feat_dim]``
    """

    def __init__(
        self,
        num_bands: int = 32,
        feat_dim: int = 128,
        num_landmarks: int = 40,
        coord_dim: int = 3,
        hidden_dim: int = 128,
        target_audio_frames: int = 373,
        upsample_factor: int = 5,
        use_motion_deltas: bool = True,
    ) -> None:
        """Initialize the visual conditioning module.

        Args:
            num_bands: Number of frequency bands ``K``.
            feat_dim: Output feature dimension ``N`` per band.
            num_landmarks: Number of lip landmarks per frame.
            coord_dim: Coordinate dimension per landmark.
            hidden_dim: Internal temporal feature dimension.
            target_audio_frames: Target audio-frame count ``T_audio``.
            upsample_factor: Linear interpolation factor from video time.
            use_motion_deltas: If true, concatenate positions and deltas.
        """
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if num_landmarks <= 0:
            raise ValueError("num_landmarks must be positive")
        if coord_dim <= 0:
            raise ValueError("coord_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if target_audio_frames <= 0:
            raise ValueError("target_audio_frames must be positive")
        if upsample_factor <= 0:
            raise ValueError("upsample_factor must be positive")

        self.num_bands = num_bands
        self.feat_dim = feat_dim
        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.target_audio_frames = target_audio_frames
        self.upsample_factor = upsample_factor
        self.use_motion_deltas = use_motion_deltas

        input_coord_dim = coord_dim * (2 if use_motion_deltas else 1)
        flat_input_dim = num_landmarks * input_coord_dim

        self.input_projection = nn.Linear(flat_input_dim, hidden_dim)
        self.temporal_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.temporal_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.band_projections = nn.ModuleList([nn.Linear(hidden_dim, feat_dim) for _ in range(num_bands)])

    def _compute_deltas(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Compute frame-to-frame motion deltas.

        Args:
            landmarks: Tensor with shape ``[B, T, L, C]``.

        Returns:
            Tensor with shape ``[B, T, L, C]``. The first frame delta is zero.
        """
        if landmarks.dim() != 4:
            raise AssertionError(f"Expected [B, T, L, C], got {tuple(landmarks.shape)}")
        deltas = torch.diff(landmarks, dim=1)
        zero_pad = torch.zeros_like(landmarks[:, :1])
        return torch.cat([zero_pad, deltas], dim=1)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Convert lip landmarks into per-band visual conditioning features.

        Args:
            landmarks: Real tensor with shape ``[B, T_video, num_landmarks, coord_dim]``.

        Returns:
            Real tensor with shape ``[B, num_bands, target_audio_frames, feat_dim]``.
        """
        if landmarks.dim() != 4:
            raise AssertionError(f"Expected [B, T_video, L, C], got {tuple(landmarks.shape)}")
        batch_size, video_frames, num_landmarks, coord_dim = landmarks.shape
        if num_landmarks != self.num_landmarks:
            raise AssertionError(f"Expected {self.num_landmarks} landmarks, got {num_landmarks}")
        if coord_dim != self.coord_dim:
            raise AssertionError(f"Expected coord_dim={self.coord_dim}, got {coord_dim}")

        if self.use_motion_deltas:
            deltas = self._compute_deltas(landmarks)
            out = torch.cat([landmarks, deltas], dim=-1)
        else:
            out = landmarks

        # [B, T_video, L, C'] -> [B, T_video, L*C']
        out = out.reshape(batch_size, video_frames, -1)

        # [B, T_video, L*C'] -> [B, T_video, hidden_dim]
        out = self.input_projection(out)

        # [B, T_video, hidden_dim] -> [B, hidden_dim, T_video]
        out = out.transpose(1, 2)
        out = torch.tanh(self.temporal_conv1(out))
        out = torch.tanh(self.temporal_conv2(out))

        target_after_upsample = video_frames * self.upsample_factor
        out = F.interpolate(
            out,
            size=target_after_upsample,
            mode="linear",
            align_corners=False,
        )

        if self.target_audio_frames > target_after_upsample:
            pad_amount = self.target_audio_frames - target_after_upsample
            out = F.pad(out, (0, pad_amount), mode="replicate")
        elif self.target_audio_frames < target_after_upsample:
            out = out[..., : self.target_audio_frames]

        # [B, hidden_dim, T_audio] -> [B, T_audio, hidden_dim]
        out = out.transpose(1, 2)

        band_outputs: list[torch.Tensor] = []
        for projection in self.band_projections:
            band_feat = torch.tanh(projection(out))
            band_outputs.append(band_feat)

        return torch.stack(band_outputs, dim=1)

