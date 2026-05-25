"""Full Audio-Visual Band-Split RNN model.

Usage example:
    >>> import torch
    >>> from models.av_bsrnn import AVBSRNN, AVBSRNNConfig
    >>> config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    >>> model = AVBSRNN(config)
    >>> wav = torch.randn(1, 47648)
    >>> landmarks = torch.randn(1, 75, 40, 3)
    >>> out = model(wav, landmarks)
    >>> out["waveform"].shape, out["spectrogram"].shape
    (torch.Size([1, 47648]), torch.Size([1, 257, 373]))

Architecture:
    noisy waveform [B, S]
        |
        v
    STFT -> noisy spectrogram X [B, F=257, T]
        |
        +--> BandSplit ----------------------> audio features [B, K, T, N]
        |                                           |
        |                                           + concat on feature dim
        |                                           |
        +--> VisualConditioning(optional) ----> visual features [B, K, T, N]
                                                    |
                                                    v
                                  BandSequenceRNN input [B, K, T, N or 2N]
                                                    |
                                                    v
                                  processed features [B, K, T, N or 2N]
                                                    |
                                                    v
                                  MaskDecoder: S_hat = M * X + R
                                                    |
                                                    v
                                  enhanced spectrogram [B, F=257, T]
                                                    |
                                                    v
                                  ISTFT -> enhanced waveform [B, S]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

import torch
from torch import nn

from models.band_sequence_rnn import BandSequenceRNN
from models.band_split import BandSplit
from models.mask_decoder import MaskDecoder
from models.stft import compute_istft, compute_stft
from models.visual_conditioning import VisualConditioningModule


class AVBSRNNOutput(TypedDict):
    """Output dictionary returned by :class:`AVBSRNN`."""

    waveform: torch.Tensor
    spectrogram: torch.Tensor
    noisy_spectrogram: torch.Tensor


@dataclass(frozen=True)
class AVBSRNNConfig:
    """Configuration for the full AV-BSRNN model."""

    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    num_freq: int = 257
    feat_dim: int = 128
    num_bands: int = 32
    num_low_bands: int = 30
    num_high_bands: int = 2
    num_layers: int = 6
    use_visual_conditioning: bool = True
    num_landmarks: int = 40
    coord_dim: int = 3
    visual_hidden_dim: int = 128
    upsample_factor: int = 5
    use_motion_deltas: bool = True
    target_audio_frames: int = 373


class AVBSRNN(nn.Module):
    """Audio-Visual Band-Split RNN for personalized speech enhancement.

    If visual conditioning is enabled, landmarks are required and are
    concatenated with audio band features before Module B.

    Input shapes:
        ``noisy_waveform``: ``[B, num_samples]`` real tensor.
        ``landmarks``: ``[B, T_video, num_landmarks, coord_dim]`` real tensor
        when visual conditioning is enabled.

    Output shapes:
        ``waveform``: ``[B, num_samples]`` real tensor.
        ``spectrogram``: ``[B, F=257, T]`` complex tensor.
        ``noisy_spectrogram``: ``[B, F=257, T]`` complex tensor.
    """

    def __init__(self, config: AVBSRNNConfig) -> None:
        """Initialize the full model from a frozen config."""
        super().__init__()
        if config.num_freq != config.n_fft // 2 + 1:
            raise ValueError(f"num_freq={config.num_freq} must equal n_fft // 2 + 1")
        if config.num_bands != config.num_low_bands + config.num_high_bands:
            raise ValueError("num_bands must equal num_low_bands + num_high_bands")

        self.config = config
        rnn_feat_dim = 2 * config.feat_dim if config.use_visual_conditioning else config.feat_dim

        self.band_split = BandSplit(feat_dim=config.feat_dim)

        if config.use_visual_conditioning:
            self.visual_conditioning: VisualConditioningModule | None = VisualConditioningModule(
                num_bands=config.num_bands,
                feat_dim=config.feat_dim,
                num_landmarks=config.num_landmarks,
                coord_dim=config.coord_dim,
                hidden_dim=config.visual_hidden_dim,
                target_audio_frames=config.target_audio_frames,
                upsample_factor=config.upsample_factor,
                use_motion_deltas=config.use_motion_deltas,
            )
        else:
            self.visual_conditioning = None

        self.band_sequence_rnn = BandSequenceRNN(
            feat_dim=rnn_feat_dim,
            num_low_bands=config.num_low_bands,
            num_high_bands=config.num_high_bands,
            num_layers=config.num_layers,
        )
        self.mask_decoder = MaskDecoder(feat_dim=rnn_feat_dim)

    def forward(self, noisy_waveform: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> AVBSRNNOutput:
        """Run the complete AV-PSE forward pass.

        Args:
            noisy_waveform: Real waveform with shape ``[B, num_samples]``.
            landmarks: Optional real landmarks with shape
                ``[B, T_video, num_landmarks, coord_dim]``. Required when
                ``config.use_visual_conditioning`` is true.

        Returns:
            Dictionary containing enhanced waveform, enhanced spectrogram,
            and noisy spectrogram.
        """
        if noisy_waveform.dim() != 2:
            raise AssertionError(f"Expected noisy_waveform [B, S], got {tuple(noisy_waveform.shape)}")
        if self.config.use_visual_conditioning:
            assert landmarks is not None, "use_visual_conditioning is True but no landmarks given"

        batch_size, num_samples = noisy_waveform.shape

        noisy_spec = compute_stft(
            noisy_waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
        )

        audio_features = self.band_split(noisy_spec)
        if audio_features.shape[1] != self.config.num_bands:
            raise AssertionError(f"BandSplit produced K={audio_features.shape[1]}, expected {self.config.num_bands}")

        if self.config.use_visual_conditioning:
            assert self.visual_conditioning is not None
            visual_features = self.visual_conditioning(landmarks)
            if visual_features.shape[:3] != audio_features.shape[:3]:
                raise AssertionError(
                    "Audio/visual feature shape mismatch before concat: "
                    f"audio={tuple(audio_features.shape)}, visual={tuple(visual_features.shape)}"
                )
            features = torch.cat([audio_features, visual_features], dim=-1)
        else:
            features = audio_features

        processed = self.band_sequence_rnn(features)
        enhanced_spec = self.mask_decoder(processed, noisy_spec)
        enhanced_waveform = compute_istft(
            enhanced_spec,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            length=num_samples,
        )

        if enhanced_waveform.shape != (batch_size, num_samples):
            raise RuntimeError(f"Expected waveform shape {(batch_size, num_samples)}, got {tuple(enhanced_waveform.shape)}")

        return {
            "waveform": enhanced_waveform,
            "spectrogram": enhanced_spec,
            "noisy_spectrogram": noisy_spec,
        }

    def count_parameters(self) -> dict[str, int]:
        """Count trainable and non-trainable parameters by component."""
        counts = {
            "band_split": sum(param.numel() for param in self.band_split.parameters()),
            "visual_conditioning": 0,
            "band_sequence_rnn": sum(param.numel() for param in self.band_sequence_rnn.parameters()),
            "mask_decoder": sum(param.numel() for param in self.mask_decoder.parameters()),
        }
        if self.visual_conditioning is not None:
            counts["visual_conditioning"] = sum(param.numel() for param in self.visual_conditioning.parameters())
        counts["total"] = sum(counts.values())
        return counts

