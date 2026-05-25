"""Tests for STFT helpers.

Usage:
    pytest tests/test_stft.py -v
"""

from __future__ import annotations

import torch

from models.stft import compute_istft, compute_stft, get_stft_frame_count


def test_forward_shape() -> None:
    wav = torch.randn(2, 48000)
    spec = compute_stft(wav)
    assert spec.shape == (2, 257, 376), f"Got {spec.shape}"
    assert spec.is_complex()


def test_round_trip_preserves_waveform() -> None:
    wav = torch.randn(2, 48000)
    spec = compute_stft(wav)
    recon = compute_istft(spec, length=48000)
    assert recon.shape == (2, 48000)
    assert torch.allclose(wav, recon, atol=1e-4), f"Max error: {(wav - recon).abs().max()}"


def test_handles_1d_input_with_batch_dimension() -> None:
    wav_1d = torch.randn(48000)
    spec = compute_stft(wav_1d)
    assert spec.shape == (1, 257, 376)


def test_device_follows_input() -> None:
    if torch.cuda.is_available():
        wav = torch.randn(2, 48000).cuda()
        spec = compute_stft(wav)
        assert spec.device.type == "cuda"
        recon = compute_istft(spec, length=48000)
        assert recon.device.type == "cuda"


def test_frame_count_helper() -> None:
    assert get_stft_frame_count(48000, hop_length=128) == 376
    assert get_stft_frame_count(32000, hop_length=128) == 251


def test_different_batch_sizes() -> None:
    for batch_size in [1, 4, 8]:
        wav = torch.randn(batch_size, 48000)
        spec = compute_stft(wav)
        assert spec.shape == (batch_size, 257, 376)
