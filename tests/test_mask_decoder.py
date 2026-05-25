"""Tests for the complex mask and residual decoder.

Usage:
    pytest tests/test_mask_decoder.py -v
"""

from __future__ import annotations

import torch

from models.band_sequence_rnn import BandSequenceRNN
from models.band_split import BandSplit
from models.mask_decoder import BandMaskMLP, MaskDecoder, ResidualMLP
from models.stft import compute_stft


def test_band_mask_mlp_output_shape() -> None:
    mlp = BandMaskMLP(feat_dim=128, band_width=4)
    x = torch.randn(2, 373, 128)
    out = mlp(x)
    assert out.shape == (2, 373, 4, 2), f"Got {out.shape}"


def test_band_mask_mlp_with_different_band_widths() -> None:
    for width in [4, 12, 28, 29]:
        mlp = BandMaskMLP(feat_dim=128, band_width=width)
        x = torch.randn(1, 100, 128)
        out = mlp(x)
        assert out.shape == (1, 100, width, 2)


def test_residual_mlp_output_shape_and_complex() -> None:
    mlp = ResidualMLP(num_bands=32, feat_dim=128, num_freq=257)
    x = torch.randn(2, 32, 373, 128)
    out = mlp(x)
    assert out.shape == (2, 257, 373)
    assert out.is_complex()


def test_mask_decoder_output_shape() -> None:
    decoder = MaskDecoder(feat_dim=128)
    features = torch.randn(2, 32, 373, 128)
    noisy_spec = torch.randn(2, 257, 373, dtype=torch.complex64)
    enhanced = decoder(features, noisy_spec)
    assert enhanced.shape == (2, 257, 373)
    assert enhanced.is_complex()


def test_band_widths_sum_correctly() -> None:
    decoder = MaskDecoder()
    assert sum(decoder.BAND_WIDTHS) == 257
    assert len(decoder.BAND_WIDTHS) == 32
    assert decoder.num_bands == 32


def test_each_band_has_its_own_mask_mlp() -> None:
    decoder = MaskDecoder()
    assert len(decoder.mask_mlps) == 32
    for i in range(32):
        assert decoder.mask_mlps[i].band_width == decoder.BAND_WIDTHS[i]


def test_mismatched_k_raises() -> None:
    decoder = MaskDecoder(feat_dim=128)
    features_wrong = torch.randn(1, 31, 373, 128)
    noisy_spec = torch.randn(1, 257, 373, dtype=torch.complex64)
    try:
        decoder(features_wrong, noisy_spec)
        assert False, "Expected assertion error"
    except AssertionError:
        pass


def test_gradient_flow() -> None:
    decoder = MaskDecoder(feat_dim=128)
    features = torch.randn(1, 32, 373, 128, requires_grad=True)
    noisy_spec = torch.randn(1, 257, 373, dtype=torch.complex64)
    enhanced = decoder(features, noisy_spec)

    loss = enhanced.abs().sum()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_end_to_end_pipeline_shape() -> None:
    wav = torch.randn(2, 47648)
    spec = compute_stft(wav)

    split = BandSplit(feat_dim=128)
    features = split(spec)

    rnn = BandSequenceRNN(feat_dim=128, num_low_bands=30, num_high_bands=2, num_layers=2)
    processed = rnn(features)

    decoder = MaskDecoder(feat_dim=128)
    enhanced = decoder(processed, spec)

    assert enhanced.shape == spec.shape
    assert enhanced.is_complex()


def test_different_feat_dim_works_for_visual_conditioning_case() -> None:
    decoder = MaskDecoder(feat_dim=256)
    features = torch.randn(1, 32, 373, 256)
    noisy_spec = torch.randn(1, 257, 373, dtype=torch.complex64)
    enhanced = decoder(features, noisy_spec)
    assert enhanced.shape == (1, 257, 373)
    assert enhanced.is_complex()


def test_batch_size_invariance() -> None:
    decoder = MaskDecoder(feat_dim=128)
    for batch_size in [1, 2, 4]:
        features = torch.randn(batch_size, 32, 373, 128)
        noisy_spec = torch.randn(batch_size, 257, 373, dtype=torch.complex64)
        enhanced = decoder(features, noisy_spec)
        assert enhanced.shape == (batch_size, 257, 373)


def test_device_handling() -> None:
    if torch.cuda.is_available():
        decoder = MaskDecoder(feat_dim=128).cuda()
        features = torch.randn(1, 32, 373, 128).cuda()
        noisy_spec = torch.randn(1, 257, 373, dtype=torch.complex64).cuda()
        enhanced = decoder(features, noisy_spec)
        assert enhanced.device.type == "cuda"


def test_parameter_count_is_reasonable() -> None:
    decoder = MaskDecoder(feat_dim=128)
    total = sum(param.numel() for param in decoder.parameters())
    assert 1_000_000 < total < 50_000_000
    print(f"MaskDecoder parameters: {total:,}")

