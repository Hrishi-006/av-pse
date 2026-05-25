"""Tests for BSRNN-S band and sequence modeling.

Usage:
    pytest tests/test_band_sequence_rnn.py -v
"""

from __future__ import annotations

import torch

from models.band_sequence_rnn import BandBlockBSRNNS, BandSequenceLayer, BandSequenceRNN, TimeBlock


def test_time_block_preserves_shape() -> None:
    block = TimeBlock(feat_dim=128)
    x = torch.randn(2, 32, 373, 128)
    out = block(x)
    assert out.shape == x.shape


def test_time_block_is_non_trivial() -> None:
    block = TimeBlock(feat_dim=128)
    x = torch.randn(2, 4, 50, 128)
    out = block(x)
    assert (out - x).abs().mean() > 1e-4


def test_band_block_bsrnns_preserves_shape() -> None:
    block = BandBlockBSRNNS(feat_dim=128, num_low_bands=30, num_high_bands=2)
    x = torch.randn(2, 32, 373, 128)
    out = block(x)
    assert out.shape == x.shape


def test_band_block_bsrnns_validates_k_mismatch() -> None:
    block = BandBlockBSRNNS(feat_dim=128, num_low_bands=30, num_high_bands=2)
    x = torch.randn(1, 31, 373, 128)
    try:
        block(x)
        assert False, "Expected assertion error for wrong K"
    except AssertionError:
        pass


def test_band_sequence_layer_preserves_shape() -> None:
    layer = BandSequenceLayer(feat_dim=128, num_low_bands=30, num_high_bands=2)
    x = torch.randn(2, 32, 373, 128)
    out = layer(x)
    assert out.shape == x.shape


def test_full_band_sequence_rnn_preserves_shape() -> None:
    model = BandSequenceRNN(feat_dim=128, num_low_bands=30, num_high_bands=2, num_layers=6)
    x = torch.randn(2, 32, 373, 128)
    out = model(x)
    assert out.shape == x.shape


def test_configurable_feat_dim_for_visual_conditioning() -> None:
    model = BandSequenceRNN(feat_dim=256, num_low_bands=30, num_high_bands=2, num_layers=2)
    x = torch.randn(1, 32, 373, 256)
    out = model(x)
    assert out.shape == (1, 32, 373, 256)


def test_different_layer_counts() -> None:
    for num_layers in [1, 3, 6]:
        model = BandSequenceRNN(feat_dim=128, num_low_bands=30, num_high_bands=2, num_layers=num_layers)
        x = torch.randn(1, 32, 373, 128)
        out = model(x)
        assert out.shape == x.shape


def test_gradient_flow_through_full_model() -> None:
    model = BandSequenceRNN(feat_dim=128, num_low_bands=30, num_high_bands=2, num_layers=6)
    x = torch.randn(1, 32, 373, 128, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"
        assert param.grad.abs().sum() > 0, f"Zero grad for {name}"


def test_information_flow_direction_bsrnns_asymmetry() -> None:
    torch.manual_seed(0)
    block = BandBlockBSRNNS(feat_dim=64, num_low_bands=30, num_high_bands=2)
    block.eval()

    with torch.no_grad():
        low_part = torch.randn(1, 30, 10, 64)
        high_part_a = torch.zeros(1, 2, 10, 64)
        high_part_b = torch.randn(1, 2, 10, 64) * 10

        x1 = torch.cat([low_part, high_part_a], dim=1)
        x2 = torch.cat([low_part, high_part_b], dim=1)

        out1 = block(x1)
        out2 = block(x2)

        low_diff = (out1[:, :30, :, :] - out2[:, :30, :, :]).abs().max()
        assert low_diff < 1e-5, (
            "BSRNN-S asymmetry broken: low-freq output changed when "
            f"high-freq input changed (max diff: {low_diff})"
        )

        high_diff = (out1[:, 30:, :, :] - out2[:, 30:, :, :]).abs().max()
        assert high_diff > 1e-3, f"High-freq output suspiciously unchanged: {high_diff}"


def test_batch_size_invariance() -> None:
    model = BandSequenceRNN(feat_dim=128, num_low_bands=30, num_high_bands=2, num_layers=2)
    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 32, 373, 128)
        out = model(x)
        assert out.shape == (batch_size, 32, 373, 128)


def test_device_handling() -> None:
    if torch.cuda.is_available():
        model = BandSequenceRNN(feat_dim=128, num_low_bands=30, num_high_bands=2, num_layers=2).cuda()
        x = torch.randn(1, 32, 373, 128).cuda()
        out = model(x)
        assert out.device.type == "cuda"


def test_parameter_count_is_reasonable() -> None:
    model = BandSequenceRNN(feat_dim=128, num_low_bands=30, num_high_bands=2, num_layers=6)
    total = sum(param.numel() for param in model.parameters())
    assert 1_000_000 < total < 20_000_000
    print(f"Total parameters: {total:,}")

