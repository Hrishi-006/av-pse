"""Tests for the BandSplit module.

Usage:
    pytest tests/test_band_split.py -v
"""

from __future__ import annotations

import torch

from models.band_split import BandSplit
from models.stft import compute_stft


def test_output_shape() -> None:
    module = BandSplit(feat_dim=128)
    spec = torch.randn(2, 257, 376, dtype=torch.complex64)
    out = module(spec)
    assert out.shape == (2, 32, 376, 128), f"Got {out.shape}"
    assert out.dtype == torch.float32


def test_band_widths_sum_correctly() -> None:
    module = BandSplit()
    assert sum(module.BAND_WIDTHS) == 257
    assert len(module.BAND_WIDTHS) == 32
    assert module.num_bands == 32


def test_each_band_has_its_own_parameters() -> None:
    module = BandSplit()
    assert len(module.norms) == 32
    assert len(module.projections) == 32

    for i, (norm, proj) in enumerate(zip(module.norms, module.projections)):
        expected_in = 2 * module.BAND_WIDTHS[i]
        assert norm.normalized_shape == (expected_in,)
        assert proj.in_features == expected_in
        assert proj.out_features == 128


def test_integration_with_stft() -> None:
    wav = torch.randn(2, 48000)
    spec = compute_stft(wav)
    module = BandSplit()
    out = module(spec)
    assert out.shape == (2, 32, 376, 128)


def test_gradient_flow() -> None:
    module = BandSplit()
    spec = torch.randn(1, 257, 376, dtype=torch.complex64)
    out = module(spec)
    loss = out.sum()
    loss.backward()

    for name, param in module.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_batch_size_invariance() -> None:
    module = BandSplit()
    for batch_size in [1, 4, 8]:
        spec = torch.randn(batch_size, 257, 376, dtype=torch.complex64)
        out = module(spec)
        assert out.shape == (batch_size, 32, 376, 128)


def test_different_feat_dim() -> None:
    for feat_dim in [64, 96, 128, 192]:
        module = BandSplit(feat_dim=feat_dim)
        spec = torch.randn(1, 257, 376, dtype=torch.complex64)
        out = module(spec)
        assert out.shape == (1, 32, 376, feat_dim)


def test_device_handling() -> None:
    if torch.cuda.is_available():
        module = BandSplit().cuda()
        spec = torch.randn(1, 257, 376, dtype=torch.complex64).cuda()
        out = module(spec)
        assert out.device.type == "cuda"

