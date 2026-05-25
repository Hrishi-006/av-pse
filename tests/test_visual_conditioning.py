"""Tests for the visual conditioning module.

Usage:
    pytest tests/test_visual_conditioning.py -v
"""

from __future__ import annotations

import torch

from models.visual_conditioning import VisualConditioningModule


def test_output_shape_with_default_config() -> None:
    module = VisualConditioningModule()
    x = torch.randn(2, 75, 40, 3)
    out = module(x)
    assert out.shape == (2, 32, 373, 128), f"Got {out.shape}"


def test_output_is_real_valued_float32() -> None:
    module = VisualConditioningModule()
    x = torch.randn(1, 75, 40, 3)
    out = module(x)
    assert out.dtype == torch.float32
    assert not out.is_complex()


def test_motion_deltas_toggle_changes_input_width() -> None:
    m1 = VisualConditioningModule(use_motion_deltas=True)
    assert m1.input_projection.in_features == 40 * 3 * 2

    m2 = VisualConditioningModule(use_motion_deltas=False)
    assert m2.input_projection.in_features == 40 * 3


def test_output_shape_unchanged_whether_deltas_are_on_or_off() -> None:
    for use_deltas in [True, False]:
        module = VisualConditioningModule(use_motion_deltas=use_deltas)
        x = torch.randn(1, 75, 40, 3)
        out = module(x)
        assert out.shape == (1, 32, 373, 128)


def test_gradient_flow() -> None:
    module = VisualConditioningModule()
    x = torch.randn(1, 75, 40, 3, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_different_batch_sizes() -> None:
    module = VisualConditioningModule()
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 75, 40, 3)
        out = module(x)
        assert out.shape == (batch_size, 32, 373, 128)


def test_each_band_has_its_own_projection() -> None:
    module = VisualConditioningModule()
    assert len(module.band_projections) == 32
    ids = {id(projection) for projection in module.band_projections}
    assert len(ids) == 32


def test_configurable_feat_dim() -> None:
    for feat_dim in [64, 96, 128, 192]:
        module = VisualConditioningModule(feat_dim=feat_dim)
        x = torch.randn(1, 75, 40, 3)
        out = module(x)
        assert out.shape == (1, 32, 373, feat_dim)


def test_configurable_num_bands() -> None:
    module = VisualConditioningModule(num_bands=20)
    x = torch.randn(1, 75, 40, 3)
    out = module(x)
    assert out.shape == (1, 20, 373, 128)


def test_deltas_computation_is_correct() -> None:
    module = VisualConditioningModule(use_motion_deltas=True)
    x = torch.randn(1, 75, 40, 3)
    deltas = module._compute_deltas(x)
    assert deltas.shape == x.shape
    assert torch.allclose(deltas[:, 0], torch.zeros_like(deltas[:, 0]))
    expected_delta_1 = x[:, 1] - x[:, 0]
    assert torch.allclose(deltas[:, 1], expected_delta_1)


def test_concatenation_with_band_split_features_integration() -> None:
    from models.band_split import BandSplit
    from models.stft import compute_stft

    wav = torch.randn(2, 47648)
    spec = compute_stft(wav)

    bs = BandSplit(feat_dim=128)
    z = bs(spec)

    vc = VisualConditioningModule()
    landmarks = torch.randn(2, 75, 40, 3)
    v = vc(landmarks)

    combined = torch.cat([z, v], dim=-1)
    assert combined.shape == (2, 32, 373, 256)


def test_full_pipeline_including_visual_conditioning() -> None:
    from models.band_sequence_rnn import BandSequenceRNN
    from models.band_split import BandSplit
    from models.mask_decoder import MaskDecoder
    from models.stft import compute_stft

    wav = torch.randn(1, 47648)
    landmarks = torch.randn(1, 75, 40, 3)

    spec = compute_stft(wav)
    z = BandSplit(feat_dim=128)(spec)
    v = VisualConditioningModule()(landmarks)
    combined = torch.cat([z, v], dim=-1)

    rnn = BandSequenceRNN(feat_dim=256, num_low_bands=30, num_high_bands=2, num_layers=2)
    processed = rnn(combined)

    enhanced = MaskDecoder(feat_dim=256)(processed, spec)
    assert enhanced.shape == (1, 257, 373)
    assert enhanced.is_complex()


def test_device_handling() -> None:
    if torch.cuda.is_available():
        module = VisualConditioningModule().cuda()
        x = torch.randn(1, 75, 40, 3).cuda()
        out = module(x)
        assert out.device.type == "cuda"


def test_parameter_count_is_reasonable() -> None:
    module = VisualConditioningModule()
    total = sum(param.numel() for param in module.parameters())
    assert 50_000 < total < 5_000_000
    print(f"VisualConditioningModule parameters: {total:,}")


def test_constant_landmarks_produce_non_zero_output() -> None:
    module = VisualConditioningModule(use_motion_deltas=True)
    module.eval()
    x = torch.ones(1, 75, 40, 3) * 0.1
    with torch.no_grad():
        out = module(x)
    assert out.abs().sum() > 0

