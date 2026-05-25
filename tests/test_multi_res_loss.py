"""Tests for the multi-resolution spectral loss.

Usage:
    pytest tests/test_multi_res_loss.py -v
"""

from __future__ import annotations

import torch
import pytest

from losses.multi_res_loss import MultiResolutionLoss


def _make_loss() -> MultiResolutionLoss:
    return MultiResolutionLoss(sample_rate=16000)


def test_output_is_scalar() -> None:
    loss_fn = _make_loss()
    enhanced = torch.randn(2, 48000)
    clean = torch.randn(2, 48000)
    loss = loss_fn(enhanced, clean)
    assert loss.shape == torch.Size([])


def test_loss_has_gradient() -> None:
    loss_fn = _make_loss()
    enhanced = torch.randn(2, 48000, requires_grad=True)
    clean = torch.randn(2, 48000)
    loss = loss_fn(enhanced, clean)
    loss.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()


def test_loss_is_positive_for_different_signals() -> None:
    loss_fn = _make_loss()
    enhanced = torch.randn(2, 48000)
    clean = torch.randn(2, 48000)
    loss = loss_fn(enhanced, clean)
    assert loss.item() > 0.0


def test_loss_is_near_zero_for_identical_signals() -> None:
    loss_fn = _make_loss()
    signal = torch.randn(2, 48000)
    loss = loss_fn(signal, signal)
    # Magnitude term: |x^p - x^p| = 0; complex term: |0| = 0. Only eps offset.
    assert loss.item() < 1e-4, f"Expected near-zero loss, got {loss.item()}"


def test_loss_decreases_as_signals_converge() -> None:
    loss_fn = _make_loss()
    clean = torch.randn(1, 48000)
    noise = torch.randn(1, 48000)

    loss_far = loss_fn(clean + noise, clean)
    loss_close = loss_fn(clean + 0.01 * noise, clean)
    assert loss_close.item() < loss_far.item()


def test_all_window_sizes_work() -> None:
    for ms in [10, 20, 30, 40]:
        loss_fn = MultiResolutionLoss(sample_rate=16000, window_ms=[ms])
        enhanced = torch.randn(1, 48000)
        clean = torch.randn(1, 48000)
        loss = loss_fn(enhanced, clean)
        assert loss.shape == torch.Size([])
        assert loss.item() > 0.0


def test_default_uses_four_resolutions() -> None:
    loss_fn = _make_loss()
    assert len(loss_fn.n_ffts) == 4


def test_n_ffts_convert_correctly_at_16khz() -> None:
    loss_fn = _make_loss()
    # 10ms→160, 20ms→320, 30ms→480, 40ms→640
    assert loss_fn.n_ffts == [160, 320, 480, 640]


def test_different_batch_sizes() -> None:
    loss_fn = _make_loss()
    for batch in [1, 2, 4]:
        enhanced = torch.randn(batch, 48000)
        clean = torch.randn(batch, 48000)
        loss = loss_fn(enhanced, clean)
        assert loss.shape == torch.Size([])


def test_shape_mismatch_raises() -> None:
    loss_fn = _make_loss()
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(torch.randn(2, 48000), torch.randn(2, 44100))


def test_non_2d_input_raises() -> None:
    loss_fn = _make_loss()
    with pytest.raises(ValueError, match="Expected \\[B, S\\]"):
        loss_fn(torch.randn(48000), torch.randn(48000))


def test_gradient_flows_through_loss() -> None:
    loss_fn = _make_loss()
    enhanced = torch.randn(1, 48000, requires_grad=True)
    clean = torch.randn(1, 48000)
    loss = loss_fn(enhanced, clean)
    loss.backward()
    assert enhanced.grad is not None
    assert enhanced.grad.abs().sum() > 0


def test_device_agnostic_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    loss_fn = _make_loss()
    enhanced = torch.randn(1, 48000).cuda()
    clean = torch.randn(1, 48000).cuda()
    loss = loss_fn(enhanced, clean)
    assert loss.device.type == "cuda"
    assert loss.shape == torch.Size([])


def test_custom_power_compression() -> None:
    loss_fn_p03 = MultiResolutionLoss(sample_rate=16000, power=0.3)
    loss_fn_p10 = MultiResolutionLoss(sample_rate=16000, power=1.0)
    enhanced = torch.randn(1, 48000)
    clean = torch.randn(1, 48000)
    loss_p03 = loss_fn_p03(enhanced, clean)
    loss_p10 = loss_fn_p10(enhanced, clean)
    # Different powers produce different losses
    assert abs(loss_p03.item() - loss_p10.item()) > 1e-6


def test_invalid_power_raises() -> None:
    with pytest.raises(ValueError, match="power"):
        MultiResolutionLoss(sample_rate=16000, power=0.0)


def test_empty_window_list_raises() -> None:
    with pytest.raises(ValueError, match="window_ms"):
        MultiResolutionLoss(sample_rate=16000, window_ms=[])
