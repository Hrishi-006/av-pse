"""Tests for waveform mixing helpers.

Usage:
    pytest tests/test_mixing.py -v
"""

from __future__ import annotations

import torch

from data.mixing import mix_audio, rms, sample_mixing_config, scale_to_snr


def test_rms_computation() -> None:
    x = torch.ones(2, 100)
    assert torch.allclose(rms(x), torch.tensor([1.0, 1.0]))
    x = torch.zeros(2, 100)
    assert torch.isfinite(rms(x)).all()


def test_snr_scaling_achieves_target_snr() -> None:
    target = torch.randn(4, 48000)
    noise = torch.randn(4, 48000)
    snr_db = torch.tensor([0.0, 5.0, 10.0, 20.0])
    scaled_noise = scale_to_snr(target, noise, snr_db)
    achieved_snr = 20 * torch.log10(rms(target) / rms(scaled_noise))
    assert torch.allclose(achieved_snr, snr_db, atol=0.1), (
        f"Achieved {achieved_snr} vs target {snr_db}"
    )


def test_mixing_produces_expected_shape() -> None:
    target = torch.randn(2, 48000) * 0.1
    noise = torch.randn(2, 48000) * 0.1
    interferer = torch.randn(2, 48000) * 0.1
    snr_db = torch.tensor([5.0, 10.0])
    sir_db = torch.tensor([5.0, 10.0])
    use_int = torch.tensor([True, False])
    mixed = mix_audio(target, noise, interferer, snr_db, sir_db, use_int)
    assert mixed.shape == (2, 48000)
    assert mixed.abs().max() <= 0.99


def test_interferer_is_excluded_when_mask_is_false() -> None:
    target = torch.ones(1, 1000) * 0.1
    noise = torch.zeros(1, 1000)
    interferer = torch.ones(1, 1000) * 100
    snr_db = torch.tensor([0.0])
    sir_db = torch.tensor([0.0])
    use_int = torch.tensor([False])
    mixed = mix_audio(target, noise, interferer, snr_db, sir_db, use_int)
    assert torch.allclose(mixed, target, atol=1e-5)


def test_sampling_config_respects_probabilities() -> None:
    torch.manual_seed(42)
    cfg = sample_mixing_config(batch_size=10000)
    frac_with_interferer = cfg["use_interferer"].float().mean()
    assert 0.45 < frac_with_interferer < 0.55
    frac_with_noise = cfg["use_noise"].float().mean()
    assert 0.75 < frac_with_noise < 0.85


def test_snr_values_are_in_range() -> None:
    cfg = sample_mixing_config(batch_size=1000, snr_range=(-5, 20))
    assert cfg["snr_db"].min() >= -5
    assert cfg["snr_db"].max() <= 20
