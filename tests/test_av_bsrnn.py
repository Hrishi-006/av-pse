"""Tests for the full AV-BSRNN model wrapper.

Usage:
    pytest tests/test_av_bsrnn.py -v
"""

from __future__ import annotations

import torch

from models.av_bsrnn import AVBSRNN, AVBSRNNConfig


def test_default_config_with_visual_conditioning() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    model = AVBSRNN(config)

    wav = torch.randn(2, 47648)
    landmarks = torch.randn(2, 75, 40, 3)

    out = model(wav, landmarks)

    assert isinstance(out, dict)
    assert "waveform" in out
    assert "spectrogram" in out
    assert "noisy_spectrogram" in out
    assert out["waveform"].shape == (2, 47648)
    assert out["spectrogram"].shape == (2, 257, 373)
    assert out["spectrogram"].is_complex()
    assert out["noisy_spectrogram"].shape == (2, 257, 373)
    assert out["noisy_spectrogram"].is_complex()


def test_without_visual_conditioning() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=False, num_layers=2)
    model = AVBSRNN(config)

    wav = torch.randn(2, 47648)
    out = model(wav)

    assert out["waveform"].shape == (2, 47648)
    assert out["spectrogram"].shape == (2, 257, 373)


def test_visual_conditioning_requires_landmarks() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    model = AVBSRNN(config)

    wav = torch.randn(1, 47648)
    try:
        model(wav, landmarks=None)
        assert False, "Expected assertion error for missing landmarks"
    except AssertionError:
        pass


def test_gradient_flow_through_full_model_with_visual() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    model = AVBSRNN(config)

    wav = torch.randn(1, 47648, requires_grad=True)
    landmarks = torch.randn(1, 75, 40, 3, requires_grad=True)

    out = model(wav, landmarks)
    loss = out["waveform"].abs().sum() + out["spectrogram"].abs().sum()
    loss.backward()

    assert wav.grad is not None
    assert landmarks.grad is not None
    assert torch.isfinite(wav.grad).all()
    assert torch.isfinite(landmarks.grad).all()

    zero_grad_params = []
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad: {name}"
        if param.grad.abs().sum() == 0:
            zero_grad_params.append(name)
    assert len(zero_grad_params) == 0, f"Zero gradients for: {zero_grad_params}"


def test_gradient_flow_without_visual() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=False, num_layers=2)
    model = AVBSRNN(config)

    wav = torch.randn(1, 47648, requires_grad=True)
    out = model(wav)
    loss = out["waveform"].abs().sum()
    loss.backward()

    assert wav.grad is not None
    for _name, param in model.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_parameter_count_breakdown() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=6)
    model = AVBSRNN(config)
    counts = model.count_parameters()

    print(f"Parameter counts: {counts}")

    assert counts["band_split"] > 0
    assert counts["visual_conditioning"] > 0
    assert counts["band_sequence_rnn"] > 0
    assert counts["mask_decoder"] > 0
    assert counts["total"] == sum(value for key, value in counts.items() if key != "total")
    assert 1_000_000 < counts["total"] < 50_000_000, f"Total: {counts['total']:,}"


def test_without_visual_has_fewer_params() -> None:
    config_with = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    config_without = AVBSRNNConfig(use_visual_conditioning=False, num_layers=2)

    model_with = AVBSRNN(config_with)
    model_without = AVBSRNN(config_without)

    assert model_without.count_parameters()["total"] < model_with.count_parameters()["total"]


def test_visual_conditioning_component_is_none_when_disabled() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=False, num_layers=2)
    model = AVBSRNN(config)
    assert model.visual_conditioning is None


def test_module_b_feat_dim_matches_visual_conditioning_state() -> None:
    config_with = AVBSRNNConfig(use_visual_conditioning=True, feat_dim=128, num_layers=2)
    model_with = AVBSRNN(config_with)
    assert model_with.band_sequence_rnn.feat_dim == 256
    assert model_with.mask_decoder.mask_mlps[0].linear_in.in_features == 256

    config_without = AVBSRNNConfig(use_visual_conditioning=False, feat_dim=128, num_layers=2)
    model_without = AVBSRNN(config_without)
    assert model_without.band_sequence_rnn.feat_dim == 128
    assert model_without.mask_decoder.mask_mlps[0].linear_in.in_features == 128


def test_round_trip_waveform_length_preserved() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    model = AVBSRNN(config)

    wav = torch.randn(1, 47648)
    landmarks = torch.randn(1, 75, 40, 3)
    out = model(wav, landmarks)
    assert out["waveform"].shape == (1, 47648)


def test_batch_size_invariance() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    model = AVBSRNN(config)

    for batch_size in [1, 2, 4]:
        wav = torch.randn(batch_size, 47648)
        landmarks = torch.randn(batch_size, 75, 40, 3)
        out = model(wav, landmarks)
        assert out["waveform"].shape == (batch_size, 47648)
        assert out["spectrogram"].shape == (batch_size, 257, 373)


def test_device_handling() -> None:
    if torch.cuda.is_available():
        config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
        model = AVBSRNN(config).cuda()
        wav = torch.randn(1, 47648).cuda()
        landmarks = torch.randn(1, 75, 40, 3).cuda()
        out = model(wav, landmarks)
        assert out["waveform"].device.type == "cuda"
        assert out["spectrogram"].device.type == "cuda"


def test_eval_mode_produces_deterministic_output() -> None:
    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    model = AVBSRNN(config)
    model.eval()

    wav = torch.randn(1, 47648)
    landmarks = torch.randn(1, 75, 40, 3)

    with torch.no_grad():
        out1 = model(wav, landmarks)
        out2 = model(wav, landmarks)

    assert torch.allclose(out1["waveform"], out2["waveform"])


def test_frozen_config_dataclass_is_immutable() -> None:
    config = AVBSRNNConfig()
    try:
        config.feat_dim = 256
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass

