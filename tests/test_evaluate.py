"""Tests for scripts/evaluate.py.

Covers:
  - si_snr() correctness
  - _agg() / _snr_bin() helpers
  - load_model() with a tiny synthetic checkpoint
  - evaluate_checkpoint() end-to-end with monkeypatched dataset
  - compute_pesq() / compute_stoi() when optional packages are available

Usage:
    pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.evaluate import (
    _agg,
    _snr_bin,
    compute_pesq,
    compute_stoi,
    evaluate_checkpoint,
    load_model,
    si_snr,
)
from models.av_bsrnn import AVBSRNN, AVBSRNNConfig

NUM_SAMPLES = 47648


# ── si_snr ────────────────────────────────────────────────────────────────────

def test_si_snr_perfect_estimate_above_50db() -> None:
    """When estimate == target, SI-SNR should be extremely high (>50 dB)."""
    signal = torch.randn(NUM_SAMPLES)
    result = si_snr(signal, signal)
    assert result > 50.0, f"Expected >50 dB for perfect estimate, got {result:.2f}"


def test_si_snr_zero_estimate_below_minus20db() -> None:
    """A zero estimate carries no speech; SI-SNR should be very negative."""
    clean = torch.randn(NUM_SAMPLES)
    zeros = torch.zeros(NUM_SAMPLES)
    result = si_snr(zeros, clean)
    assert result < -20.0, f"Expected <-20 dB for zero estimate, got {result:.2f}"


def test_si_snr_returns_float() -> None:
    result = si_snr(torch.randn(NUM_SAMPLES), torch.randn(NUM_SAMPLES))
    assert isinstance(result, float)


def test_si_snr_improves_as_signals_converge() -> None:
    """Adding less noise should give a higher SI-SNR."""
    clean = torch.randn(NUM_SAMPLES)
    far   = clean + 0.5  * torch.randn(NUM_SAMPLES)
    close = clean + 0.01 * torch.randn(NUM_SAMPLES)
    assert si_snr(close, clean) > si_snr(far, clean)


def test_si_snr_mean_subtracted() -> None:
    """SI-SNR is scale-invariant: doubling estimate shouldn't change the value."""
    clean = torch.randn(NUM_SAMPLES)
    est   = clean + 0.05 * torch.randn(NUM_SAMPLES)
    assert abs(si_snr(est, clean) - si_snr(2 * est, clean)) < 1.0


# ── _agg ─────────────────────────────────────────────────────────────────────

def test_agg_empty_list() -> None:
    result = _agg([])
    assert result == {"mean": None, "std": None, "median": None}


def test_agg_all_none() -> None:
    result = _agg([None, None, None])  # type: ignore[list-item]
    assert result["mean"] is None


def test_agg_values() -> None:
    result = _agg([1.0, 2.0, 3.0])
    assert abs(result["mean"] - 2.0) < 1e-6
    assert result["std"] is not None
    assert result["median"] is not None


def test_agg_skips_none_entries() -> None:
    result = _agg([1.0, None, 3.0])  # type: ignore[list-item]
    assert abs(result["mean"] - 2.0) < 1e-6


def test_agg_single_value() -> None:
    result = _agg([42.0])
    assert abs(result["mean"] - 42.0) < 1e-9
    assert result["std"] == 0.0
    assert abs(result["median"] - 42.0) < 1e-9


# ── _snr_bin ─────────────────────────────────────────────────────────────────

def test_snr_bin_low() -> None:
    assert _snr_bin(-5.0) == "low"
    assert _snr_bin(-0.001) == "low"


def test_snr_bin_mid() -> None:
    assert _snr_bin(0.0) == "mid"
    assert _snr_bin(5.0) == "mid"
    assert _snr_bin(9.999) == "mid"


def test_snr_bin_high() -> None:
    assert _snr_bin(10.0) == "high"
    assert _snr_bin(20.0) == "high"


def test_snr_bin_boundary_zero() -> None:
    """0.0 dB is the low/mid boundary — should be 'mid'."""
    assert _snr_bin(0.0) == "mid"


def test_snr_bin_boundary_ten() -> None:
    """10.0 dB is the mid/high boundary — should be 'high'."""
    assert _snr_bin(10.0) == "high"


# ── load_model ────────────────────────────────────────────────────────────────

def _tiny_cfg(use_visual: bool = False) -> dict:
    """Minimal config dict for a fast, low-memory model."""
    return {
        "model": {
            "sample_rate":             16000,
            "n_fft":                   512,
            "hop_length":              128,
            "win_length":              512,
            "num_freq":                257,
            "feat_dim":                8,
            "num_bands":               32,
            "num_low_bands":           30,
            "num_high_bands":          2,
            "num_layers":              1,
            "use_visual_conditioning": use_visual,
            "num_landmarks":           40,
            "coord_dim":               3,
            "visual_hidden_dim":       8,
            "upsample_factor":         5,
            "use_motion_deltas":       False,
            "target_audio_frames":     373,
        },
        "data": {
            "num_samples":       NUM_SAMPLES,
            "snr_range":         [-5.0, 20.0],
            "sir_range":         [-5.0, 20.0],
            "mix_probabilities": [0.5, 0.3, 0.2],
        },
    }


def _build_and_save_checkpoint(cfg: dict, ckpt_path: Path) -> AVBSRNN:
    """Instantiate tiny AVBSRNN, save checkpoint, return the model."""
    mc = cfg["model"]
    config = AVBSRNNConfig(
        feat_dim=mc["feat_dim"],
        num_layers=mc["num_layers"],
        use_visual_conditioning=mc["use_visual_conditioning"],
        visual_hidden_dim=mc["visual_hidden_dim"],
        use_motion_deltas=mc["use_motion_deltas"],
    )
    model = AVBSRNN(config)
    torch.save(
        {"model_state": model.state_dict(), "step": 1, "val_loss": 0.5},
        str(ckpt_path),
    )
    return model


def test_load_model_returns_avbsrnn(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    _build_and_save_checkpoint(cfg, ckpt)

    loaded = load_model(ckpt, cfg, torch.device("cpu"))
    assert isinstance(loaded, AVBSRNN)


def test_load_model_in_eval_mode(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    _build_and_save_checkpoint(cfg, ckpt)

    loaded = load_model(ckpt, cfg, torch.device("cpu"))
    assert not loaded.training, "load_model should return model in eval mode"


def test_load_model_weights_match(tmp_path: Path) -> None:
    """Loaded weights should exactly match the saved checkpoint."""
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    original = _build_and_save_checkpoint(cfg, ckpt)

    loaded = load_model(ckpt, cfg, torch.device("cpu"))
    for (n1, p1), (n2, p2) in zip(
        original.state_dict().items(),
        loaded.state_dict().items(),
    ):
        assert torch.allclose(p1, p2), f"Weight mismatch in {n1}"


# ── evaluate_checkpoint (end-to-end with synthetic data) ─────────────────────

class _FakeDataset(torch.utils.data.Dataset):
    """Four synthetic clips that mimic GRIDAVDataset output format."""

    SNR_VALUES = [-3.0, 5.0, 15.0, 2.0]  # low, mid, high, mid bins

    def __init__(self, n: int = 4, n_samples: int = NUM_SAMPLES) -> None:
        self.n = n
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        clean = torch.randn(self.n_samples)
        noisy = clean + 0.3 * torch.randn(self.n_samples)
        return {
            "clean":     clean,
            "noisy":     noisy,
            "landmarks": torch.zeros(75, 40, 3),
            "clip_id":   f"fake_{idx:04d}",
            "speaker":   f"s{idx + 1}",
            "mix_type":  torch.tensor(idx % 3),
            "snr_db":    torch.tensor(self.SNR_VALUES[idx % len(self.SNR_VALUES)]),
            "sir_db":    torch.tensor(0.0),
        }


def _fake_collate(batch: list[dict]) -> dict:
    return {
        "noisy":     torch.stack([b["noisy"]     for b in batch]),
        "clean":     torch.stack([b["clean"]     for b in batch]),
        "landmarks": torch.stack([b["landmarks"] for b in batch]),
        "clip_ids":  [b["clip_id"]  for b in batch],
        "speakers":  [b["speaker"]  for b in batch],
        "mix_types": torch.stack([b["mix_type"]  for b in batch]),
        "snr_db":    torch.stack([b["snr_db"]    for b in batch]),
        "sir_db":    torch.stack([b["sir_db"]    for b in batch]),
    }


def _patch_evaluate(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Monkeypatch dataset, collate, and perceptual metrics in the evaluate module."""
    import scripts.evaluate as ev

    monkeypatch.setattr(ev, "GRIDAVDataset", lambda **kwargs: _FakeDataset())
    monkeypatch.setattr(ev, "grid_av_collate", _fake_collate)
    # Return deterministic non-None values so no clips are skipped
    monkeypatch.setattr(ev, "compute_pesq", lambda ref, est, sr=16000: 3.0)
    monkeypatch.setattr(ev, "compute_stoi", lambda ref, est, sr=16000: 0.85)


def test_evaluate_checkpoint_output_files_exist(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    _build_and_save_checkpoint(cfg, ckpt)
    _patch_evaluate(monkeypatch)

    fake_dir = tmp_path / "data"
    fake_dir.mkdir()

    evaluate_checkpoint(
        checkpoint_path=ckpt,
        cfg=cfg,
        manifest_path=fake_dir / "manifest.json",
        audio_dir=fake_dir,
        landmark_dir=fake_dir,
        noise_dir=fake_dir,
        output_dir=tmp_path / "out",
        batch_size=2,
        device=torch.device("cpu"),
        num_audio_samples=4,
    )

    assert (tmp_path / "out" / "metrics.csv").exists()
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "samples").is_dir()


def test_evaluate_checkpoint_result_structure(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    _build_and_save_checkpoint(cfg, ckpt)
    _patch_evaluate(monkeypatch)

    fake_dir = tmp_path / "data"
    fake_dir.mkdir()

    results = evaluate_checkpoint(
        checkpoint_path=ckpt,
        cfg=cfg,
        manifest_path=fake_dir / "manifest.json",
        audio_dir=fake_dir,
        landmark_dir=fake_dir,
        noise_dir=fake_dir,
        output_dir=tmp_path / "out",
        batch_size=2,
        device=torch.device("cpu"),
    )

    # Top-level keys
    for key in ("num_clips", "num_skipped", "zero_landmarks", "overall",
                "by_mix_type", "by_snr_bin"):
        assert key in results, f"Missing key: {key!r}"

    assert isinstance(results["num_clips"], int)
    assert results["num_clips"] > 0, "Should have evaluated at least one clip"

    # overall sub-keys
    expected_metrics = [
        "baseline_sisnr", "enhanced_sisnr", "sisnr_improvement",
        "baseline_pesq",  "enhanced_pesq",  "pesq_improvement",
        "baseline_stoi",  "enhanced_stoi",  "stoi_improvement",
    ]
    for k in expected_metrics:
        assert k in results["overall"], f"Missing overall key: {k!r}"

    # by_mix_type has three buckets
    for mt in ("target_noise", "target_interferer_noise", "target_interferer"):
        assert mt in results["by_mix_type"], f"Missing mix-type key: {mt!r}"

    # by_snr_bin has five buckets
    assert len(results["by_snr_bin"]) == 5


def test_evaluate_checkpoint_csv_columns(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    _build_and_save_checkpoint(cfg, ckpt)
    _patch_evaluate(monkeypatch)

    fake_dir = tmp_path / "data"
    fake_dir.mkdir()

    evaluate_checkpoint(
        checkpoint_path=ckpt,
        cfg=cfg,
        manifest_path=fake_dir / "manifest.json",
        audio_dir=fake_dir,
        landmark_dir=fake_dir,
        noise_dir=fake_dir,
        output_dir=tmp_path / "out",
        batch_size=4,
        device=torch.device("cpu"),
    )

    with (tmp_path / "out" / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) > 0
    for col in ("clip_id", "speaker", "snr_db",
                "enhanced_sisnr", "sisnr_improvement"):
        assert col in rows[0], f"Missing CSV column: {col!r}"


def test_evaluate_checkpoint_json_roundtrip(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    _build_and_save_checkpoint(cfg, ckpt)
    _patch_evaluate(monkeypatch)

    fake_dir = tmp_path / "data"
    fake_dir.mkdir()

    results = evaluate_checkpoint(
        checkpoint_path=ckpt,
        cfg=cfg,
        manifest_path=fake_dir / "manifest.json",
        audio_dir=fake_dir,
        landmark_dir=fake_dir,
        noise_dir=fake_dir,
        output_dir=tmp_path / "out",
        batch_size=4,
        device=torch.device("cpu"),
    )

    with (tmp_path / "out" / "summary.json").open() as fh:
        loaded = json.load(fh)

    assert loaded["num_clips"] == results["num_clips"]
    assert loaded["zero_landmarks"] == results["zero_landmarks"]


def test_evaluate_zero_landmarks_recorded(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """zero_landmarks flag should be faithfully recorded in results."""
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    _build_and_save_checkpoint(cfg, ckpt)
    _patch_evaluate(monkeypatch)

    fake_dir = tmp_path / "data"
    fake_dir.mkdir()

    for flag in (False, True):
        results = evaluate_checkpoint(
            checkpoint_path=ckpt,
            cfg=cfg,
            manifest_path=fake_dir / "manifest.json",
            audio_dir=fake_dir,
            landmark_dir=fake_dir,
            noise_dir=fake_dir,
            output_dir=tmp_path / f"out_{flag}",
            batch_size=4,
            device=torch.device("cpu"),
            zero_landmarks=flag,
        )
        assert results["zero_landmarks"] is flag


def test_evaluate_saves_audio_samples(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """samples/ directory should contain WAV triples for saved clips."""
    cfg = _tiny_cfg()
    ckpt = tmp_path / "model.pt"
    _build_and_save_checkpoint(cfg, ckpt)
    _patch_evaluate(monkeypatch)

    fake_dir = tmp_path / "data"
    fake_dir.mkdir()

    evaluate_checkpoint(
        checkpoint_path=ckpt,
        cfg=cfg,
        manifest_path=fake_dir / "manifest.json",
        audio_dir=fake_dir,
        landmark_dir=fake_dir,
        noise_dir=fake_dir,
        output_dir=tmp_path / "out",
        batch_size=4,
        device=torch.device("cpu"),
        num_audio_samples=4,
    )

    wavs = list((tmp_path / "out" / "samples").glob("*.wav"))
    # Each saved clip → 3 WAVs (noisy, enhanced, clean)
    assert len(wavs) > 0
    assert len(wavs) % 3 == 0, f"Expected WAV triples, got {len(wavs)} files"


# ── compute_pesq / compute_stoi (optional deps) ──────────────────────────────

def _speech_like(n: int = NUM_SAMPLES, amplitude: float = 0.1) -> np.ndarray:
    """Low-amplitude random signal that PESQ/STOI can process."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) * amplitude).astype(np.float32)


def test_compute_pesq_returns_float_in_range() -> None:
    pytest.importorskip("pesq")
    ref = _speech_like()
    est = ref + _speech_like(amplitude=0.01)
    result = compute_pesq(ref, est, sr=16000)
    assert result is not None
    assert isinstance(result, float)
    # Nominal MOS-LQO range is [-0.5, 4.5], but implementations may overshoot
    # slightly on near-identical signals; use a generous sentinel.
    assert -1.0 <= result <= 5.0, f"PESQ suspiciously out of range: {result}"


def test_compute_pesq_clips_amplitude_without_crash() -> None:
    """compute_pesq clips to [-1, 1] internally — overshooting shouldn't raise."""
    pytest.importorskip("pesq")
    ref = np.ones(NUM_SAMPLES, dtype=np.float32) * 5.0
    est = np.ones(NUM_SAMPLES, dtype=np.float32) * 4.9
    result = compute_pesq(ref, est, sr=16000)
    assert result is None or isinstance(result, float)


def test_compute_stoi_returns_float_in_range() -> None:
    pytest.importorskip("pystoi")
    ref = _speech_like()
    est = ref + _speech_like(amplitude=0.01)
    result = compute_stoi(ref, est, sr=16000)
    assert result is not None
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0, f"STOI out of range: {result}"


def test_compute_stoi_degenerate_zeros_graceful() -> None:
    """Degenerate zero input should return None, not crash."""
    pytest.importorskip("pystoi")
    ref = np.zeros(NUM_SAMPLES, dtype=np.float32)
    est = np.zeros(NUM_SAMPLES, dtype=np.float32)
    result = compute_stoi(ref, est, sr=16000)
    assert result is None or isinstance(result, float)
