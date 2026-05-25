"""Tests for the GRID AV dataset.

Usage:
    pytest tests/test_grid_dataset.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import soundfile as sf
import torch

from data.grid_dataset import GRIDAVDataset, grid_av_collate


@pytest.fixture
def fake_dataset_dir() -> Iterator[dict[str, str]]:
    """Create a temporary directory with a fake manifest and synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        audio_dir = tmpdir / "audio"
        landmark_dir = tmpdir / "landmarks"
        noise_dir = tmpdir / "noise"
        audio_dir.mkdir()
        landmark_dir.mkdir()
        noise_dir.mkdir()

        speakers = ["s1", "s2", "s3"]
        entries: list[dict[str, str]] = []
        for speaker in speakers:
            (audio_dir / speaker).mkdir()
            (landmark_dir / speaker).mkdir()
            for clip_idx in range(2):
                clip_id = f"clip{clip_idx}"
                wav = (np.random.randn(47648) * 0.1).astype(np.float32)
                sf.write(audio_dir / speaker / f"{clip_id}.wav", wav, 16000)

                landmarks = (np.random.randn(75, 40, 3) * 0.05).astype(np.float32)
                np.save(landmark_dir / speaker / f"{clip_id}.npy", landmarks)

                entries.append(
                    {
                        "speaker": speaker,
                        "clip_id": clip_id,
                        "audio_path": f"{speaker}/{clip_id}.wav",
                        "landmark_path": f"{speaker}/{clip_id}.npy",
                    }
                )

        for i in range(2):
            noise = (np.random.randn(47648 * 5) * 0.1).astype(np.float32)
            sf.write(noise_dir / f"noise{i}.wav", noise, 16000)

        manifest = {
            "metadata": {
                "sample_rate": 16000,
                "audio_samples_per_clip": 47648,
                "video_fps": 25,
                "frames_per_clip": 75,
                "num_lip_landmarks": 40,
                "lip_normalization": "per_frame_mean_centered",
            },
            "splits": {
                "train": entries[:4],
                "val": entries[4:5],
                "test": entries[5:],
            },
            "speakers": {
                "train": speakers[:2],
                "val": [speakers[2]],
                "test": [speakers[2]],
            },
        }
        manifest_path = tmpdir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        yield {
            "manifest_path": str(manifest_path),
            "audio_dir": str(audio_dir),
            "landmark_dir": str(landmark_dir),
            "noise_dir": str(noise_dir),
        }


def test_dataset_length(fake_dataset_dir: dict[str, str]) -> None:
    ds = GRIDAVDataset(split="train", **fake_dataset_dir)
    assert len(ds) == 4


def test_sample_shapes(fake_dataset_dir: dict[str, str]) -> None:
    ds = GRIDAVDataset(split="train", **fake_dataset_dir)
    sample = ds[0]
    assert sample["noisy"].shape == (47648,)
    assert sample["clean"].shape == (47648,)
    assert sample["landmarks"].shape == (75, 40, 3)
    assert sample["noisy"].dtype == torch.float32
    assert sample["clean"].dtype == torch.float32
    assert sample["landmarks"].dtype == torch.float32


def test_no_nans(fake_dataset_dir: dict[str, str]) -> None:
    ds = GRIDAVDataset(split="train", **fake_dataset_dir)
    for i in range(len(ds)):
        sample = ds[i]
        assert torch.isfinite(sample["noisy"]).all()
        assert torch.isfinite(sample["clean"]).all()
        assert torch.isfinite(sample["landmarks"]).all()


def test_val_reproducibility(fake_dataset_dir: dict[str, str]) -> None:
    ds1 = GRIDAVDataset(split="val", **fake_dataset_dir)
    ds2 = GRIDAVDataset(split="val", **fake_dataset_dir)
    s1 = ds1[0]
    s2 = ds2[0]
    assert torch.allclose(s1["noisy"], s2["noisy"])


def test_mix_probabilities(fake_dataset_dir: dict[str, str]) -> None:
    ds = GRIDAVDataset(
        split="train",
        mix_probabilities=(0.5, 0.3, 0.2),
        seed=42,
        **fake_dataset_dir,
    )
    counts = {0: 0, 1: 0, 2: 0}
    for trial in range(500):
        ds.seed = trial
        for i in range(len(ds)):
            mix_type = ds[i]["mix_type"]
            counts[mix_type] += 1

    total = sum(counts.values())
    assert 0.4 < counts[0] / total < 0.6
    assert 0.2 < counts[1] / total < 0.4
    assert 0.1 < counts[2] / total < 0.3


def test_collate(fake_dataset_dir: dict[str, str]) -> None:
    ds = GRIDAVDataset(split="train", **fake_dataset_dir)
    batch = grid_av_collate([ds[0], ds[1], ds[2]])
    assert batch["noisy"].shape == (3, 47648)
    assert batch["clean"].shape == (3, 47648)
    assert batch["landmarks"].shape == (3, 75, 40, 3)
    assert len(batch["speakers"]) == 3
    assert batch["snr_db"].shape == (3,)


def test_dataloader(fake_dataset_dir: dict[str, str]) -> None:
    from torch.utils.data import DataLoader

    ds = GRIDAVDataset(split="train", **fake_dataset_dir)
    loader = DataLoader(
        ds,
        batch_size=2,
        collate_fn=grid_av_collate,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert batch["noisy"].shape == (2, 47648)
    assert batch["clean"].shape == (2, 47648)
    assert batch["landmarks"].shape == (2, 75, 40, 3)


def test_end_to_end_with_model(fake_dataset_dir: dict[str, str]) -> None:
    from torch.utils.data import DataLoader

    from models.av_bsrnn import AVBSRNN, AVBSRNNConfig

    ds = GRIDAVDataset(split="train", **fake_dataset_dir)
    loader = DataLoader(ds, batch_size=2, collate_fn=grid_av_collate, shuffle=False)
    batch = next(iter(loader))

    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    model = AVBSRNN(config)

    out = model(batch["noisy"], batch["landmarks"])
    assert out["waveform"].shape == batch["clean"].shape
    assert out["spectrogram"].shape == (2, 257, 373)


def test_interferer_is_different_speaker(fake_dataset_dir: dict[str, str]) -> None:
    """When interferer is used, it should be from a different speaker."""
    ds = GRIDAVDataset(
        split="train",
        mix_probabilities=(0.0, 0.0, 1.0),
        seed=42,
        **fake_dataset_dir,
    )
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["mix_type"] == 2


def test_speaker_disjoint(fake_dataset_dir: dict[str, str]) -> None:
    ds_train = GRIDAVDataset(split="train", **fake_dataset_dir)
    ds_val = GRIDAVDataset(split="val", **fake_dataset_dir)
    train_speakers = set(ds_train.entries_by_speaker.keys())
    val_speakers = set(ds_val.entries_by_speaker.keys())
    assert len(train_speakers & val_speakers) == 0
