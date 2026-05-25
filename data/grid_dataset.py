"""GRID audio-visual dataset with on-the-fly speech/noise mixing.

Usage example:
    >>> from torch.utils.data import DataLoader
    >>> from data.grid_dataset import GRIDAVDataset, grid_av_collate
    >>> dataset = GRIDAVDataset(
    ...     manifest_path="/kaggle/input/grid-av/manifest.json",
    ...     audio_dir="/kaggle/input/grid-audio",
    ...     landmark_dir="/kaggle/input/grid-landmarks",
    ...     noise_dir="/kaggle/input/demand-noise",
    ...     split="train",
    ... )
    >>> loader = DataLoader(dataset, batch_size=8, collate_fn=grid_av_collate)
    >>> batch = next(iter(loader))
    >>> batch["noisy"].shape, batch["clean"].shape, batch["landmarks"].shape
    (torch.Size([8, 48000]), torch.Size([8, 48000]), torch.Size([8, 75, 40, 3]))
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from data.mixing import mix_audio


class GRIDAVDataset(Dataset[dict[str, Any]]):
    """Audio-visual personalized speech enhancement dataset over GRID.

    Each item returns a target clean waveform, matching lip landmarks, and a
    noisy mixture generated on the fly.

    Output shapes:
        ``noisy``: ``[num_samples]`` float32 waveform.
        ``clean``: ``[num_samples]`` float32 waveform.
        ``landmarks``: ``[num_frames, num_landmarks, 3]`` float32 landmarks.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        audio_dir: str | Path,
        landmark_dir: str | Path,
        noise_dir: str | Path,
        split: str = "train",
        snr_range: tuple[float, float] = (-5.0, 20.0),
        sir_range: tuple[float, float] = (-5.0, 20.0),
        mix_probabilities: tuple[float, float, float] = (0.5, 0.3, 0.2),
        num_samples: int = 48000,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the dataset from a manifest and data roots."""
        super().__init__()
        assert split in ("train", "val", "test")
        assert abs(sum(mix_probabilities) - 1.0) < 1e-6

        self.manifest_path = Path(manifest_path)
        self.audio_dir = Path(audio_dir)
        self.landmark_dir = Path(landmark_dir)
        self.noise_dir = Path(noise_dir)
        self.split = split
        self.snr_range = snr_range
        self.sir_range = sir_range
        self.mix_probabilities = mix_probabilities
        self.num_samples = num_samples

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.metadata: dict[str, Any] = manifest["metadata"]
        self.entries: list[dict[str, str]] = manifest["splits"][split]

        self.entries_by_speaker: dict[str, list[dict[str, str]]] = {}
        for entry in self.entries:
            speaker = entry["speaker"]
            self.entries_by_speaker.setdefault(speaker, []).append(entry)
        self.all_speakers = list(self.entries_by_speaker.keys())

        self.noise_paths = sorted(self.noise_dir.rglob("*.wav"))
        if len(self.noise_paths) == 0:
            raise ValueError(f"No .wav files found in {self.noise_dir}")

        if split in ("val", "test") and seed is None:
            seed = 42
        self.seed = seed

    def __len__(self) -> int:
        """Return number of manifest entries in the selected split."""
        return len(self.entries)

    def _load_audio(self, audio_relpath: str) -> torch.Tensor:
        """Load a WAV file as mono float32 and pad/crop to ``num_samples``."""
        path = self.audio_dir / audio_relpath
        wav, sample_rate = sf.read(str(path), dtype="float32")
        expected_sample_rate = int(self.metadata["sample_rate"])
        assert sample_rate == expected_sample_rate, (
            f"Expected sample rate {expected_sample_rate}, got {sample_rate} in {path}"
        )

        if wav.ndim > 1:
            wav = wav.mean(axis=-1)

        wav = self._pad_or_crop(np.asarray(wav, dtype=np.float32))
        return torch.from_numpy(wav)

    def _load_landmarks(self, landmark_relpath: str) -> torch.Tensor:
        """Load a landmark ``.npy`` file and validate its shape."""
        path = self.landmark_dir / landmark_relpath
        landmarks = np.load(str(path))
        expected_frames = int(self.metadata["frames_per_clip"])
        expected_landmarks = int(self.metadata["num_lip_landmarks"])
        expected_shape = (expected_frames, expected_landmarks, 3)
        assert landmarks.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {landmarks.shape} in {path}"
        )
        return torch.from_numpy(landmarks.astype(np.float32, copy=False))

    def _load_random_noise_segment(self, rng: random.Random) -> torch.Tensor:
        """Load a random noise clip and crop/tile it to ``num_samples``."""
        noise_path = rng.choice(self.noise_paths)
        wav, sample_rate = sf.read(str(noise_path), dtype="float32")
        expected_sample_rate = int(self.metadata["sample_rate"])
        if sample_rate != expected_sample_rate:
            raise ValueError(
                f"Noise file {noise_path} has sr={sample_rate}, expected "
                f"{expected_sample_rate}. Resample your noise corpus to 16 kHz "
                "before training."
            )

        if wav.ndim > 1:
            wav = wav.mean(axis=-1)

        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) == 0:
            raise ValueError(f"Noise file {noise_path} is empty")
        if len(wav) > self.num_samples:
            start = rng.randint(0, len(wav) - self.num_samples)
            wav = wav[start : start + self.num_samples]
        elif len(wav) < self.num_samples:
            repeats = (self.num_samples // len(wav)) + 1
            wav = np.tile(wav, repeats)[: self.num_samples]

        return torch.from_numpy(wav.astype(np.float32, copy=False))

    def _sample_interferer(self, target_speaker: str, rng: random.Random) -> torch.Tensor:
        """Sample an interferer waveform from a speaker different from target."""
        other_speakers = [speaker for speaker in self.all_speakers if speaker != target_speaker]
        if len(other_speakers) == 0:
            return torch.zeros(self.num_samples, dtype=torch.float32)

        interferer_speaker = rng.choice(other_speakers)
        interferer_entry = rng.choice(self.entries_by_speaker[interferer_speaker])
        return self._load_audio(interferer_entry["audio_path"])

    def _sample_mix_type(self, rng: random.Random) -> int:
        """Sample mix type: 0 target+noise, 1 target+interferer+noise, 2 target+interferer."""
        random_value = rng.random()
        if random_value < self.mix_probabilities[0]:
            return 0
        if random_value < self.mix_probabilities[0] + self.mix_probabilities[1]:
            return 1
        return 2

    def _pad_or_crop(self, wav: np.ndarray) -> np.ndarray:
        """Pad with zeros or crop from the front to ``num_samples``."""
        if len(wav) < self.num_samples:
            wav = np.pad(wav, (0, self.num_samples - len(wav)))
        elif len(wav) > self.num_samples:
            wav = wav[: self.num_samples]
        return wav.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one clean clip and generate one noisy mixture."""
        if self.seed is not None:
            rng = random.Random(self.seed + idx)
        else:
            rng = random.Random()

        entry = self.entries[idx]
        clean = self._load_audio(entry["audio_path"])
        landmarks = self._load_landmarks(entry["landmark_path"])

        mix_type = self._sample_mix_type(rng)
        use_noise = mix_type in (0, 1)
        use_interferer = mix_type in (1, 2)

        if use_noise:
            noise = self._load_random_noise_segment(rng)
        else:
            noise = torch.zeros(self.num_samples, dtype=torch.float32)

        if use_interferer:
            interferer = self._sample_interferer(entry["speaker"], rng)
        else:
            interferer = torch.zeros(self.num_samples, dtype=torch.float32)

        snr_db = rng.uniform(*self.snr_range)
        sir_db = rng.uniform(*self.sir_range)

        noisy = mix_audio(
            target=clean.unsqueeze(0),
            noise=noise.unsqueeze(0),
            interferer=interferer.unsqueeze(0),
            snr_db=torch.tensor([snr_db], dtype=torch.float32),
            sir_db=torch.tensor([sir_db], dtype=torch.float32),
            use_interferer=torch.tensor([use_interferer], dtype=torch.bool),
        ).squeeze(0)

        return {
            "noisy": noisy,
            "clean": clean,
            "landmarks": landmarks,
            "speaker": entry["speaker"],
            "clip_id": entry["clip_id"],
            "mix_type": mix_type,
            "snr_db": snr_db,
            "sir_db": sir_db,
        }


def grid_av_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch tensor fields and keep lightweight metadata for inspection."""
    return {
        "noisy": torch.stack([sample["noisy"] for sample in batch]),
        "clean": torch.stack([sample["clean"] for sample in batch]),
        "landmarks": torch.stack([sample["landmarks"] for sample in batch]),
        "speakers": [sample["speaker"] for sample in batch],
        "clip_ids": [sample["clip_id"] for sample in batch],
        "mix_types": torch.tensor([sample["mix_type"] for sample in batch], dtype=torch.long),
        "snr_db": torch.tensor([sample["snr_db"] for sample in batch], dtype=torch.float32),
        "sir_db": torch.tensor([sample["sir_db"] for sample in batch], dtype=torch.float32),
    }
