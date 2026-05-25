from __future__ import annotations

import json
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocessing.build_manifest import build_manifest
from preprocessing.extract_landmarks import (
    VideoTask,
    adjust_frame_count,
    mean_center_landmarks,
    process_video,
)
from preprocessing.utils import get_lip_landmark_indices


def _require_mediapipe_solutions() -> None:
    pytest.importorskip("mediapipe")
    try:
        get_lip_landmark_indices()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _write_silent_wav(path: Path, sample_rate: int, samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


def test_lip_landmark_indices_are_deterministic() -> None:
    _require_mediapipe_solutions()
    first = get_lip_landmark_indices()
    second = get_lip_landmark_indices()
    assert first == second
    assert first == sorted(first)
    assert 30 <= len(first) <= 50


def test_synthetic_video_with_no_face_is_skipped(tmp_path: Path) -> None:
    _require_mediapipe_solutions()
    video_path = tmp_path / "s1" / "video" / "mpg_6000" / "noise.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (64, 64),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(123)
    for _ in range(75):
        frame = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    task = VideoTask(
        speaker_id="s1",
        clip_id="noise",
        video_path=video_path,
        output_dir=tmp_path / "out",
        target_fps=25,
        num_frames=75,
        lip_indices=tuple(get_lip_landmark_indices()),
    )
    result = process_video(task)
    assert result.skipped
    assert not result.succeeded
    assert result.output_path is None


def test_normalization_removes_translation() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(10, 40, 3)).astype(np.float32)
    shifted = base + np.array([0.3, 0.2, 0.1], dtype=np.float32)

    centered_base = mean_center_landmarks(base)
    centered_shifted = mean_center_landmarks(shifted)

    np.testing.assert_allclose(centered_base, centered_shifted, atol=1e-6)


def test_padding_to_target_frames() -> None:
    landmarks = np.arange(50 * 40 * 3, dtype=np.float32).reshape(50, 40, 3)
    padded = adjust_frame_count(landmarks, 75)

    assert padded.shape == (75, 40, 3)
    np.testing.assert_allclose(padded[:50], landmarks)
    for i in range(50, 75):
        np.testing.assert_allclose(padded[i], landmarks[49])


def test_manifest_builder_skeleton(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    landmark_root = tmp_path / "landmarks"

    pairs = [("s1", "clip1"), ("s1", "clip2"), ("s2", "clip1"), ("s2", "clip2")]
    for speaker, clip_id in pairs:
        _write_silent_wav(audio_root / speaker / "audio" / f"{clip_id}.wav", sample_rate=16000, samples=160)
        landmark_path = landmark_root / speaker / f"{clip_id}.npy"
        landmark_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(landmark_path, np.zeros((5, 40, 3), dtype=np.float32))

    manifest, dropped = build_manifest(
        audio_dir=audio_root,
        landmark_dir=landmark_root,
        train_speakers=["s1"],
        val_speakers=["s2"],
        test_speakers=[],
        sample_rate=16000,
        audio_samples_per_clip=160,
        video_fps=25,
        frames_per_clip=5,
        num_lip_landmarks=40,
    )

    output_path = tmp_path / "manifest.json"
    output_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert dropped == []
    assert set(loaded) == {"metadata", "splits", "speakers"}
    assert len(loaded["splits"]["train"]) == 2
    assert len(loaded["splits"]["val"]) == 2
    assert loaded["splits"]["test"] == []
    assert set(loaded["speakers"]["train"]).isdisjoint(set(loaded["speakers"]["val"]))
    for split in ("train", "val"):
        for entry in loaded["splits"][split]:
            assert "audio_path" in entry
            assert "landmark_path" in entry
