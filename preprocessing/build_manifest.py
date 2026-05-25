"""Build a manifest pairing GRID audio with extracted lip landmarks.

Usage example:
    python preprocessing/build_manifest.py \
        --audio_dir /path/to/GRID \
        --landmark_dir /path/to/landmarks_out \
        --output /path/to/landmarks_out/manifest.json \
        --train_speakers s1,s2,s3 \
        --val_speakers s27,s28 \
        --test_speakers s31,s32
"""

from __future__ import annotations

import argparse
import json
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    from preprocessing.utils import setup_logger, walk_audio_files
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import setup_logger, walk_audio_files


def parse_speaker_list(value: str) -> list[str]:
    """Parse a comma-separated speaker list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def read_wav_info(path: Path) -> tuple[int, int, int]:
    """Read WAV sample rate, frame count, and channel count.

    Args:
        path: WAV path.

    Returns:
        ``(sample_rate, num_frames, num_channels)``.
    """
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnframes(), handle.getnchannels()


def validate_audio(path: Path, sample_rate: int, expected_samples: int) -> tuple[bool, str]:
    """Validate a WAV file against expected sample rate and length."""
    try:
        rate, frames, channels = read_wav_info(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"wav_read_failed: {exc}"
    if rate != sample_rate:
        return False, f"sample_rate_mismatch expected={sample_rate} actual={rate}"
    if channels != 1:
        return False, f"channel_mismatch expected=1 actual={channels}"
    if frames != expected_samples:
        return False, f"sample_count_mismatch expected={expected_samples} actual={frames}"
    return True, "ok"


def validate_landmarks(path: Path, frames_per_clip: int, num_landmarks: int) -> tuple[bool, str]:
    """Validate landmark file shape and dtype compatibility."""
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"npy_read_failed: {exc}"
    expected_shape = (frames_per_clip, num_landmarks, 3)
    if arr.shape != expected_shape:
        return False, f"landmark_shape_mismatch expected={expected_shape} actual={arr.shape}"
    if not np.issubdtype(arr.dtype, np.floating):
        return False, f"landmark_dtype_not_float actual={arr.dtype}"
    return True, "ok"


def build_manifest(
    audio_dir: Path,
    landmark_dir: Path,
    train_speakers: list[str],
    val_speakers: list[str],
    test_speakers: list[str],
    sample_rate: int,
    audio_samples_per_clip: int,
    video_fps: int,
    frames_per_clip: int,
    num_lip_landmarks: int,
    logger: Any | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build a manifest dictionary.

    Landmark input shape is ``[frames_per_clip, num_lip_landmarks, 3]``.
    Audio files must be mono WAV with ``audio_samples_per_clip`` samples.
    """
    audio_index = {(speaker, clip): path for speaker, clip, path in walk_audio_files(audio_dir)}

    split_by_speaker = {
        "train": set(train_speakers),
        "val": set(val_speakers),
        "test": set(test_speakers),
    }

    manifest: dict[str, Any] = {
        "metadata": {
            "sample_rate": sample_rate,
            "audio_samples_per_clip": audio_samples_per_clip,
            "video_fps": video_fps,
            "frames_per_clip": frames_per_clip,
            "num_lip_landmarks": num_lip_landmarks,
            "lip_normalization": "per_frame_mean_centered",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "splits": {"train": [], "val": [], "test": []},
        "speakers": {"train": train_speakers, "val": val_speakers, "test": test_speakers},
    }
    dropped: list[str] = []

    landmark_files = sorted(path for path in landmark_dir.glob("s*/*.npy") if path.is_file())
    for landmark_path in tqdm(landmark_files, desc="Building manifest"):
        speaker = landmark_path.parent.name
        clip_id = landmark_path.stem

        split = next((name for name, speakers in split_by_speaker.items() if speaker in speakers), None)
        if split is None:
            dropped.append(f"{speaker}/{clip_id}: speaker_not_in_split")
            continue

        audio_path = audio_index.get((speaker, clip_id))
        if audio_path is None:
            dropped.append(f"{speaker}/{clip_id}: missing_audio")
            continue

        audio_ok, audio_message = validate_audio(audio_path, sample_rate, audio_samples_per_clip)
        if not audio_ok:
            dropped.append(f"{speaker}/{clip_id}: {audio_message}")
            continue

        landmark_ok, landmark_message = validate_landmarks(landmark_path, frames_per_clip, num_lip_landmarks)
        if not landmark_ok:
            dropped.append(f"{speaker}/{clip_id}: {landmark_message}")
            continue

        entry = {
            "speaker": speaker,
            "clip_id": clip_id,
            "audio_path": str(audio_path.relative_to(audio_dir)),
            "landmark_path": str(landmark_path.relative_to(landmark_dir)),
        }
        manifest["splits"][split].append(entry)

    if logger is not None:
        for item in dropped:
            logger.warning("Dropped %s", item)

    return manifest, dropped


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build GRID audio-landmark manifest JSON.")
    parser.add_argument("--audio_dir", type=Path, required=True, help="GRID audio root.")
    parser.add_argument("--landmark_dir", type=Path, required=True, help="Extracted landmark root.")
    parser.add_argument("--output", type=Path, required=True, help="Manifest JSON output path.")
    parser.add_argument("--train_speakers", type=str, required=True, help="Comma-separated training speakers.")
    parser.add_argument("--val_speakers", type=str, required=True, help="Comma-separated validation speakers.")
    parser.add_argument("--test_speakers", type=str, required=True, help="Comma-separated test speakers.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Expected WAV sample rate.")
    parser.add_argument("--audio_samples_per_clip", type=int, default=48000, help="Expected WAV samples per clip.")
    parser.add_argument("--video_fps", type=int, default=25, help="Video FPS recorded in manifest metadata.")
    parser.add_argument("--frames_per_clip", type=int, default=75, help="Expected landmark frames per clip.")
    parser.add_argument("--num_lip_landmarks", type=int, default=40, help="Expected lip landmarks per frame.")
    return parser.parse_args()


def main() -> None:
    """Run manifest building from the command line."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("build_manifest", args.output.parent / "manifest_log.txt")

    train_speakers = parse_speaker_list(args.train_speakers)
    val_speakers = parse_speaker_list(args.val_speakers)
    test_speakers = parse_speaker_list(args.test_speakers)

    speaker_sets = [set(train_speakers), set(val_speakers), set(test_speakers)]
    if speaker_sets[0] & speaker_sets[1] or speaker_sets[0] & speaker_sets[2] or speaker_sets[1] & speaker_sets[2]:
        raise ValueError("train/val/test speaker lists must be disjoint")

    manifest, dropped = build_manifest(
        audio_dir=args.audio_dir,
        landmark_dir=args.landmark_dir,
        train_speakers=train_speakers,
        val_speakers=val_speakers,
        test_speakers=test_speakers,
        sample_rate=args.sample_rate,
        audio_samples_per_clip=args.audio_samples_per_clip,
        video_fps=args.video_fps,
        frames_per_clip=args.frames_per_clip,
        num_lip_landmarks=args.num_lip_landmarks,
        logger=logger,
    )

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    for split, entries in manifest["splits"].items():
        speakers = sorted({entry["speaker"] for entry in entries})
        logger.info("%s: clips=%d speakers=%d %s", split, len(entries), len(speakers), speakers)
    logger.info("Dropped clips: %d", len(dropped))
    logger.info("Wrote manifest: %s", args.output)


if __name__ == "__main__":
    main()

