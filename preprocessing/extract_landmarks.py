"""Extract MediaPipe lip landmarks from GRID videos.

Usage example:
    python preprocessing/extract_landmarks.py \
        --video_dir /path/to/GRID \
        --output_dir /path/to/landmarks_out \
        --target_fps 25 \
        --num_frames 75 \
        --num_workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp_pool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

try:
    from preprocessing.utils import get_face_mesh_module, get_lip_landmark_indices, setup_logger, walk_video_files
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import get_face_mesh_module, get_lip_landmark_indices, setup_logger, walk_video_files


_FACE_MESH: Any | None = None


@dataclass(frozen=True)
class VideoTask:
    """Single video extraction task."""

    speaker_id: str
    clip_id: str
    video_path: Path
    output_dir: Path
    target_fps: int
    num_frames: int
    lip_indices: tuple[int, ...]


@dataclass(frozen=True)
class VideoResult:
    """Result of processing one video."""

    speaker_id: str
    clip_id: str
    video_path: Path
    output_path: Path | None
    succeeded: bool
    skipped: bool
    message: str
    warnings: tuple[str, ...]


def _create_face_mesh() -> Any:
    """Create one MediaPipe FaceMesh instance for the current process."""
    face_mesh = get_face_mesh_module()

    return face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _init_worker() -> None:
    """Initialize MediaPipe FaceMesh once per worker process."""
    global _FACE_MESH
    _FACE_MESH = _create_face_mesh()


def mean_center_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Mean-center lip landmarks per frame.

    Args:
        landmarks: Array with shape ``[T, L, 3]``.

    Returns:
        Mean-centered array with shape ``[T, L, 3]`` and dtype ``float32``.
    """
    if landmarks.ndim != 3 or landmarks.shape[2] != 3:
        raise ValueError(f"Expected [T, L, 3], got {landmarks.shape}")
    centered = landmarks - landmarks.mean(axis=1, keepdims=True)
    return centered.astype(np.float32)


def adjust_frame_count(landmarks: np.ndarray, target_frames: int) -> np.ndarray:
    """Center-crop or last-frame-pad landmarks to ``target_frames``.

    Args:
        landmarks: Array with shape ``[T, L, 3]``.
        target_frames: Required frame count.

    Returns:
        Array with shape ``[target_frames, L, 3]``.
    """
    if landmarks.ndim != 3:
        raise ValueError(f"Expected [T, L, C], got {landmarks.shape}")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    frame_count = landmarks.shape[0]
    if frame_count == 0:
        raise ValueError("Cannot pad/trim an empty landmark array")
    if frame_count > target_frames:
        start = (frame_count - target_frames) // 2
        return landmarks[start : start + target_frames].astype(np.float32)
    if frame_count < target_frames:
        pad_count = target_frames - frame_count
        pad = np.repeat(landmarks[-1:, :, :], pad_count, axis=0)
        return np.concatenate([landmarks, pad], axis=0).astype(np.float32)
    return landmarks.astype(np.float32)


def interpolate_missing_frames(landmarks: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Linearly interpolate missing frames along time.

    Args:
        landmarks: Array with shape ``[T, L, 3]`` containing NaNs for missing frames.
        missing: Boolean array with shape ``[T]``.

    Returns:
        Interpolated array with shape ``[T, L, 3]``.
    """
    if landmarks.shape[0] != missing.shape[0]:
        raise ValueError("landmarks and missing must share the same time dimension")

    valid = ~missing
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        raise ValueError("Cannot interpolate: no valid frames")
    if valid_idx.size == 1:
        return np.repeat(landmarks[valid_idx[0] : valid_idx[0] + 1], landmarks.shape[0], axis=0).astype(np.float32)

    flat = landmarks.reshape(landmarks.shape[0], -1)
    output = np.empty_like(flat, dtype=np.float32)
    frame_idx = np.arange(landmarks.shape[0], dtype=np.float32)

    for dim in range(flat.shape[1]):
        output[:, dim] = np.interp(frame_idx, valid_idx.astype(np.float32), flat[valid_idx, dim]).astype(np.float32)

    return output.reshape(landmarks.shape).astype(np.float32)


def process_video(task: VideoTask) -> VideoResult:
    """Process one video and save a landmark ``.npy`` file.

    Input video frames are decoded as ``[H, W, 3]`` BGR arrays.
    Saved landmarks have shape ``[num_frames, num_lip_landmarks, 3]``.
    """
    global _FACE_MESH
    warnings: list[str] = []
    output_path = task.output_dir / task.speaker_id / f"{task.clip_id}.npy"

    try:
        if _FACE_MESH is None:
            _FACE_MESH = _create_face_mesh()

        cap = cv2.VideoCapture(str(task.video_path))
        if not cap.isOpened():
            return VideoResult(task.speaker_id, task.clip_id, task.video_path, None, False, True, "video_open_failed", ())

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0 and not math.isclose(fps, float(task.target_fps), abs_tol=0.5):
            warnings.append(f"fps_mismatch expected={task.target_fps} actual={fps:.3f}")

        frames: list[np.ndarray] = []
        missing_flags: list[bool] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _FACE_MESH.process(rgb)
            if result.multi_face_landmarks:
                face = result.multi_face_landmarks[0]
                coords = np.array(
                    [[face.landmark[idx].x, face.landmark[idx].y, face.landmark[idx].z] for idx in task.lip_indices],
                    dtype=np.float32,
                )
                frames.append(coords)
                missing_flags.append(False)
            else:
                frames.append(np.full((len(task.lip_indices), 3), np.nan, dtype=np.float32))
                missing_flags.append(True)

        cap.release()

        if not frames:
            return VideoResult(task.speaker_id, task.clip_id, task.video_path, None, False, True, "no_frames", tuple(warnings))

        landmarks = np.stack(frames).astype(np.float32)
        missing = np.array(missing_flags, dtype=bool)
        missing_ratio = float(missing.mean())
        if missing_ratio >= 0.10:
            message = f"too_many_missing_frames missing_ratio={missing_ratio:.4f}"
            return VideoResult(task.speaker_id, task.clip_id, task.video_path, None, False, True, message, tuple(warnings))

        if missing.any():
            landmarks = interpolate_missing_frames(landmarks, missing)

        landmarks = adjust_frame_count(landmarks, task.num_frames)
        landmarks = mean_center_landmarks(landmarks)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, landmarks)

        return VideoResult(task.speaker_id, task.clip_id, task.video_path, output_path, True, False, "ok", tuple(warnings))

    except Exception as exc:  # noqa: BLE001 - bad clips should not stop the run
        return VideoResult(task.speaker_id, task.clip_id, task.video_path, None, False, True, f"error: {exc}", tuple(warnings))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract MediaPipe lip landmarks from GRID videos.")
    parser.add_argument("--video_dir", type=Path, required=True, help="GRID video root containing speaker subdirs.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for output .npy landmarks.")
    parser.add_argument("--target_fps", type=int, default=25, help="Expected video FPS.")
    parser.add_argument("--num_frames", type=int, default=75, help="Required output frames per clip.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes.")
    return parser.parse_args()


def main() -> None:
    """Run landmark extraction from the command line."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("extract_landmarks", output_dir / "landmarks_log.txt")
    lip_indices = get_lip_landmark_indices()

    logger.info("Lip landmark indices (%d): %s", len(lip_indices), lip_indices)
    logger.info("Video dir: %s", args.video_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Target FPS: %d | Frames per clip: %d", args.target_fps, args.num_frames)

    tasks = [
        VideoTask(
            speaker_id=speaker_id,
            clip_id=clip_id,
            video_path=path,
            output_dir=output_dir,
            target_fps=args.target_fps,
            num_frames=args.num_frames,
            lip_indices=tuple(lip_indices),
        )
        for speaker_id, clip_id, path in walk_video_files(args.video_dir)
    ]
    logger.info("Found %d video files", len(tasks))

    results: list[VideoResult] = []
    if args.num_workers <= 1:
        _init_worker()
        for task in tqdm(tasks, desc="Extracting landmarks"):
            results.append(process_video(task))
    else:
        with mp_pool.Pool(processes=args.num_workers, initializer=_init_worker) as pool:
            iterator = pool.imap_unordered(process_video, tasks)
            for result in tqdm(iterator, total=len(tasks), desc="Extracting landmarks"):
                results.append(result)

    skipped = [result for result in results if result.skipped]
    succeeded = [result for result in results if result.succeeded]

    with (output_dir / "skipped.txt").open("w", encoding="utf-8") as handle:
        for result in sorted(skipped, key=lambda item: (item.speaker_id, item.clip_id)):
            handle.write(f"{result.video_path}\t{result.message}\n")

    for result in sorted(results, key=lambda item: (item.speaker_id, item.clip_id)):
        for warning in result.warnings:
            logger.warning("%s/%s: %s", result.speaker_id, result.clip_id, warning)
        if result.skipped:
            logger.error("%s/%s skipped: %s", result.speaker_id, result.clip_id, result.message)

    metadata = {
        "lip_landmark_indices": lip_indices,
        "num_landmarks": len(lip_indices),
        "coord_dims": 3,
        "normalization": "per_frame_mean_centered",
        "target_fps": args.target_fps,
        "num_frames": args.num_frames,
        "total_processed": len(results),
        "total_succeeded": len(succeeded),
        "total_skipped": len(skipped),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Processed=%d succeeded=%d skipped=%d", len(results), len(succeeded), len(skipped))


if __name__ == "__main__":
    main()
