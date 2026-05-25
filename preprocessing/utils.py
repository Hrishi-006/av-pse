"""Shared preprocessing helpers.

Usage example:
    from pathlib import Path
    from preprocessing.utils import get_lip_landmark_indices, walk_video_files

    indices = get_lip_landmark_indices()
    videos = list(walk_video_files(Path("/path/to/GRID")))
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterator


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Create a logger that writes to stdout and a file.

    Args:
        name: Logger name.
        log_file: Destination log file.

    Returns:
        Configured logger.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_lip_landmark_indices() -> list[int]:
    """Return sorted unique MediaPipe lip landmark indices.

    Returns:
        Sorted list of landmark indices from ``FACEMESH_LIPS``.
    """
    face_mesh = get_face_mesh_module()
    pairs = face_mesh.FACEMESH_LIPS
    return sorted({idx for pair in pairs for idx in pair})


def get_face_mesh_module() -> Any:
    """Return MediaPipe's ``face_mesh`` solutions module.

    Some Python environments expose MediaPipe solutions at
    ``mediapipe.solutions`` while others expose them via importable
    submodules. The extraction pipeline requires the solutions API.
    """
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return mp.solutions.face_mesh
    except Exception:
        pass

    try:
        from mediapipe.python.solutions import face_mesh

        return face_mesh
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe FaceMesh solutions API is unavailable. Install a mediapipe "
            "build that provides mediapipe.solutions.face_mesh."
        ) from exc


def _iter_speaker_dirs(root: Path) -> Iterator[Path]:
    """Yield speaker directories sorted by name."""
    if not root.exists():
        return
    yield from sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("s"))


def walk_video_files(root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(speaker_id, clip_id, path)`` for GRID videos.

    Supports both ``root/s1/*.mpg`` and ``root/s1/video/mpg_6000/*.mpg``.
    """
    for speaker_dir in _iter_speaker_dirs(root):
        candidates = [speaker_dir, speaker_dir / "video" / "mpg_6000"]
        seen: set[Path] = set()
        for video_dir in candidates:
            if not video_dir.is_dir():
                continue
            for path in sorted(video_dir.glob("*.mpg")):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield speaker_dir.name, path.stem, path


def walk_audio_files(root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(speaker_id, clip_id, path)`` for GRID audio files.

    Supports both ``root/s1/*.wav`` and ``root/s1/audio/*.wav``.
    """
    for speaker_dir in _iter_speaker_dirs(root):
        candidates = [speaker_dir, speaker_dir / "audio"]
        seen: set[Path] = set()
        for audio_dir in candidates:
            if not audio_dir.is_dir():
                continue
            for path in sorted(audio_dir.glob("*.wav")):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield speaker_dir.name, path.stem, path
