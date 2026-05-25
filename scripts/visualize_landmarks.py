"""Visualize a single clip's lip landmarks to sanity-check extraction.

Usage:
    python scripts/visualize_landmarks.py <path_to_npy>
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    npy_path = Path(sys.argv[1])
    if not npy_path.exists():
        print(f"File not found: {npy_path}")
        sys.exit(1)

    lm = np.load(npy_path)
    print(f"Loaded: {npy_path}")
    print(f"  Shape:  {lm.shape}")
    print(f"  Dtype:  {lm.dtype}")
    print(f"  Min:    {lm.min():.6f}")
    print(f"  Max:    {lm.max():.6f}")
    print(f"  Frame 0 mean (should be ~0 if mean-centered): {lm[0].mean(axis=0)}")

    # Plot first, middle, last frames side by side
    num_frames = lm.shape[0]
    indices = [0, num_frames // 2, num_frames - 1]
    titles = ["First frame", "Middle frame", "Last frame"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, idx, title in zip(axes, indices, titles):
        # Flip y because image coordinates have origin at top-left
        ax.scatter(lm[idx, :, 0], -lm[idx, :, 1], s=20)
        ax.set_title(f"{title} (frame {idx})")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    output_path = npy_path.parent / f"{npy_path.stem}_preview.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"\nSaved preview to: {output_path}")


if __name__ == "__main__":
    main()