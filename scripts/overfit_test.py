"""Overfit sanity test for AV-PSE.

Trains on a fixed set of 4 clips for N iterations and verifies that loss
drops by >=80% and final SI-SNR exceeds 15 dB.  Proves the full pipeline
(STFT → BandSplit → VisualConditioning → BandSequenceRNN → MaskDecoder →
ISTFT → MultiResolutionLoss) is wired correctly before committing GPU time.

Usage:
    python scripts/overfit_test.py
    python scripts/overfit_test.py \
        --audio_dir /path/to/GRID/s1/audio \
        --landmark_dir preprocessed/landmarks/s1 \
        --num_iterations 500 \
        --output_dir overfit_output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.mixing import scale_to_snr
from losses.multi_res_loss import MultiResolutionLoss
from models.av_bsrnn import AVBSRNN, AVBSRNNConfig


# ── Clip registry ────────────────────────────────────────────────────────────

# Four clips from s1 whose audio and landmarks are both known-good.
DEFAULT_CLIPS = ["bbaf2n", "bbaf3s", "bbaf4p", "bbaf5a"]
NUM_SAMPLES = 48000  # 3 s @ 16 kHz


# ── SI-SNR metric (no external library) ─────────────────────────────────────

def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SNR in dB.

    Args:
        estimate: ``[B, S]`` enhanced waveform.
        target:   ``[B, S]`` clean reference waveform.

    Returns:
        Scalar mean SI-SNR in dB over the batch.
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    scale = dot / target_energy
    projection = scale * target
    noise = estimate - projection
    ratio = (projection ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps).mean()


# ── Dataset ──────────────────────────────────────────────────────────────────

class SmallOverfitDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed dataset that returns the same pre-loaded tensors every call.

    Audio is padded/cropped to ``NUM_SAMPLES``. Noisy version is generated
    once at init by adding Gaussian noise scaled to ``snr_db``.
    """

    def __init__(
        self,
        audio_dir: Path,
        landmark_dir: Path,
        clip_ids: list[str],
        snr_db: float = 5.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)

        self.clips: list[dict[str, torch.Tensor]] = []
        for clip_id in clip_ids:
            audio_path = audio_dir / f"{clip_id}.wav"
            landmark_path = landmark_dir / f"{clip_id}.npy"

            wav, sr = sf.read(str(audio_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            wav_t = torch.from_numpy(wav)
            # Pad or crop to exactly NUM_SAMPLES
            if wav_t.shape[0] < NUM_SAMPLES:
                wav_t = F.pad(wav_t, (0, NUM_SAMPLES - wav_t.shape[0]))
            else:
                wav_t = wav_t[:NUM_SAMPLES]

            landmarks = torch.from_numpy(np.load(str(landmark_path)).astype(np.float32))

            # Generate fixed noise: same seed per clip for reproducibility
            noise = torch.randn(NUM_SAMPLES, generator=rng)
            # Scale noise to achieve target SNR
            scaled_noise = scale_to_snr(
                wav_t.unsqueeze(0),
                noise.unsqueeze(0),
                torch.tensor([snr_db]),
            ).squeeze(0)
            noisy = (wav_t + scaled_noise).clamp(-0.99, 0.99)

            self.clips.append(
                {"noisy": noisy, "clean": wav_t, "landmarks": landmarks, "clip_id": clip_id}
            )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.clips[idx]


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "noisy": torch.stack([b["noisy"] for b in batch]),
        "clean": torch.stack([b["clean"] for b in batch]),
        "landmarks": torch.stack([b["landmarks"] for b in batch]),
        "clip_ids": [b["clip_id"] for b in batch],
    }


# ── Gradient norm helpers ─────────────────────────────────────────────────────

def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


# ── Logging ──────────────────────────────────────────────────────────────────

def build_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("overfit")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(output_dir / "training_log.txt", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── Audio saving ─────────────────────────────────────────────────────────────

def save_wav(path: Path, wav: torch.Tensor, sr: int = 16000) -> None:
    arr = wav.detach().cpu().float().numpy()
    peak = np.abs(arr).max()
    if peak > 1e-6:
        arr = arr / peak * 0.95
    sf.write(str(path), arr, sr, subtype="PCM_16")


# ── Loss curve ───────────────────────────────────────────────────────────────

def save_loss_curve(losses: list[float], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses, linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Multi-resolution loss")
        ax.set_title("Overfit sanity test — loss curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass  # matplotlib not installed — skip silently


# ── Main training routine ─────────────────────────────────────────────────────

def run_overfit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "sample_outputs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(output_dir)

    device = torch.device(args.device)
    logger.info("=" * 60)
    logger.info("AV-PSE OVERFIT SANITY TEST")
    logger.info("=" * 60)
    logger.info("device       : %s", device)
    logger.info("audio_dir    : %s", args.audio_dir)
    logger.info("landmark_dir : %s", args.landmark_dir)
    logger.info("clips        : %s", DEFAULT_CLIPS)
    logger.info("iterations   : %d", args.num_iterations)
    logger.info("learning_rate: %g", args.learning_rate)
    logger.info("")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = SmallOverfitDataset(
        audio_dir=Path(args.audio_dir),
        landmark_dir=Path(args.landmark_dir),
        clip_ids=DEFAULT_CLIPS,
        snr_db=5.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=len(DEFAULT_CLIPS),
        shuffle=False,
        collate_fn=collate,
    )
    batch = next(iter(loader))
    noisy = batch["noisy"].to(device)
    clean = batch["clean"].to(device)
    landmarks = batch["landmarks"].to(device)

    # Sanity: verify noisy ≠ clean
    diff = (noisy - clean).abs().mean().item()
    logger.info("mean |noisy-clean| = %.4f  (should be > 0)", diff)
    assert diff > 1e-4, "noisy and clean tensors are identical — mixing bug"

    # ── Model ─────────────────────────────────────────────────────────────────
    config = AVBSRNNConfig(use_visual_conditioning=True, num_layers=2)
    model = AVBSRNN(config).to(device)
    param_counts = model.count_parameters()
    logger.info("parameters   : %s total", f"{param_counts['total']:,}")
    logger.info("")

    # ── Loss + optimizer ──────────────────────────────────────────────────────
    loss_fn = MultiResolutionLoss(sample_rate=16000)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("%-8s  %-10s  %-10s", "iter", "loss", "SI-SNR(dB)")
    logger.info("-" * 35)

    losses: list[float] = []
    model.train()
    t0 = time.monotonic()

    # Track output at iter 0 for change-detection diagnostic
    with torch.no_grad():
        init_out = model(noisy, landmarks)["waveform"].detach().clone()

    for it in range(args.num_iterations):
        optimizer.zero_grad()
        out = model(noisy, landmarks)
        enhanced = out["waveform"]
        loss = loss_fn(enhanced, clean)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if it % 10 == 0:
            with torch.no_grad():
                snr_val = si_snr(enhanced.detach(), clean).item()
            logger.info("%5d     %.5f    %+.2f", it, loss_val, snr_val)

        if it % 50 == 0:
            gn = {
                "band_split":        grad_norm(model.band_split),
                "visual_cond":       grad_norm(model.visual_conditioning) if model.visual_conditioning else 0.0,
                "band_seq_rnn":      grad_norm(model.band_sequence_rnn),
                "mask_decoder":      grad_norm(model.mask_decoder),
            }
            logger.info(
                "  grad_norms  band_split=%.3f  visual=%.3f  rnn=%.3f  mask=%.3f",
                gn["band_split"], gn["visual_cond"], gn["band_seq_rnn"], gn["mask_decoder"],
            )

    elapsed = time.monotonic() - t0
    logger.info("")
    logger.info("Training finished in %.1f s", elapsed)

    # ── Final metrics ─────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        final_out = model(noisy, landmarks)
        final_enhanced = final_out["waveform"]
        final_loss = loss_fn(final_enhanced, clean).item()
        final_out_spec = final_out["waveform"]

        # Per-clip SI-SNR
        si_snr_per_clip: list[float] = []
        for i in range(len(DEFAULT_CLIPS)):
            val = si_snr(final_enhanced[i : i + 1], clean[i : i + 1]).item()
            si_snr_per_clip.append(round(val, 2))

    start_loss = losses[0]
    drop_pct = 100.0 * (start_loss - final_loss) / (start_loss + 1e-8)
    mean_si_snr = float(np.mean(si_snr_per_clip))

    logger.info("")
    logger.info("─" * 50)
    logger.info("start_loss  : %.5f", start_loss)
    logger.info("final_loss  : %.5f", final_loss)
    logger.info("loss_drop   : %.1f%%", drop_pct)
    logger.info("SI-SNR/clip : %s dB", si_snr_per_clip)
    logger.info("mean SI-SNR : %.2f dB", mean_si_snr)

    # ── Diagnostic: did model output change? ──────────────────────────────────
    output_change = (final_enhanced.detach() - init_out).abs().mean().item()
    logger.info("output_delta: %.6f  (should be > 0)", output_change)

    # ── Save WAVs ─────────────────────────────────────────────────────────────
    for i, clip_id in enumerate(DEFAULT_CLIPS):
        save_wav(sample_dir / f"clip_{i}_{clip_id}_enhanced.wav", final_enhanced[i])
        save_wav(sample_dir / f"clip_{i}_{clip_id}_clean.wav",    clean[i].cpu())
        save_wav(sample_dir / f"clip_{i}_{clip_id}_noisy.wav",    noisy[i].cpu())
    logger.info("Saved WAVs → %s", sample_dir)

    # ── Loss curve ────────────────────────────────────────────────────────────
    save_loss_curve(losses, output_dir)

    # ── Verdict ───────────────────────────────────────────────────────────────
    checks = {
        "loss_drop_pct >= 80": drop_pct >= 80.0,
        "mean_si_snr > 15 dB": mean_si_snr > 15.0,
        "output_changed":       output_change > 1e-6,
    }
    passed = all(checks.values())

    logger.info("")
    logger.info("─" * 50)
    for check, ok in checks.items():
        logger.info("  %s  %s", "✓" if ok else "✗", check)
    logger.info("─" * 50)

    if passed:
        logger.info("✅  OVERFIT TEST PASSED")
    else:
        logger.info("❌  OVERFIT TEST FAILED")
        failed = [k for k, v in checks.items() if not v]
        logger.info("    Failed checks: %s", failed)
        if not checks["loss_drop_pct >= 80"]:
            logger.info(
                "    DIAGNOSTIC: loss only dropped %.1f%% — check grad norms above "
                "for dead components, or try a higher learning rate.", drop_pct
            )
        if not checks["mean_si_snr > 15 dB"]:
            logger.info(
                "    DIAGNOSTIC: SI-SNR=%.2f dB — model may need more iterations "
                "or the loss function may not be driving SI-SNR directly.", mean_si_snr
            )
        if not checks["output_changed"]:
            logger.info(
                "    DIAGNOSTIC: model output did not change — frozen weights or "
                "zero learning rate."
            )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    metrics = {
        "start_loss":    round(start_loss, 6),
        "final_loss":    round(final_loss, 6),
        "loss_drop_pct": round(drop_pct, 2),
        "si_snr_per_clip": {
            clip_id: si_snr_per_clip[i]
            for i, clip_id in enumerate(DEFAULT_CLIPS)
        },
        "mean_si_snr_db":  round(mean_si_snr, 2),
        "num_iterations":  args.num_iterations,
        "checks":          checks,
        "passed":          passed,
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics → %s", output_dir / "final_metrics.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AV-PSE overfit sanity test")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/Users/hrishikeshbingewar/Downloads/av_project/GRID/s1/audio",
    )
    parser.add_argument(
        "--landmark_dir",
        type=str,
        default="preprocessed/landmarks/s1",
    )
    parser.add_argument("--output_dir",      type=str,   default="overfit_output")
    parser.add_argument("--num_iterations",  type=int,   default=500)
    parser.add_argument("--learning_rate",   type=float, default=1e-3)
    parser.add_argument("--device",          type=str,   default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run_overfit(parse_args())
