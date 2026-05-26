"""Evaluate a trained AV-PSE checkpoint on the held-out test split.

Computes SI-SNR, PESQ-WB, and STOI for both the noisy baseline and the
enhanced output, then reports aggregate stats overall, by mix type, and by
input-SNR bin.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --config configs/default.yaml \
        --manifest_path /kaggle/input/grid-av/manifest.json \
        --audio_dir /kaggle/input/grid-audio \
        --landmark_dir /kaggle/input/grid-landmarks \
        --noise_dir /kaggle/input/demand-noise \
        --output_dir eval_output \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.grid_dataset import GRIDAVDataset, grid_av_collate
from models.av_bsrnn import AVBSRNN, AVBSRNNConfig


# ── Metric helpers ────────────────────────────────────────────────────────────

def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant SNR in dB for a single clip (1-D tensors).

    Args:
        estimate: Enhanced waveform ``[S]``.
        target:   Clean reference waveform ``[S]``.

    Returns:
        SI-SNR in dB as a Python float.
    """
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    dot = (estimate * target).sum()
    target_energy = (target ** 2).sum() + eps
    scale = dot / target_energy
    projection = scale * target
    noise = estimate - projection
    ratio = (projection ** 2).sum() / ((noise ** 2).sum() + eps)
    return 10 * torch.log10(ratio + eps).item()


def compute_pesq(ref_np: np.ndarray, est_np: np.ndarray, sr: int = 16000) -> float | None:
    """PESQ wideband score (range: -0.5 to 4.5). Returns ``None`` on failure.

    Args:
        ref_np: Clean reference, shape ``[S]``, float32.
        est_np: Enhanced estimate, shape ``[S]``, float32.
        sr:     Sample rate (must be 16000 for wideband mode).
    """
    try:
        from pesq import pesq
        ref = np.clip(ref_np, -1.0, 1.0).astype(np.float32)
        est = np.clip(est_np, -1.0, 1.0).astype(np.float32)
        return float(pesq(sr, ref, est, mode="wb"))
    except Exception:
        return None


def compute_stoi(ref_np: np.ndarray, est_np: np.ndarray, sr: int = 16000) -> float | None:
    """STOI score (range: 0–1). Returns ``None`` on failure.

    Args:
        ref_np: Clean reference, shape ``[S]``, float32.
        est_np: Enhanced estimate, shape ``[S]``, float32.
        sr:     Sample rate.
    """
    try:
        from pystoi import stoi
        ref = np.clip(ref_np, -1.0, 1.0).astype(np.float32)
        est = np.clip(est_np, -1.0, 1.0).astype(np.float32)
        return float(stoi(ref, est, sr, extended=False))
    except Exception:
        return None


def save_wav(path: Path, wav: torch.Tensor, sr: int = 16000) -> None:
    """Save a 1-D waveform tensor as a 16-bit PCM WAV file."""
    arr = np.clip(wav.detach().cpu().float().numpy(), -1.0, 1.0)
    sf.write(str(path), arr, sr, subtype="PCM_16")


# ── Model loading ─────────────────────────────────────────────────────────────

_log = logging.getLogger("av_pse.evaluate")


def _infer_arch_from_state(
    state: dict[str, Any],
    base_mc: dict[str, Any],
) -> dict[str, Any]:
    """Return a corrected copy of ``cfg["model"]`` using shapes from the checkpoint.

    Detects and fixes the three params most commonly varied between experiments:
    ``feat_dim``, ``num_layers``, and ``use_visual_conditioning``.  Everything
    else is taken verbatim from ``base_mc``.
    """
    mc = dict(base_mc)

    # feat_dim  ── band_split.projections.0.weight has shape [feat_dim, band_bins]
    proj_key = "band_split.projections.0.weight"
    if proj_key in state:
        inferred = state[proj_key].shape[0]
        if inferred != mc["feat_dim"]:
            _log.warning(
                "feat_dim mismatch: config=%d, checkpoint=%d — using checkpoint value.",
                mc["feat_dim"], inferred,
            )
            mc["feat_dim"] = inferred

    # num_layers ── count distinct index tokens in band_sequence_rnn.layers.<N>.*
    layer_indices = {
        int(k.split(".")[2])
        for k in state
        if k.startswith("band_sequence_rnn.layers.")
    }
    if layer_indices:
        inferred = len(layer_indices)
        if inferred != mc["num_layers"]:
            _log.warning(
                "num_layers mismatch: config=%d, checkpoint=%d — using checkpoint value.",
                mc["num_layers"], inferred,
            )
            mc["num_layers"] = inferred

    # use_visual_conditioning ── presence of any visual_conditioning.* key
    inferred_vis = any(k.startswith("visual_conditioning.") for k in state)
    if inferred_vis != mc["use_visual_conditioning"]:
        _log.warning(
            "use_visual_conditioning mismatch: config=%s, checkpoint=%s — using checkpoint value.",
            mc["use_visual_conditioning"], inferred_vis,
        )
        mc["use_visual_conditioning"] = inferred_vis

    return mc


def load_model(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> AVBSRNN:
    """Reconstruct AVBSRNN from a checkpoint and load its weights.

    Architecture priority (prevents size-mismatch crashes):
      1. ``model_cfg`` saved inside the checkpoint by train.py — fully
         self-describing, YAML is ignored for architecture.
      2. Auto-detected from state-dict shapes — fixes ``feat_dim`` /
         ``num_layers`` / ``use_visual_conditioning`` mismatches between
         the YAML and an older checkpoint.
      3. ``cfg["model"]`` verbatim — final fallback.

    Only ``model_state`` is loaded; optimizer / scheduler states are ignored.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    if "model_cfg" in ckpt:
        # Checkpoint saved by train.py after Stage 13 — fully self-describing.
        config = AVBSRNNConfig(**ckpt["model_cfg"])
        _log.info("Architecture loaded from checkpoint's saved model_cfg.")
    else:
        # Legacy checkpoint: patch cfg["model"] from state-dict shapes.
        mc = _infer_arch_from_state(state, cfg["model"])
        config = AVBSRNNConfig(
            sample_rate=mc["sample_rate"],
            n_fft=mc["n_fft"],
            hop_length=mc["hop_length"],
            win_length=mc["win_length"],
            num_freq=mc["num_freq"],
            feat_dim=mc["feat_dim"],
            num_bands=mc["num_bands"],
            num_low_bands=mc["num_low_bands"],
            num_high_bands=mc["num_high_bands"],
            num_layers=mc["num_layers"],
            use_visual_conditioning=mc["use_visual_conditioning"],
            num_landmarks=mc["num_landmarks"],
            coord_dim=mc["coord_dim"],
            visual_hidden_dim=mc["visual_hidden_dim"],
            upsample_factor=mc["upsample_factor"],
            use_motion_deltas=mc["use_motion_deltas"],
            target_audio_frames=mc["target_audio_frames"],
        )

    model = AVBSRNN(config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Sample selection helpers ──────────────────────────────────────────────────

def _snr_bin(snr_db: float) -> str:
    """Map an input SNR value to a named difficulty bin."""
    if snr_db < 0.0:
        return "low"
    if snr_db < 10.0:
        return "mid"
    return "high"


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _agg(vals: list[float | None]) -> dict[str, float | None]:
    """Mean, std, and median of a list, ignoring None and NaN."""
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return {"mean": None, "std": None, "median": None}
    return {
        "mean":   float(np.mean(v)),
        "std":    float(np.std(v)),
        "median": float(np.median(v)),
    }


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    manifest_path: Path,
    audio_dir: Path,
    landmark_dir: Path,
    noise_dir: Path,
    output_dir: Path,
    batch_size: int = 8,
    device: torch.device | None = None,
    num_audio_samples: int = 10,
    zero_landmarks: bool = False,
) -> dict[str, Any]:
    """Run full evaluation on the test split.

    Args:
        checkpoint_path:   Path to a ``.pt`` checkpoint file.
        cfg:               Parsed YAML config dict.
        manifest_path:     Path to ``manifest.json``.
        audio_dir:         GRID audio root.
        landmark_dir:      Landmark ``.npy`` root.
        noise_dir:         DEMAND noise root.
        output_dir:        Directory to write ``metrics.csv``, ``summary.json``,
                           and ``samples/``.
        batch_size:        Forward-pass batch size.
        device:            Torch device (defaults to CUDA if available).
        num_audio_samples: Total audio triples to save (≈3 low, 4 mid, 3 high SNR).
        zero_landmarks:    If ``True``, zero the landmark tensor before passing to
                           the model (ablation: disables visual signal without
                           retraining).

    Returns:
        Summary dict with ``overall``, ``by_mix_type``, ``by_snr_bin`` keys.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(checkpoint_path, cfg, device)
    use_visual: bool = cfg["model"]["use_visual_conditioning"]
    sr: int = cfg["model"]["sample_rate"]

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    dc = cfg["data"]
    dataset = GRIDAVDataset(
        manifest_path=manifest_path,
        audio_dir=audio_dir,
        landmark_dir=landmark_dir,
        noise_dir=noise_dir,
        split="test",
        snr_range=tuple(dc.get("snr_range", [-5.0, 20.0])),
        sir_range=tuple(dc.get("sir_range", [-5.0, 20.0])),
        mix_probabilities=tuple(dc.get("mix_probabilities", [0.5, 0.3, 0.2])),
        num_samples=dc.get("num_samples", 47648),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=grid_av_collate,
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    per_clip: list[dict[str, Any]] = []
    skipped = 0

    # Sample-saving state: spread num_audio_samples across 3 difficulty bins.
    n_low = max(1, num_audio_samples * 3 // 10)
    n_mid = max(1, num_audio_samples * 4 // 10)
    n_high = num_audio_samples - n_low - n_mid
    bin_targets  = {"low": n_low, "mid": n_mid, "high": n_high}
    bin_counts: dict[str, int] = {"low": 0, "mid": 0, "high": 0}
    saved_samples: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
            noisy     = batch["noisy"].to(device)
            clean     = batch["clean"].to(device)
            landmarks = batch["landmarks"].to(device)

            if zero_landmarks:
                landmarks = torch.zeros_like(landmarks)

            lm_arg = landmarks if use_visual else None
            enhanced = model(noisy, lm_arg)["waveform"]

            for i in range(noisy.shape[0]):
                clip_id  = batch["clip_ids"][i]
                speaker  = batch["speakers"][i]
                mix_type = int(batch["mix_types"][i].item())
                snr_val  = float(batch["snr_db"][i].item())
                sir_val  = float(batch["sir_db"][i].item())

                e = enhanced[i].cpu()
                c = clean[i].cpu()
                n = noisy[i].cpu()

                e_np = e.float().numpy()
                c_np = c.float().numpy()
                n_np = n.float().numpy()

                # SI-SNR (never fails)
                enh_sisnr  = si_snr(e, c)
                base_sisnr = si_snr(n, c)

                # PESQ-WB
                enh_pesq  = compute_pesq(c_np, e_np, sr)
                base_pesq = compute_pesq(c_np, n_np, sr)

                # STOI
                enh_stoi  = compute_stoi(c_np, e_np, sr)
                base_stoi = compute_stoi(c_np, n_np, sr)

                # Skip clips where both perceptual metrics failed
                if enh_pesq is None and enh_stoi is None:
                    skipped += 1
                    continue

                def _imp(a: float | None, b: float | None) -> float | None:
                    return (a - b) if (a is not None and b is not None) else None

                per_clip.append({
                    "clip_id":           clip_id,
                    "speaker":           speaker,
                    "mix_type":          mix_type,
                    "snr_db":            snr_val,
                    "sir_db":            sir_val,
                    "baseline_sisnr":    base_sisnr,
                    "enhanced_sisnr":    enh_sisnr,
                    "sisnr_improvement": enh_sisnr - base_sisnr,
                    "baseline_pesq":     base_pesq,
                    "enhanced_pesq":     enh_pesq,
                    "pesq_improvement":  _imp(enh_pesq,  base_pesq),
                    "baseline_stoi":     base_stoi,
                    "enhanced_stoi":     enh_stoi,
                    "stoi_improvement":  _imp(enh_stoi,  base_stoi),
                })

                # Decide whether to save this clip's audio
                b_key = _snr_bin(snr_val)
                if bin_counts[b_key] < bin_targets[b_key]:
                    bin_counts[b_key] += 1
                    saved_samples.append({
                        "idx":      len(saved_samples),
                        "clip_id":  clip_id,
                        "speaker":  speaker,
                        "snr_db":   snr_val,
                        "mix_type": mix_type,
                        "noisy":    n,
                        "clean":    c,
                        "enhanced": e,
                    })

    # ── Save audio samples ────────────────────────────────────────────────────
    mix_tag = {0: "noise", 1: "int_noise", 2: "int_only"}
    for s in saved_samples:
        stem = (
            f"{s['idx']:02d}_spk{s['speaker']}"
            f"_snr{s['snr_db']:+.1f}"
            f"_{mix_tag.get(s['mix_type'], str(s['mix_type']))}"
        )
        save_wav(sample_dir / f"{stem}_noisy.wav",    s["noisy"],    sr)
        save_wav(sample_dir / f"{stem}_enhanced.wav", s["enhanced"], sr)
        save_wav(sample_dir / f"{stem}_clean.wav",    s["clean"],    sr)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _col(key: str) -> list[float | None]:
        return [r[key] for r in per_clip]

    overall = {
        "baseline_sisnr":    _agg(_col("baseline_sisnr")),
        "enhanced_sisnr":    _agg(_col("enhanced_sisnr")),
        "sisnr_improvement": _agg(_col("sisnr_improvement")),
        "baseline_pesq":     _agg(_col("baseline_pesq")),
        "enhanced_pesq":     _agg(_col("enhanced_pesq")),
        "pesq_improvement":  _agg(_col("pesq_improvement")),
        "baseline_stoi":     _agg(_col("baseline_stoi")),
        "enhanced_stoi":     _agg(_col("enhanced_stoi")),
        "stoi_improvement":  _agg(_col("stoi_improvement")),
    }

    mix_names = {0: "target_noise", 1: "target_interferer_noise", 2: "target_interferer"}
    by_mix: dict[str, Any] = {}
    for mt in [0, 1, 2]:
        sub = [r for r in per_clip if r["mix_type"] == mt]
        by_mix[mix_names[mt]] = {
            "count":             len(sub),
            "sisnr_improvement": _agg([r["sisnr_improvement"] for r in sub]),
            "pesq_improvement":  _agg([r["pesq_improvement"]  for r in sub]),
            "stoi_improvement":  _agg([r["stoi_improvement"]  for r in sub]),
        }

    snr_bins = [(-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
    by_snr: dict[str, Any] = {}
    for lo, hi in snr_bins:
        key = f"[{lo:+d},{hi:+d})"
        sub = [r for r in per_clip if lo <= r["snr_db"] < hi]
        by_snr[key] = {
            "count":             len(sub),
            "sisnr_improvement": _agg([r["sisnr_improvement"] for r in sub]),
            "pesq_improvement":  _agg([r["pesq_improvement"]  for r in sub]),
            "stoi_improvement":  _agg([r["stoi_improvement"]  for r in sub]),
        }

    results: dict[str, Any] = {
        "num_clips":       len(per_clip),
        "num_skipped":     skipped,
        "zero_landmarks":  zero_landmarks,
        "overall":         overall,
        "by_mix_type":     by_mix,
        "by_snr_bin":      by_snr,
    }

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if per_clip:
        csv_path = output_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_clip[0].keys()))
            writer.writeheader()
            writer.writerows(per_clip)

    # ── Write JSON ────────────────────────────────────────────────────────────
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return results


# ── Human-readable summary ───────────────────────────────────────────────────

def print_summary(results: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Print a formatted evaluation summary to stdout."""
    ov = results["overall"]
    use_visual = cfg["model"]["use_visual_conditioning"]
    zeroed     = results.get("zero_landmarks", False)

    def _fmt(d: dict | None, fmt_str: str = ".2f") -> str:
        if d is None or d.get("mean") is None:
            return "N/A"
        return f"{d['mean']:{fmt_str}} ± {d['std']:{fmt_str}}"

    def _imp(d: dict | None, fmt_str: str = "+.2f", unit: str = "") -> str:
        if d is None or d.get("mean") is None:
            return "N/A"
        return f"{d['mean']:{fmt_str}}{unit}"

    vis_str = "ZEROED" if zeroed else ("ENABLED" if use_visual else "DISABLED")

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Test clips:           {results['num_clips']}")
    print(f"Skipped clips:        {results['num_skipped']}")
    print(f"Visual conditioning:  {vis_str}")
    print()
    print(f"{'OVERALL METRICS (mean ± std)':}")
    print(f"{'':26} {'Baseline':>20}  {'Enhanced':>20}  {'Improvement':>14}")
    print(f"{'SI-SNR (dB):':26} {_fmt(ov['baseline_sisnr']):>20}  {_fmt(ov['enhanced_sisnr']):>20}  {_imp(ov['sisnr_improvement'], unit=' dB'):>14}")
    print(f"{'PESQ-WB:':26} {_fmt(ov['baseline_pesq']):>20}  {_fmt(ov['enhanced_pesq']):>20}  {_imp(ov['pesq_improvement']):>14}")
    print(f"{'STOI:':26} {_fmt(ov['baseline_stoi'], '.3f'):>20}  {_fmt(ov['enhanced_stoi'], '.3f'):>20}  {_imp(ov['stoi_improvement'], '.3f'):>14}")

    print()
    print("BY MIX TYPE (mean SI-SNR improvement):")
    mix_labels = {
        "target_noise":            "Target + noise",
        "target_interferer_noise": "Target + interferer + noise",
        "target_interferer":       "Target + interferer only",
    }
    for key, label in mix_labels.items():
        d = results["by_mix_type"].get(key, {})
        n = d.get("count", 0)
        imp = _imp(d.get("sisnr_improvement"), unit=" dB")
        print(f"  {label:<38} {imp}  (n={n})")

    print()
    print("BY INPUT SNR (mean SI-SNR improvement):")
    for key, d in results["by_snr_bin"].items():
        n = d.get("count", 0)
        imp = _imp(d.get("sisnr_improvement"), unit=" dB")
        print(f"  {key:>12} dB:  {imp}  (n={n})")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AV-PSE checkpoint")
    parser.add_argument("--checkpoint",       type=Path, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--config",           type=Path, required=True,
                        help="Path to training YAML config")
    parser.add_argument("--output_dir",       type=Path, default=Path("eval_output"))
    parser.add_argument("--manifest_path",    type=Path, required=True)
    parser.add_argument("--audio_dir",        type=Path, required=True)
    parser.add_argument("--landmark_dir",     type=Path, required=True)
    parser.add_argument("--noise_dir",        type=Path, required=True)
    parser.add_argument("--num_audio_samples", type=int, default=10)
    parser.add_argument("--device",           type=str,  default="cuda")
    parser.add_argument("--batch_size",       type=int,  default=8)
    args = parser.parse_args()

    # Validate paths
    for p, name in [
        (args.checkpoint,    "checkpoint"),
        (args.config,        "config"),
        (args.manifest_path, "manifest_path"),
        (args.audio_dir,     "audio_dir"),
        (args.landmark_dir,  "landmark_dir"),
        (args.noise_dir,     "noise_dir"),
    ]:
        if not p.exists():
            print(f"ERROR: {name} path not found: {p}")
            sys.exit(1)

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU.")
        dev = "cpu"
    device = torch.device(dev)

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        manifest_path=args.manifest_path,
        audio_dir=args.audio_dir,
        landmark_dir=args.landmark_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        num_audio_samples=args.num_audio_samples,
    )
    print_summary(results, cfg)
    print(f"Results written to: {args.output_dir}/")
    print(f"  {args.output_dir}/metrics.csv")
    print(f"  {args.output_dir}/summary.json")
    print(f"  {args.output_dir}/samples/   ({results['num_clips']} clips evaluated)")


if __name__ == "__main__":
    main()
