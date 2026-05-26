"""Ablation runner: visual conditioning ON vs OFF.

Runs evaluate_checkpoint twice (or once per provided checkpoint) and prints
a side-by-side comparison table showing SI-SNR, PESQ-WB, and STOI improvements
with and without visual landmarks.

Usage — zero-landmarks ablation (single checkpoint):
    python scripts/run_ablation.py \
        --checkpoint checkpoints/best.pt \
        --config configs/default.yaml \
        --manifest_path /kaggle/input/grid-av/manifest.json \
        --audio_dir /kaggle/input/grid-audio \
        --landmark_dir /kaggle/input/grid-landmarks \
        --noise_dir /kaggle/input/demand-noise \
        --output_dir ablation_output

Usage — separate audio-only checkpoint ablation:
    python scripts/run_ablation.py \
        --checkpoint checkpoints/best.pt \
        --audio_only_checkpoint checkpoints/audio_only_best.pt \
        --config configs/default.yaml \
        ...
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow running from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.evaluate import evaluate_checkpoint, print_summary
import torch


# ── Formatting helpers ────────────────────────────────────────────────────────

def _mean(d: dict[str, Any] | None) -> float | None:
    """Extract mean from an _agg() dict, returning None when unavailable."""
    if d is None:
        return None
    return d.get("mean")


def _fmt(val: float | None, fmt: str = ".3f", suffix: str = "") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}{suffix}"


def _delta(a: float | None, b: float | None, fmt: str = "+.3f") -> str:
    """Format the difference (a - b), labelling the direction."""
    if a is None or b is None:
        return "N/A"
    diff = a - b
    return f"{diff:{fmt}}"


def _bar(val: float | None, ref: float | None, width: int = 10) -> str:
    """Simple ASCII progress bar: filled proportion = val/ref, clamped to [0,1]."""
    if val is None or ref is None or ref == 0:
        return "?" * width
    frac = min(max(val / ref, 0.0), 1.0)
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


# ── Report builder ────────────────────────────────────────────────────────────

def _build_report(
    res_vis: dict[str, Any],
    res_blind: dict[str, Any],
    label_vis: str,
    label_blind: str,
) -> str:
    """Render a multi-section comparison report as a string."""
    lines: list[str] = []

    def h(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("─" * len(title))

    def row(label: str, vis: str, blind: str, delta: str) -> None:
        lines.append(f"  {label:<28} {vis:>12}  {blind:>12}  {delta:>10}")

    def hdr() -> None:
        lines.append(f"  {'':28} {label_vis:>12}  {label_blind:>12}  {'Δ(vis-blind)':>10}")
        lines.append(f"  {'-'*28} {'-'*12}  {'-'*12}  {'-'*10}")

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("=" * 68)
    lines.append("  AV-PSE ABLATION REPORT: Visual Conditioning vs Blind")
    lines.append("=" * 68)
    lines.append(f"  Visual run :  {label_vis}  "
                 f"({'zero-landmarks' if res_vis['zero_landmarks'] else 'landmarks enabled'})")
    lines.append(f"  Blind run  :  {label_blind}  "
                 f"({'zero-landmarks' if res_blind['zero_landmarks'] else 'landmarks disabled'})")
    lines.append(f"  Visual clips: {res_vis['num_clips']}   "
                 f"Blind clips: {res_blind['num_clips']}")

    # ── Overall ───────────────────────────────────────────────────────────────
    h("OVERALL — Improvement over noisy baseline (mean ± std)")
    hdr()

    ov_v  = res_vis["overall"]
    ov_b  = res_blind["overall"]

    for metric_key, label, fmt, suffix in [
        ("sisnr_improvement", "SI-SNR imp (dB)",  ".2f", " dB"),
        ("pesq_improvement",  "PESQ-WB imp",       ".3f", ""),
        ("stoi_improvement",  "STOI imp",           ".4f", ""),
    ]:
        v_mean = _mean(ov_v.get(metric_key))
        b_mean = _mean(ov_b.get(metric_key))
        row(
            label,
            _fmt(v_mean, fmt, suffix),
            _fmt(b_mean, fmt, suffix),
            _delta(v_mean, b_mean, "+.3f"),
        )

    # ── Absolute (enhanced) ───────────────────────────────────────────────────
    h("OVERALL — Absolute metrics (enhanced output, mean)")
    hdr()
    for metric_key, label, fmt in [
        ("enhanced_sisnr", "SI-SNR (dB)",  ".2f"),
        ("enhanced_pesq",  "PESQ-WB",       ".3f"),
        ("enhanced_stoi",  "STOI",           ".4f"),
    ]:
        v_mean = _mean(ov_v.get(metric_key))
        b_mean = _mean(ov_b.get(metric_key))
        row(label, _fmt(v_mean, fmt), _fmt(b_mean, fmt), _delta(v_mean, b_mean, "+.3f"))

    # ── By mix type ───────────────────────────────────────────────────────────
    h("BY MIX TYPE — SI-SNR improvement (mean)")
    hdr()
    mix_labels = {
        "target_noise":            "Target + noise",
        "target_interferer_noise": "Target + int + noise",
        "target_interferer":       "Target + int only",
    }
    for mk, mlabel in mix_labels.items():
        dv = res_vis["by_mix_type"].get(mk, {})
        db = res_blind["by_mix_type"].get(mk, {})
        vm = _mean(dv.get("sisnr_improvement"))
        bm = _mean(db.get("sisnr_improvement"))
        row(
            f"{mlabel} (n={dv.get('count', 0)})",
            _fmt(vm, ".2f", " dB"),
            _fmt(bm, ".2f", " dB"),
            _delta(vm, bm, "+.3f"),
        )

    # ── By SNR bin ────────────────────────────────────────────────────────────
    h("BY INPUT SNR BIN — SI-SNR improvement (mean)")
    hdr()
    for key in sorted(res_vis["by_snr_bin"].keys()):
        dv = res_vis["by_snr_bin"].get(key, {})
        db = res_blind["by_snr_bin"].get(key, {})
        vm = _mean(dv.get("sisnr_improvement"))
        bm = _mean(db.get("sisnr_improvement"))
        row(
            f"{key} dB (n={dv.get('count', 0)})",
            _fmt(vm, ".2f", " dB"),
            _fmt(bm, ".2f", " dB"),
            _delta(vm, bm, "+.3f"),
        )

    lines.append("")
    lines.append("=" * 68)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: visual conditioning ON vs OFF"
    )
    parser.add_argument("--checkpoint",         type=Path, required=True,
                        help="Path to the main (visual-on) .pt checkpoint")
    parser.add_argument("--audio_only_checkpoint", type=Path, default=None,
                        help="Path to a separate audio-only checkpoint (optional). "
                             "If omitted, the same checkpoint is used with zeroed landmarks.")
    parser.add_argument("--config",             type=Path, required=True)
    parser.add_argument("--output_dir",         type=Path, default=Path("ablation_output"))
    parser.add_argument("--manifest_path",      type=Path, required=True)
    parser.add_argument("--audio_dir",          type=Path, required=True)
    parser.add_argument("--landmark_dir",       type=Path, required=True)
    parser.add_argument("--noise_dir",          type=Path, required=True)
    parser.add_argument("--num_audio_samples",  type=int,  default=10)
    parser.add_argument("--device",             type=str,  default="cuda")
    parser.add_argument("--batch_size",         type=int,  default=8)
    args = parser.parse_args()

    # Validate required paths
    for p, name in [
        (args.checkpoint,    "checkpoint"),
        (args.config,        "config"),
        (args.manifest_path, "manifest_path"),
        (args.audio_dir,     "audio_dir"),
        (args.landmark_dir,  "landmark_dir"),
        (args.noise_dir,     "noise_dir"),
    ]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}", file=sys.stderr)
            sys.exit(1)

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        dev = "cpu"
    device = torch.device(dev)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir   = args.output_dir / "visual"
    blind_dir = args.output_dir / "blind"

    common_kwargs: dict[str, Any] = dict(
        cfg=cfg,
        manifest_path=args.manifest_path,
        audio_dir=args.audio_dir,
        landmark_dir=args.landmark_dir,
        noise_dir=args.noise_dir,
        batch_size=args.batch_size,
        device=device,
        num_audio_samples=args.num_audio_samples,
    )

    # ── Visual run ────────────────────────────────────────────────────────────
    print("\n[1/2] Running VISUAL (visual conditioning enabled) …")
    res_vis = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=vis_dir,
        zero_landmarks=False,
        **common_kwargs,
    )
    print_summary(res_vis, cfg)

    # ── Blind run ─────────────────────────────────────────────────────────────
    print("\n[2/2] Running BLIND (visual conditioning ablated) …")
    if args.audio_only_checkpoint is not None:
        # Separate audio-only checkpoint: disable visual in config copy
        cfg_audio = copy.deepcopy(cfg)
        cfg_audio["model"]["use_visual_conditioning"] = False
        res_blind = evaluate_checkpoint(
            checkpoint_path=args.audio_only_checkpoint,
            cfg=cfg_audio,
            manifest_path=args.manifest_path,
            audio_dir=args.audio_dir,
            landmark_dir=args.landmark_dir,
            noise_dir=args.noise_dir,
            output_dir=blind_dir,
            batch_size=args.batch_size,
            device=device,
            num_audio_samples=args.num_audio_samples,
            zero_landmarks=False,
        )
        label_vis   = str(args.checkpoint.name)
        label_blind = str(args.audio_only_checkpoint.name)
    else:
        # Same checkpoint, zero the landmark tensor at inference time
        res_blind = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            output_dir=blind_dir,
            zero_landmarks=True,
            **common_kwargs,
        )
        label_vis   = str(args.checkpoint.name) + " [landmarks ON]"
        label_blind = str(args.checkpoint.name) + " [landmarks ZEROED]"

    print_summary(res_blind, cfg)

    # ── Comparison report ─────────────────────────────────────────────────────
    report = _build_report(res_vis, res_blind, label_vis, label_blind)
    print("\n" + report)

    report_path = args.output_dir / "ablation_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Combined JSON
    ablation_summary = {
        "visual":  res_vis,
        "blind":   res_blind,
        "labels":  {"visual": label_vis, "blind": label_blind},
    }
    with (args.output_dir / "ablation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(ablation_summary, fh, indent=2)

    print(f"\nOutputs written to: {args.output_dir}/")
    print(f"  ablation_report.txt")
    print(f"  ablation_summary.json")
    print(f"  visual/   (metrics.csv, summary.json, samples/)")
    print(f"  blind/    (metrics.csv, summary.json, samples/)")


if __name__ == "__main__":
    main()
