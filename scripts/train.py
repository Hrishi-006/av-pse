"""Training entry point for AV-PSE.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml \
        --override training.batch_size=8 model.num_layers=4
    python scripts/train.py --config configs/default.yaml \
        --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import logging
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from data.grid_dataset import GRIDAVDataset, grid_av_collate
from losses.multi_res_loss import MultiResolutionLoss
from models.av_bsrnn import AVBSRNN, AVBSRNNConfig


# ── Config helpers ──────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _set_nested(cfg: dict[str, Any], dotkey: str, value: str) -> None:
    """Write a dotted key (e.g. 'training.batch_size') into a nested dict.

    The string value is coerced: 'true'/'false' → bool, integers and floats
    are parsed automatically, otherwise kept as a string.
    """
    parts = dotkey.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    leaf = parts[-1]

    # Coerce type
    low = value.strip().lower()
    if low in ("true", "false"):
        node[leaf] = low == "true"
    else:
        for cast in (int, float):
            try:
                node[leaf] = cast(value)
                return
            except ValueError:
                pass
        node[leaf] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    """Apply a list of 'key=value' override strings to cfg in-place."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override!r}")
        key, val = override.split("=", 1)
        _set_nested(cfg, key.strip(), val.strip())


# ── Logging ─────────────────────────────────────────────────────────────────

def build_logger(checkpoint_dir: Path) -> logging.Logger:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "train.log"

    logger = logging.getLogger("av_pse")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Seeding ─────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model / data / optimizer builders ───────────────────────────────────────

def build_model(cfg: dict[str, Any]) -> AVBSRNN:
    mc = cfg["model"]
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
    return AVBSRNN(config)


def build_dataloaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    dc = cfg["data"]
    tc = cfg["training"]

    common = dict(
        manifest_path=dc["manifest_path"],
        audio_dir=dc["audio_dir"],
        landmark_dir=dc["landmark_dir"],
        noise_dir=dc["noise_dir"],
        snr_range=tuple(dc["snr_range"]),
        sir_range=tuple(dc["sir_range"]),
        mix_probabilities=tuple(dc["mix_probabilities"]),
        num_samples=dc["num_samples"],
    )
    train_ds = GRIDAVDataset(**common, split="train")
    val_ds = GRIDAVDataset(**common, split="val", seed=42)

    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc["num_workers"],
        collate_fn=grid_av_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=tc["num_workers"],
        collate_fn=grid_av_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_loss(cfg: dict[str, Any]) -> MultiResolutionLoss:
    lc = cfg["loss"]
    return MultiResolutionLoss(
        sample_rate=cfg["model"]["sample_rate"],
        window_ms=lc["window_ms"],
        power=lc["power"],
        hop_divisor=lc["hop_divisor"],
    )


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: dict[str, Any],
) -> tuple[torch.optim.Adam, torch.optim.lr_scheduler.CosineAnnealingLR]:
    tc = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tc["max_steps"],
        eta_min=tc["lr_min"],
    )
    return optimizer, scheduler


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "step": step,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> tuple[int, float]:
    """Load checkpoint into model/optimizer/scheduler. Returns (step, val_loss).

    Scheduler state is loaded only when the saved state-dict keys exactly match
    the current scheduler type (e.g. both CosineAnnealingLR).  When the keys
    differ (e.g. old StepLR checkpoint, new CosineAnnealingLR run) the saved
    state is skipped and the new scheduler is fast-forwarded by calling
    ``scheduler.step()`` *step* times so the LR is at the correct cosine
    position.  PyTorch's ``load_state_dict`` is a plain ``dict.update`` and
    will NOT raise on a type mismatch, so we must detect it via key comparison.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    step: int = ckpt["step"]

    saved_sched = ckpt.get("scheduler_state")
    scheduler_restored = False

    if saved_sched is not None:
        current_keys = set(scheduler.state_dict().keys())
        saved_keys = set(saved_sched.keys())

        if current_keys == saved_keys:
            # Same scheduler type — safe to restore normally.
            try:
                scheduler.load_state_dict(saved_sched)
                scheduler_restored = True
            except (ValueError, KeyError, RuntimeError) as exc:
                if logger:
                    logger.warning("Scheduler state load failed (%s); will fast-forward.", exc)
        else:
            # Type mismatch (e.g. StepLR → CosineAnnealingLR).
            extra_saved = sorted(saved_keys - current_keys)
            extra_current = sorted(current_keys - saved_keys)
            if logger:
                logger.warning(
                    "Scheduler type mismatch — checkpoint keys %s not in current "
                    "scheduler (which has %s). Skipping saved state; "
                    "fast-forwarding new scheduler %d steps instead.",
                    extra_saved, extra_current, step,
                )

    if not scheduler_restored:
        # Advance the fresh scheduler to the correct position without triggering
        # the "scheduler.step() before optimizer.step()" warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(step):
                scheduler.step()

    return step, ckpt.get("val_loss", float("inf"))


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    loss_fn: MultiResolutionLoss,
    device: torch.device,
    use_visual: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        landmarks = batch["landmarks"].to(device) if use_visual else None
        out = model(noisy, landmarks)
        loss = loss_fn(out["waveform"], clean)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ── Training loop ────────────────────────────────────────────────────────────

def train(
    cfg: dict[str, Any],
    resume_path: Path | None = None,
) -> None:
    tc = cfg["training"]
    seed_everything(tc["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(tc["checkpoint_dir"])
    logger = build_logger(checkpoint_dir)
    logger.info("Device: %s", device)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    use_visual: bool = cfg["model"]["use_visual_conditioning"]
    step = 0
    best_val_loss = float("inf")

    if resume_path is not None:
        step, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, device, logger=logger
        )
        logger.info("Resumed from %s at step %d (best val=%.4f)", resume_path, step, best_val_loss)

    param_count = model.count_parameters()
    logger.info("Parameters: %s", {k: f"{v:,}" for k, v in param_count.items()})

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(
        "Train samples=%d  Val samples=%d",
        len(train_loader.dataset),  # type: ignore[arg-type]
        len(val_loader.dataset),    # type: ignore[arg-type]
    )

    model.train()
    max_steps: int = tc["max_steps"]
    grad_clip: float = tc["grad_clip_norm"]
    log_every: int = tc["log_every_steps"]
    val_every: int = tc["val_every_steps"]

    running_loss = 0.0
    t0 = time.monotonic()

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            landmarks = batch["landmarks"].to(device) if use_visual else None

            optimizer.zero_grad()
            out = model(noisy, landmarks)
            loss = loss_fn(out["waveform"], clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            running_loss += loss.item()

            if step % log_every == 0:
                avg_loss = running_loss / log_every
                elapsed = time.monotonic() - t0
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "step=%d  loss=%.4f  lr=%.2e  elapsed=%.0fs",
                    step, avg_loss, lr, elapsed,
                )
                running_loss = 0.0

            if step % val_every == 0:
                val_loss = validate(model, val_loader, loss_fn, device, use_visual)
                logger.info("  [val] step=%d  val_loss=%.4f", step, val_loss)

                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model, optimizer, scheduler, step, val_loss,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model, optimizer, scheduler, step, val_loss,
                    )
                    logger.info("  [val] new best: %.4f → saved best.pt", best_val_loss)

    logger.info("Training complete at step %d. Best val loss: %.4f", step, best_val_loss)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train AV-PSE model")
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. training.batch_size=8",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    apply_overrides(cfg, args.override)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
