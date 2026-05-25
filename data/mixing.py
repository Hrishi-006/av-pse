"""Waveform mixing helpers for AV-PSE training simulation.

Usage:
    >>> import torch
    >>> from data.mixing import mix_audio, sample_mixing_config
    >>> target = torch.randn(2, 48000)
    >>> noise = torch.randn(2, 48000)
    >>> interferer = torch.randn(2, 48000)
    >>> cfg = sample_mixing_config(batch_size=2, device=target.device)
    >>> mixed = mix_audio(target, noise, interferer, cfg["snr_db"], cfg["sir_db"], cfg["use_interferer"])
"""

from __future__ import annotations

import torch


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute root-mean-square per sample in batch.

    Args:
        x: shape [B, num_samples]

    Returns:
        rms values of shape [B]
    """
    return torch.sqrt(torch.mean(x.pow(2), dim=-1) + eps)


def scale_to_snr(
    target: torch.Tensor,
    interferer: torch.Tensor,
    snr_db: torch.Tensor,
) -> torch.Tensor:
    """
    Scale `interferer` so that mixing with `target` achieves the SNR.

    Args:
        target: clean target [B, num_samples]
        interferer: noise or competing speech [B, num_samples]
        snr_db: desired SNR in dB [B]

    Returns:
        Scaled interferer [B, num_samples] such that the target SNR is met.
    """
    target_rms = rms(target)
    interferer_rms = rms(interferer)
    desired_interferer_rms = target_rms / torch.pow(10.0, snr_db / 20.0)
    scale = desired_interferer_rms / interferer_rms
    return interferer * scale.unsqueeze(-1)


def mix_audio(
    target: torch.Tensor,
    noise: torch.Tensor,
    interferer: torch.Tensor,
    snr_db: torch.Tensor,
    sir_db: torch.Tensor,
    use_interferer: torch.Tensor,
    peak_clip: float = 0.99,
) -> torch.Tensor:
    """
    Mix target + scaled_noise + (optional) scaled_interferer.

    Args:
        target: [B, num_samples] clean target waveform
        noise: [B, num_samples] noise waveform
        interferer: [B, num_samples] competing speech waveform
        snr_db: [B] SNR of target vs noise in dB
        sir_db: [B] SIR of target vs interferer in dB
        use_interferer: [B] bool mask for interferer contribution
        peak_clip: max absolute amplitude after mixing

    Returns:
        Mixed noisy waveform [B, num_samples], clipped to +/-peak_clip.
    """
    scaled_noise = scale_to_snr(target, noise, snr_db)
    scaled_interferer = scale_to_snr(target, interferer, sir_db)
    masked_interferer = scaled_interferer * use_interferer.float().unsqueeze(-1)
    mixed = target + scaled_noise + masked_interferer
    return torch.clamp(mixed, min=-peak_clip, max=peak_clip)


def sample_mixing_config(
    batch_size: int,
    snr_range: tuple[float, float] = (-5.0, 20.0),
    sir_range: tuple[float, float] = (-5.0, 20.0),
    mix_probabilities: tuple[float, float, float] = (0.5, 0.3, 0.2),
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Sample mixing parameters for a training batch.

    mix_probabilities: (p_target_noise_only, p_target_interferer_noise,
                        p_target_interferer_only) must sum to 1.0.

    Returns:
        Dict with snr_db, sir_db, use_interferer, and use_noise tensors.
    """
    probabilities = torch.tensor(mix_probabilities, dtype=torch.float32, device=device)
    mix_type = torch.multinomial(probabilities, num_samples=batch_size, replacement=True)

    use_noise = (mix_type == 0) | (mix_type == 1)
    use_interferer = (mix_type == 1) | (mix_type == 2)

    snr_low, snr_high = snr_range
    sir_low, sir_high = sir_range
    snr_db = torch.empty(batch_size, device=device).uniform_(snr_low, snr_high)
    sir_db = torch.empty(batch_size, device=device).uniform_(sir_low, sir_high)

    return {
        "snr_db": snr_db,
        "sir_db": sir_db,
        "use_interferer": use_interferer,
        "use_noise": use_noise,
    }
