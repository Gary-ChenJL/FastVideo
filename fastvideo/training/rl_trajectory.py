# SPDX-License-Identifier: Apache-2.0
"""
Trajectory collection and rollout utilities for RL training.

This module implements:
1. Flow-GRPO-Fast trajectory sampling (1-2 denoising steps)
2. Log probability extraction from transformer
3. Value prediction integration
"""

import torch
import torch.nn as nn
from typing import Any
from dataclasses import dataclass

from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import get_sigmas

logger = init_logger(__name__)


@dataclass
class Trajectory:
    """Container for a single rollout trajectory."""
    latents: torch.Tensor  # [B, C, T, H, W]
    log_probs: torch.Tensor  # [B]
    values: torch.Tensor  # [B]
    timesteps: torch.Tensor  # [B]
    prompts: list[str]


def collect_grpo_fast_trajectory(
    transformer: nn.Module,
    noise_scheduler: Any,
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    timesteps: torch.Tensor,
    num_denoising_steps: int = 2,
    use_sde: bool = True,
    device: torch.device = torch.device("cuda")
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect trajectory using Flow-GRPO-Fast approach.

    Instead of full denoising rollout (50 steps), sample at random
    intermediate timesteps and do 1-2 denoising steps.

    Args:
        transformer: Policy model (transformer)
        noise_scheduler: Diffusion scheduler
        latents: Clean latents [B, C, T, H, W]
        encoder_hidden_states: Text embeddings
        encoder_attention_mask: Text attention mask
        timesteps: Random timesteps for noise injection [B]
        num_denoising_steps: Number of denoising steps (1-2 for fast)
        use_sde: Use SDE sampling for stochasticity
        device: Device

    Returns:
        final_latents: Denoised latents [B, C, T, H, W]
        log_probs: Log probabilities [B]
    """
    batch_size = latents.shape[0]

    # Step 1: Add noise at the sampled timestep
    noise = torch.randn_like(latents)
    sigmas = get_sigmas(
        noise_scheduler,
        device,
        timesteps,
        n_dim=latents.ndim,
        dtype=latents.dtype
    )

    # Noisy input at timestep t
    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

    # Step 2: Perform num_denoising_steps of denoising
    current_latents = noisy_latents
    accumulated_log_probs = torch.zeros(batch_size, device=device)

    for step in range(num_denoising_steps):
        # Predict noise
        with torch.no_grad() if step < num_denoising_steps - 1 else torch.enable_grad():
            model_output = transformer(
                hidden_states=current_latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps.to(device, dtype=torch.bfloat16),
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False
            )

        # Extract predicted noise/velocity
        if isinstance(model_output, tuple):
            predicted = model_output[0]
        else:
            predicted = model_output

        # Compute log probability (negative MSE for Gaussian likelihood)
        # log p(x_t | x_0) ∝ -||predicted - target||^2
        target = noise - latents  # Flow matching target
        log_prob_step = -((predicted - target) ** 2).flatten(1).mean(dim=1)
        accumulated_log_probs += log_prob_step

        # Denoise one step (simplified Euler step)
        if step < num_denoising_steps - 1:
            # Update latents (move towards predicted clean latent)
            step_size = sigmas / num_denoising_steps
            if use_sde:
                # Add stochastic noise for SDE sampling
                sde_noise = torch.randn_like(current_latents) * (step_size * 0.5)
                current_latents = current_latents - step_size * predicted + sde_noise
            else:
                # Deterministic ODE step
                current_latents = current_latents - step_size * predicted

    final_latents = current_latents

    # Normalize log probs
    log_probs = accumulated_log_probs / num_denoising_steps

    logger.debug(
        "Collected trajectory: timesteps=%s, log_probs=%.3f",
        timesteps[0].item() if batch_size > 0 else None,
        log_probs.mean().item()
    )

    return final_latents, log_probs


def compute_value_predictions(
    value_model: nn.Module,
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute value predictions for given latents.

    Args:
        value_model: Value function model
        latents: Latents [B, C, T, H, W]
        encoder_hidden_states: Text embeddings
        timesteps: Timesteps
        encoder_attention_mask: Attention mask

    Returns:
        values: Value predictions [B]
    """
    with torch.no_grad():
        values = value_model(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps.to(latents.device, dtype=torch.bfloat16),
            encoder_attention_mask=encoder_attention_mask
        )

    logger.debug("Computed values: mean=%.3f", values.mean().item())

    return values


def extract_log_probs_from_transformer_output(
    model_output: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Extract log probabilities from transformer output.

    For diffusion/flow models, we approximate log p(x) using the
    negative squared error under Gaussian likelihood assumption.

    Args:
        model_output: Model predictions [B, C, T, H, W]
        target: Target values [B, C, T, H, W]
        reduction: "mean" or "none"

    Returns:
        log_probs: Log probabilities [B] or [B, ...]
    """
    # Compute MSE per sample
    mse = ((model_output - target) ** 2)

    if reduction == "mean":
        mse = mse.flatten(1).mean(dim=1)  # [B]
    elif reduction == "none":
        pass  # Keep full shape

    # Log probability under Gaussian: log p(x) ∝ -MSE
    # Normalize by a constant to keep values reasonable
    log_probs = -mse / 2.0

    return log_probs
