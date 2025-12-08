# SPDX-License-Identifier: Apache-2.0
"""
PPO (Proximal Policy Optimization) algorithm implementation.

This module implements standard PPO for video generation models.
PPO is a robust on-policy algorithm that uses a clipped surrogate objective.

References:
    - Schulman et al. "Proximal Policy Optimization Algorithms"
      https://arxiv.org/abs/1707.06347
"""

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger
from .base import BaseRLAlgorithm

logger = init_logger(__name__)


class PPOAlgorithm(BaseRLAlgorithm):
    """
    PPO (Proximal Policy Optimization) algorithm.

    PPO is a policy gradient method that uses a clipped surrogate objective
    to prevent large policy updates that could destabilize training.

    Key features:
    - Clipped surrogate objective
    - GAE for advantage estimation
    - Value function clipping
    - Multiple epochs of updates per data batch
    """

    @property
    def name(self) -> str:
        return "ppo"

    @property
    def requires_value_model(self) -> bool:
        # PPO always requires a value model for baseline
        return True

    @property
    def requires_reference_model(self) -> bool:
        # PPO doesn't require a separate reference model
        return False

    def _validate_config(self) -> None:
        """Validate PPO-specific configuration."""
        config = self.config

        if config.rl_policy_clip_range <= 0:
            raise ValueError(
                f"rl_policy_clip_range must be positive, got {config.rl_policy_clip_range}"
            )

        if config.rl_gamma < 0 or config.rl_gamma > 1:
            raise ValueError(
                f"rl_gamma must be in [0, 1], got {config.rl_gamma}"
            )

        if config.rl_lambda < 0 or config.rl_lambda > 1:
            raise ValueError(
                f"rl_lambda must be in [0, 1], got {config.rl_lambda}"
            )

        logger.info(
            "PPO config validated: clip_range=%.2f, gamma=%.3f, lambda=%.3f",
            config.rl_policy_clip_range,
            config.rl_gamma,
            config.rl_lambda
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        GAE reduces variance in advantage estimation while allowing some bias,
        controlled by the lambda parameter.

        Args:
            rewards: Rewards [B] or [B, T]
            values: Value predictions [B] or [B, T]
            next_values: Next value predictions [B] or [B, T]
            dones: Episode termination flags [B] or [B, T]

        Returns:
            advantages: GAE advantages
            returns: TD(lambda) returns
        """
        if dones is None:
            dones = torch.zeros_like(rewards)

        gamma = self.config.rl_gamma
        lambda_ = self.config.rl_lambda

        # Compute TD residuals: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        deltas = rewards + gamma * next_values * (1.0 - dones) - values

        # If single step (no time dimension), return directly
        if deltas.dim() == 1:
            advantages = deltas
            returns = advantages + values
            return self._normalize_advantages(advantages), returns

        # Multi-step: compute GAE recursively
        batch_size, num_steps = deltas.shape
        advantages = torch.zeros_like(deltas)
        gae = torch.zeros(batch_size, device=deltas.device)

        # Backward pass to compute GAE
        for t in reversed(range(num_steps)):
            gae = deltas[:, t] + gamma * lambda_ * (1.0 - dones[:, t]) * gae
            advantages[:, t] = gae

        # Returns are advantages + values
        returns = advantages + values

        return self._normalize_advantages(advantages), returns

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Normalize advantages to have zero mean and unit variance."""
        if self.config.rl_normalize_advantages:
            mean = advantages.mean()
            std = advantages.std()
            return (advantages - mean) / (std + epsilon)
        return advantages

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute PPO clipped surrogate objective.

        The clipped objective prevents large policy updates by clipping
        the probability ratio.

        Args:
            log_probs: Log probabilities from current policy [B]
            old_log_probs: Log probabilities from old policy [B]
            advantages: Advantages [B]

        Returns:
            loss: Policy loss (scalar)
            info: Dictionary with diagnostic information
        """
        clip_range = self.config.rl_policy_clip_range

        # Compute importance ratio: r_t = pi_new(a|s) / pi_old(a|s)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Compute diagnostics
        with torch.no_grad():
            # Clip fraction: how often ratios were clipped
            clip_fraction = (
                (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
            ).float().mean()

            # Approximate KL divergence
            # Using the approximation: KL ≈ (ratio - 1) - log(ratio)
            approx_kl = ((ratio - 1) - log_ratio).mean()

            # Importance ratio stats
            importance_ratio_mean = ratio.mean()
            importance_ratio_std = ratio.std()

        info = {
            "policy_loss": policy_loss.item(),
            "clip_fraction": clip_fraction.item(),
            "kl_divergence": approx_kl.item(),
            "importance_ratio_mean": importance_ratio_mean.item(),
            "importance_ratio_std": importance_ratio_std.item(),
        }

        return policy_loss, info

    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute value function loss with optional clipping.

        Args:
            values: Value predictions from current model [B]
            returns: Target returns [B]
            old_values: Value predictions from old model [B] (for clipping)

        Returns:
            loss: Value loss (scalar)
            info: Dictionary with diagnostic information
        """
        clip_range = self.config.rl_value_clip_range

        # Standard MSE loss
        value_loss_unclipped = F.mse_loss(values, returns, reduction="none")

        # Clipped value loss (optional, but commonly used)
        if old_values is not None and clip_range > 0:
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -clip_range,
                clip_range
            )
            value_loss_clipped = F.mse_loss(values_clipped, returns, reduction="none")
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = value_loss_unclipped.mean()

        # Compute diagnostics
        with torch.no_grad():
            explained_variance = 1.0 - (returns - values).var() / (returns.var() + 1e-8)

        info = {
            "value_loss": value_loss.item(),
            "explained_variance": explained_variance.item(),
            "value_mean": values.mean().item(),
            "value_std": values.std().item(),
        }

        return value_loss, info
