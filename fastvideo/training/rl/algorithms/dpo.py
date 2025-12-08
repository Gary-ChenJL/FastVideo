# SPDX-License-Identifier: Apache-2.0
"""
DPO (Direct Preference Optimization) algorithm implementation.

This module implements DPO for video generation models.
DPO is an offline algorithm that learns directly from preference pairs
without requiring a reward model.

References:
    - Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
      https://arxiv.org/abs/2305.18290
    - Diffusion-DPO: https://arxiv.org/abs/2311.12908
"""

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from .base import BaseRLAlgorithm, AlgorithmOutput

logger = init_logger(__name__)


class DPOAlgorithm(BaseRLAlgorithm):
    """
    DPO (Direct Preference Optimization) algorithm.

    DPO is an offline preference learning algorithm that directly optimizes
    the policy from preference pairs without an explicit reward model.
    It's more stable than RLHF-based approaches and doesn't require
    online sampling.

    Key features:
    - No reward model needed (implicit reward)
    - Offline training from preference datasets
    - KL-constrained optimization via beta parameter
    - Reference model for regularization
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize DPO algorithm.

        Args:
            config: Algorithm configuration (should have dpo_beta)
        """
        super().__init__(config)
        # DPO-specific parameters (with defaults if not in config)
        self.beta = getattr(config, 'dpo_beta', 0.1)
        self.label_smoothing = getattr(config, 'dpo_label_smoothing', 0.0)
        self.reference_free = getattr(config, 'dpo_reference_free', False)

    @property
    def name(self) -> str:
        return "dpo"

    @property
    def requires_value_model(self) -> bool:
        # DPO doesn't use a value model
        return False

    @property
    def requires_reference_model(self) -> bool:
        # DPO requires a reference model for KL penalty
        # Unless using reference-free variant
        return not self.reference_free

    def _validate_config(self) -> None:
        """Validate DPO-specific configuration."""
        if self.beta <= 0:
            raise ValueError(f"dpo_beta must be positive, got {self.beta}")

        if self.label_smoothing < 0 or self.label_smoothing > 0.5:
            raise ValueError(
                f"dpo_label_smoothing must be in [0, 0.5], got {self.label_smoothing}"
            )

        logger.info(
            "DPO config validated: beta=%.2f, label_smoothing=%.3f, reference_free=%s",
            self.beta,
            self.label_smoothing,
            self.reference_free
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DPO doesn't use traditional advantages.

        This method returns dummy values since DPO operates on preference pairs
        rather than reward-based advantages.
        """
        # DPO doesn't use GAE - return zeros
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        return advantages, returns

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Standard policy loss interface (not used in DPO).

        DPO uses compute_dpo_loss instead for preference-based training.
        This method is provided for interface compatibility.
        """
        # Return dummy loss for interface compatibility
        dummy_loss = torch.tensor(0.0, device=log_probs.device)
        return dummy_loss, {"policy_loss": 0.0}

    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        DPO doesn't use a value function.

        Returns zero loss for interface compatibility.
        """
        dummy_loss = torch.tensor(0.0, device=values.device)
        return dummy_loss, {"value_loss": 0.0}

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor | None = None,
        reference_rejected_logps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute the DPO loss from preference pairs.

        The DPO objective is:
            L_DPO = -log(sigmoid(beta * (log(pi/pi_ref)_chosen - log(pi/pi_ref)_rejected)))

        Args:
            policy_chosen_logps: Log probs of chosen samples under policy [B]
            policy_rejected_logps: Log probs of rejected samples under policy [B]
            reference_chosen_logps: Log probs of chosen samples under reference [B]
            reference_rejected_logps: Log probs of rejected samples under reference [B]

        Returns:
            loss: DPO loss (scalar)
            info: Dictionary with diagnostic information
        """
        # Compute log probability ratios
        if self.reference_free:
            # Reference-free DPO: use only policy log probs
            chosen_rewards = self.beta * policy_chosen_logps
            rejected_rewards = self.beta * policy_rejected_logps
        else:
            # Standard DPO with reference model
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError(
                    "Reference model log probs required for standard DPO. "
                    "Set dpo_reference_free=True for reference-free variant."
                )
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        # DPO loss: -log(sigmoid(chosen_reward - rejected_reward))
        logits = chosen_rewards - rejected_rewards

        # Apply label smoothing if configured
        if self.label_smoothing > 0:
            # Smooth the labels: instead of 1 and 0, use (1-eps) and eps
            # This is equivalent to soft cross-entropy
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # Compute diagnostics
        with torch.no_grad():
            # Accuracy: how often we correctly rank chosen > rejected
            accuracy = (logits > 0).float().mean()

            # Reward margins
            reward_margin = (chosen_rewards - rejected_rewards).mean()

            # Individual rewards
            chosen_reward_mean = chosen_rewards.mean()
            rejected_reward_mean = rejected_rewards.mean()

        info = {
            "dpo_loss": loss.item(),
            "dpo_accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": chosen_reward_mean.item(),
            "rejected_reward": rejected_reward_mean.item(),
        }

        return loss, info

    def compute_loss(
        self,
        training_batch: TrainingBatch
    ) -> AlgorithmOutput:
        """
        Compute DPO loss from training batch.

        Note: This method expects the training batch to contain preference pair
        information (chosen vs rejected). The standard TrainingBatch may need
        to be extended for DPO training.

        Args:
            training_batch: Training batch (should contain preference pair data)

        Returns:
            AlgorithmOutput with DPO loss and metrics
        """
        # Check if batch has DPO-specific fields
        # These fields need to be added to TrainingBatch for DPO training
        policy_chosen_logps = getattr(training_batch, 'policy_chosen_logps', None)
        policy_rejected_logps = getattr(training_batch, 'policy_rejected_logps', None)
        reference_chosen_logps = getattr(training_batch, 'reference_chosen_logps', None)
        reference_rejected_logps = getattr(training_batch, 'reference_rejected_logps', None)

        if policy_chosen_logps is None or policy_rejected_logps is None:
            logger.warning(
                "DPO training requires preference pairs (chosen/rejected log probs). "
                "Returning zero loss."
            )
            dummy_loss = torch.tensor(0.0)
            return AlgorithmOutput(
                policy_loss=dummy_loss,
                value_loss=None,
                total_loss=dummy_loss,
                metrics={"dpo_loss": 0.0, "warning": "no_preference_data"}
            )

        # Compute DPO loss
        loss, info = self.compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
        )

        return AlgorithmOutput(
            policy_loss=loss,
            value_loss=None,
            total_loss=loss,
            metrics=info
        )

    def compute_implicit_reward(
        self,
        policy_logps: torch.Tensor,
        reference_logps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute implicit reward from DPO formulation.

        The implicit reward in DPO is:
            r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))

        Args:
            policy_logps: Log probs under policy [B]
            reference_logps: Log probs under reference [B]

        Returns:
            implicit_rewards: Implicit reward scores [B]
        """
        return self.beta * (policy_logps - reference_logps)
