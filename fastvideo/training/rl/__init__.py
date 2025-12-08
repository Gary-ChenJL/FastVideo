# SPDX-License-Identifier: Apache-2.0
"""
RL training module for FastVideo.

This module provides reinforcement learning training capabilities with
pluggable algorithms (GRPO, PPO, DPO).
"""

from .rl_pipeline import RLPipeline, create_rl_pipeline
from .algorithms import (
    BaseRLAlgorithm,
    AlgorithmOutput,
    GRPOAlgorithm,
    PPOAlgorithm,
    DPOAlgorithm,
    create_algorithm,
    register_algorithm,
    get_available_algorithms,
)

__all__ = [
    # Pipeline
    "RLPipeline",
    "create_rl_pipeline",
    # Algorithm base classes
    "BaseRLAlgorithm",
    "AlgorithmOutput",
    # Algorithm implementations
    "GRPOAlgorithm",
    "PPOAlgorithm",
    "DPOAlgorithm",
    # Factory functions
    "create_algorithm",
    "register_algorithm",
    "get_available_algorithms",
]
