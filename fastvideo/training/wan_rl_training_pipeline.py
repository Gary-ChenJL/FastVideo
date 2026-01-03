# SPDX-License-Identifier: Apache-2.0
"""
Wan RL Training Pipeline entry point.

This module provides the entry point for RL training (GRPO, PPO, DPO) on Wan models.
It extends the RLPipeline with Wan-specific initialization.

Usage:
    torchrun --nproc_per_node=NUM_GPUS fastvideo/training/wan_rl_training_pipeline.py \
        --model_path <path> --rl_mode --rl_algorithm grpo ...
"""

import sys
from copy import deepcopy

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline
from fastvideo.training.rl import RLPipeline

logger = init_logger(__name__)


class WanRLTrainingPipeline(RLPipeline):
    """
    RL training pipeline for Wan models.

    Supports multiple RL algorithms:
    - GRPO (Group Relative Policy Optimization) with GRPO-Guard
    - PPO (Proximal Policy Optimization)
    - DPO (Direct Preference Optimization)

    Select algorithm via --rl_algorithm flag.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize Wan-specific scheduler."""
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """May be used in future refactors."""
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline for RL training."""
        logger.info("Initializing validation pipeline for RL training...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        validation_pipeline = WanPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

        self.validation_pipeline = validation_pipeline


def main(args) -> None:
    """Main entry point for RL training."""
    logger.info("Starting RL training pipeline...")

    pipeline = WanRLTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args

    logger.info("Algorithm: %s", args.rl_args.rl_algorithm)
    pipeline.train()
    logger.info("RL training pipeline done")


if __name__ == "__main__":
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
