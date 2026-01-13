# SPDX-License-Identifier: Apache-2.0
"""
RL training pipeline for FastVideo.

This module implements GRPO (Group Relative Policy Optimization) training for video generation models.
It extends the base TrainingPipeline with RL-specific functionality including trajectory collection,
reward computation, advantage estimation, and GRPO loss computation.

Reference:
    Flow-GRPO: https://github.com/yifan123/flow_grpo
"""

from tkinter import SEL_FIRST
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Union

import math

from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.rl.rewards import (
    create_reward_models,
    MultiRewardAggregator,
    ValueModel
)
from .rl_utils import (
    sample_random_timesteps,
    compute_reward_statistics,
)
from fastvideo.training.rl.wan_grpo_utils import wan_pipeline_with_logprob, sde_step_with_logprob
from fastvideo.training.rl.stat_tracking import PerPromptStatTracker
from fastvideo.training.training_utils import (
    get_scheduler,
    count_trainable
)
from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline
from fastvideo.distributed import get_local_torch_device
from fastvideo.dataset.rl_prompt_dataset import build_rl_prompt_dataloader
from copy import deepcopy
from collections.abc import Iterator
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.utils import set_random_seed
import contextlib
import tqdm
from collections import defaultdict
from concurrent import futures
import imageio
import numpy as np
import tempfile
import time
import os 

logger = init_logger(__name__)


class RLPipeline(TrainingPipeline):
    """
    RL training pipeline for flow matching models.

    This pipeline implements GRPO (Group Relative Policy Optimization) for video generation models.
    It handles:
    - Trajectory collection with log probability computation
    - Reward computation using reward models
    - Advantage computation with per-prompt stat tracking
    - GRPO policy loss with clipping and KL regularization
    """

    def __init__(
        self,
        model_path: str,
        fastvideo_args: TrainingArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, nn.Module] | None = None
    ) -> None:
        """Initialize RL pipeline."""
        if not fastvideo_args.rl_args.rl_mode:
            logger.warning(
                "rl_mode is False, but RLPipeline is being initialized. "
                "Setting rl_mode=True."
            )
            fastvideo_args.rl_args.rl_mode = True

        super().__init__(model_path, fastvideo_args, required_config_modules, loaded_modules)

        # RL-specific components (will be initialized in initialize_training_pipeline)
        self.reward_models: MultiRewardAggregator | None = None
        self.value_model: ValueModel | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        self.value_scheduler: Any | None = None
        
        # Sampling pipeline for generating videos with log probabilities
        # Will be initialized in initialize_training_pipeline
        self.sampling_pipeline: WanPipeline | None = None
        
        # Per-prompt stat tracker for advantage normalization
        # Will be initialized in initialize_training_pipeline
        self.stat_tracker: PerPromptStatTracker | None = None

        logger.info(
            "Initialized RLPipeline with algorithm: %s",
            fastvideo_args.rl_args.rl_algorithm
        )

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        pass

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the RL training pipeline with algorithm, reward and value models."""
        # Call parent initialization for basic setup (optimizer, scheduler, etc.)
        # But we'll override the dataloader initialization
        super().initialize_training_pipeline(training_args)
        
        # Override dataloader with RL prompt dataloader
        # Get RL dataset configuration from training_args
        rl_dataset_path = training_args.rl_dataset_path if training_args.rl_dataset_path else training_args.data_path
        rl_dataset_type = training_args.rl_dataset_type  # "text" or "geneval"
        rl_num_image_per_prompt = training_args.rl_num_image_per_prompt  # k parameter
        num_replicas = 1  # Single GPU training
        rank = 0  # Single GPU training
        
        logger.info("Initializing RL prompt dataloader...")
        logger.info("  dataset_path: %s", rl_dataset_path)
        logger.info("  dataset_type: %s", rl_dataset_type)
        logger.info("  num_image_per_prompt (k): %s", rl_num_image_per_prompt)
        
        # Build RL prompt dataloader
        train_dataloader, test_dataloader, train_dataset, test_dataset, train_sampler = build_rl_prompt_dataloader(
            dataset_path=rl_dataset_path,
            dataset_type=rl_dataset_type,
            split='train',
            train_batch_size=training_args.train_batch_size,
            test_batch_size=8,  # Hardcoded for now
            k=rl_num_image_per_prompt,
            seed=training_args.seed if training_args.seed is not None else 42,
            train_num_workers=training_args.dataloader_num_workers,
            test_num_workers=0,
            num_replicas=num_replicas,
            rank=rank
        )
               
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.train_loader_iter = iter(self.train_dataloader)
        self.current_epoch = 0
        
        logger.info("train_dataloader length: %s", len(self.train_dataloader))

        self.num_update_steps_per_epoch = math.ceil(
                len(self.train_dataloader) /
                training_args.gradient_accumulation_steps * training_args.sp_size /
                training_args.train_sp_batch_size)
        self.num_train_epochs = math.ceil(training_args.max_train_steps /
                                            self.num_update_steps_per_epoch)

        logger.info("Initializing RL-specific components...")

        # Initialize reward models (GRPO always uses reward models)
        self.reward_models = create_reward_models(
            reward_models=training_args.rl_args.reward_models,
            device=str(self.device)
        )
        logger.info("Loaded reward models: %s", self.reward_models)

        # Initialize sampling pipeline for trajectory collection
        self._initialize_sampling_pipeline(training_args)
        
        # Initialize per-prompt stat tracker for advantage normalization
        global_std = getattr(training_args.rl_args, 'global_std', False)
        self.stat_tracker = PerPromptStatTracker(global_std=global_std)
        logger.info("Initialized PerPromptStatTracker with global_std=%s", global_std)

        logger.info("RL pipeline initialization complete")

    def _initialize_value_model(self, training_args: TrainingArgs) -> None:
        """Initialize the value model and its optimizer."""
        if training_args.rl_args.value_model_share_backbone:
            # Share transformer backbone with policy
            logger.info("Value model will share backbone with policy transformer")
            self.value_model = ValueModel(
                self.transformer,
                share_backbone=True
            )
        else:
            # Separate value model (clone transformer architecture)
            logger.info("Creating separate value model")
            # TODO: Implement separate value model initialization
            # For now, use shared backbone
            self.value_model = ValueModel(
                self.transformer,
                share_backbone=True
            )

        # Create optimizer and scheduler for value model
        if not training_args.rl_args.value_model_share_backbone:
            value_params = list(self.value_model.parameters())
            self.value_optimizer = torch.optim.AdamW(
                value_params,
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=training_args.weight_decay,
                eps=1e-8,
            )

            self.value_scheduler = get_scheduler(
                training_args.lr_scheduler,
                optimizer=self.value_optimizer,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
                last_epoch=self.init_steps - 1,
            )

            logger.info("Created separate optimizer and scheduler for value model")
            logger.info("Value model trainable parameters: %s B",
                       round(count_trainable(self.value_model) / 1e9, 3))

    def _initialize_sampling_pipeline(self, training_args: TrainingArgs) -> None:
        """
        Initialize a WanPipeline for sampling videos with log probabilities.
        
        This pipeline is used for on-policy trajectory collection during RL training.
        It shares the transformer with the training pipeline but runs in eval mode.
        """
        logger.info("Initializing sampling pipeline for trajectory collection...")
        
        # Create a copy of training args for inference mode
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        
        # Create WanPipeline for sampling, sharing the transformer
        self.sampling_pipeline = WanPipeline.from_pretrained(
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
            dit_cpu_offload=True
        )
        
        # Ensure transformer is in eval mode for sampling
        self.sampling_pipeline.get_module("transformer").eval()
        
        logger.info("Sampling pipeline initialized")

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Get next batch of prompts from RL prompt dataloader.
        
        The RL prompt dataloader returns (prompts, metadatas) tuples from the collate function.
        This method extracts prompts and stores them in training_batch for use in collect_trajectories.
        
        Args:
            training_batch: Current training batch
        
        Returns:
            Updated training_batch with prompts in input_kwargs
        """
        with self.tracker.timed("timing/get_next_batch"):
            try:
                batch = next(self.train_loader_iter)
            except StopIteration:
                # Reset iterator for next epoch
                self.current_epoch += 1
                logger.info("Starting epoch %s", self.current_epoch)
                self.train_loader_iter = iter(self.train_dataloader)
                batch = next(self.train_loader_iter)
            
            # RL prompt dataloader returns (prompts, metadatas) tuple
            prompts, metadatas = batch
            
            # Store prompts and metadatas in training_batch for use in collect_trajectories
            if training_batch.input_kwargs is None:
                training_batch.input_kwargs = {}
            training_batch.input_kwargs["prompts"] = prompts
            training_batch.input_kwargs["metadata"] = metadatas
            
            # Also store in infos for compatibility (convert metadatas to info_list format)
            if metadatas:
                training_batch.infos = [
                    {"prompt": prompt, "metadata": metadata}
                    for prompt, metadata in zip(prompts, metadatas)
                ]
            else:
                training_batch.infos = [
                    {"prompt": prompt, "caption": prompt}
                    for prompt in prompts
                ]
        
        return training_batch

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline for RL (similar to base implementation)."""
        # For now, reuse the base implementation
        # In the future, we can add RL-specific validation (e.g., compare old vs new policy)
        logger.info("RL validation pipeline will be implemented based on task requirements")
        # Set validation_pipeline to None for now
        self.validation_pipeline = None
    def _get_t5_prompt_embeds(
        self,
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]] = None,
    max_sequence_length: int = 226,
    num_videos_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 226,
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = text_encoder[0].device
        dtype = text_encoder[0].dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds = self._get_t5_prompt_embeds(
            text_encoder=text_encoder[0],
            tokenizer=tokenizer[0],
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
            dtype=dtype,
        )

        return prompt_embeds
    def compute_text_embeddings(self, prompt, text_encoders, tokenizers, max_sequence_length, device):
        with torch.no_grad():
            prompt_embeds = self.encode_prompt(
                text_encoders, tokenizers, prompt, max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(device)
            # pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        return prompt_embeds
      
    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        """
        Generate validation videos, calculate rewards, and log to tracker.
        (Single GPU Version)
        """
        # --- 1. Setup & Config ---
        training_args.inference_mode = True
        training_args.dit_cpu_offload = False
        
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

       

        # --- 2. Data Preparation ---
        # Prepare Negative Embeddings
        neg_prompt_embed = self.compute_text_embeddings(
            [""], 
            self.get_module("text_encoder"),
            self.get_module("tokenizer"),
            max_sequence_length=512, 
            device=self.device
        )
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.training_args.rl_args.sample_test_batch_size, 1, 1)

        
        # Setup Dataset
        validation_dataset = ValidationDataset(training_args.validation_dataset_file)
        validation_dataloader = DataLoader(validation_dataset,
                                        batch_size=training_args.eval_batch_size, # Ensure this arg exists
                                        num_workers=0)

        self.transformer.eval()

        # --- 3. Inference Loop ---
        # Container for results
        step_results = {
            "videos": [],
            "captions": [],
            "rewards": defaultdict(list)
        }

        for batch_idx, validation_batch in enumerate(validation_dataloader):
            # Extract prompts
            prompts, prompt_metadata = validation_batch

            # Compute Embeddings
            prompt_embeds = self.compute_text_embeddings(
                prompts,
                self.get_module("text_encoder"),
                self.get_module("tokenizer"),
                max_sequence_length=512,
                device=self.device
            )
            if len(prompt_embeds)<len(sample_neg_prompt_embeds):
                sample_neg_prompt_embeds  = sample_neg_prompt_embeds [:len(prompt_embeds)]

            # Run Inference
            with torch.no_grad():
                videos, latents, log_probs, _ = wan_pipeline_with_logprob(
                    self.validation_pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.training_args.rl_args.eval_num_steps,
                    guidance_scale=self.training_args.rl_args.eval_guidance_scale,
                    output_type="pt",
                    return_dict=False,
                    num_frames=self.training_args.frames,
                    height=self.training_args.height,
                    width=self.training_args.width,
                    determistic=True,
                )

            # Check Reward Calculation
            
            # future_rewards = self.executor.submit(self.reward_fn, videos, prompts, prompt_metadata, only_strict=False)
            # rewards_dict, _ = future_rewards.result()
            rewards_dict = self.reward_models.compute_rewards(videos, prompts, return_individual=True)

            # --- 5. Process Outputs ---
            # Store Rewards
            for k, v in rewards_dict.items():
                step_results["rewards"][k].append(v)

            # Store Videos (Convert to numpy uint8)
            # Using the correct rearrange logic we discussed: b c t h w -> b t h w c
            video_permuted = videos.permute(0, 2, 3, 4, 1)
            
            for v_idx, video_tensor in enumerate(video_permuted):
                video_np = (video_tensor.cpu().numpy() * 255).astype(np.uint8)
                step_results["videos"].append(video_np)
                step_results["captions"].append(prompts[v_idx])

        # --- 6. Consolidate Results (No Distributed Gathering) ---
        # Flatten the rewards lists into single arrays
        all_rewards = {k: np.concatenate(v) for k, v in step_results["rewards"].items()}
        all_videos = step_results["videos"]
        all_captions = step_results["captions"]

        # --- 7. Logging ---
        # Save Videos
        video_filenames = []
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        num_samples_to_save = min(15, len(all_videos))
        
        for i in range(num_samples_to_save):
            filename = os.path.join(
                training_args.output_dir,
                f"val_step_{global_step}_idx_{i}.mp4"
            )
            imageio.mimsave(filename, all_videos[i], fps=8, codec="libx264")
            video_filenames.append(filename)

        # Log to Tracker
        if hasattr(self.tracker, "log_artifacts"):
            wandb_videos = []
            for i in range(num_samples_to_save):
                # Create caption with reward stats
                reward_str = " | ".join(f"{k}: {all_rewards[k][i]:.2f}" for k in all_rewards)
                full_caption = f"{all_captions[i][:100]} | {reward_str}"
                
                wandb_videos.append(
                    wandb.Video(video_filenames[i], caption=full_caption, fps=8)
                )
            
            # Calculate Mean Rewards
            mean_rewards = {f"eval_reward_{k}": np.mean(v) for k, v in all_rewards.items()}
            
            logs = {
                "eval_images": wandb_videos,
                **mean_rewards
            }
            self.tracker.log_artifacts(logs, global_step)


        training_args.inference_mode = False
        self.transformer.train()    

    def collect_trajectories(self, epoch, text_encoders, tokenizers, sample_neg_prompt_embeds) -> TrainingBatch:
        """
        Collect on-policy trajectories by generating videos with log probabilities.
        
        This method implements the GRPO sampling phase:
        1. Gets prompts from the training batch
        2. Uses wan_pipeline_with_logprob to generate videos with log probabilities
        3. Stores latents, log_probs, timesteps, and KL divergences in TrainingBatch
        
        Ported from FlowGRPO's sampling loop to work with FastVideo's TrainingBatch structure.
        
        Args:
            training_batch: Current training batch (should contain prompts in input_kwargs or infos)
        
        Returns:
            Updated training_batch with sampling results:
            - latents: [B, num_steps+1, C, T, H, W] - latents at each denoising step
            - log_probs: [B, num_steps] - log probabilities at each step
            - timesteps: [B, num_steps] - timesteps used
            - old_log_probs: [B, num_steps] - copy of log_probs for importance ratio
            - kl: [B, num_steps] - KL divergences (if kl_reward > 0)
            - prompt_ids: [B, seq_len] - prompt token IDs for stat tracking
            - prompt_embeds: [B, seq_len, hidden_dim] - prompt embeddings used
            - negative_prompt_embeds: [B, seq_len, hidden_dim] - negative embeddings for CFG
        """
        self.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(self.training_args.rl_args.sample_num_batches_per_epoch),
            # desc=f"Epoch {epoch}: sampling",
            # disable=not accelerator.is_local_main_process,
            position=0,
        ):
            self.train_sampler.set_epoch(epoch * self.training_args.rl_args.sample_num_batches_per_epoch + i)
            prompts, prompt_metadata = next(self.train_loader_iter)

            prompt_embeds = self.compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=512,
                device=self.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=512, 
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            # if i==0 and epoch % self.training_args.eval_freq == 0 and epoch>0:
            #     self.eval(pipeline, self.test_dataloader, text_encoders, tokenizers, eval_reward_fn, executor)
            # if i==0 and epoch % self.training_args.save_freq == 0 and epoch>0 and accelerator.is_main_process:
            #     save_ckpt(self.training_args.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
            # 这里是故意的，因为前两个epoch收集的group size会有bug,经过两个epoch后，group_size稳定成指定的
            if epoch < 2:
                continue
            # sample
            for j in tqdm(
                range(self.training_args.rl_args.sample_time_per_prompt),
                # desc=f"Epoch {epoch}: sampling | multi sample per prompt",
                # disable=not accelerator.is_local_main_process,
                position=1,
            ):
                # with autocast():
                with torch.no_grad():
                    videos, latents, log_probs, kls = wan_pipeline_with_logprob(
                        self,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=self.training_args.rl_args.num_steps,
                        guidance_scale=self.training_args.rl_args.guidance_scale,
                        output_type="pt",
                        return_dict=False,
                        num_frames=self.training_args.num_frames,
                        height=self.training_args.num_height,
                        width=self.training_args.num_width, 
                        kl_reward=getattr(self.training_args.rl_args, 'kl_reward', 0.0),
                )

                latents = torch.stack(
                    latents, dim=1
                )  # (batch_size, num_steps + 1, 16, 96, 96)
                log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
                kls = torch.stack(kls, dim=1) 
                kl = kls.detach()

                timesteps = self.get_module("scheduler").timesteps.repeat(
                    self.training_args.rl_args.sample_train_batch_size, 1
                )  # (batch_size, num_steps)

                # compute rewards asynchronously
                # rewards = executor.submit(reward_fn, videos, prompts, prompt_metadata, only_strict=True)
                # images b, 3, 512, 512
                # yield to to make sure reward computation starts
                # time.sleep(0)
                rewards = self.reward_models.compute_rewards(videos, prompts, return_individual=True)
                
                samples.append(
                    {
                        "prompt_ids": prompt_ids,   # b, 77
                        "prompt_embeds": prompt_embeds,    # b, 205, 4096
                        "negative_prompt_embeds": sample_neg_prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[
                            :, :-1
                        ],  # each entry is the latent before timestep t.   b, 11, 16, 64, 64
                        "next_latents": latents[
                            :, 1:
                        ],  # each entry is the latent after timestep t
                        "log_probs": log_probs,   # b, t + 1
                        "kl": kl,
                        "rewards": rewards,
                    }
                )
        for sample in tqdm(
        samples,
        desc="Waiting for rewards",
        # disable=not accelerator.is_local_main_process,
        position=0,
    ):
            # rewards, reward_metadata = sample["rewards"].result()
            rewards = sample["rewards"]
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=self.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }
        # gather rewards across processes
        # gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in samples["rewards"].items()}
        # log rewards and images
        # accelerator.log(
        #     {
        #         "epoch": epoch,
        #         **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
        #         "kl": samples["kl"].mean().cpu().numpy(),
        #         "kl_abs": samples["kl"].abs().mean().cpu().numpy()
        #     },
        #     step=global_step,
        # )
        advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)
        advantages = torch.as_tensor(advantages)
        # samples["advantages"] = (
        #     advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
        #     .to(accelerator.device)
        # )
        samples["advantages"] = advantages.to(self.device)
        # del samples["rewards"]
        # del samples["prompt_ids"]



    def compute_rewards(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute rewards by decoding latents to videos and calling reward models.
        
        This method implements Step 5 of GRPO training:
        1. Extracts final latents from the denoising trajectory
        2. Decodes latents to videos using VAE
        3. Calls reward models with decoded videos
        4. Applies KL reward penalty if configured
        5. Stores reward scores and statistics in TrainingBatch
        
        Ported from FlowGRPO's reward computation to work with FastVideo's TrainingBatch.
        
        Args:
            training_batch: Training batch with latents [B, num_steps+1, C, T, H, W]
        
        Returns:
            Updated training_batch with:
            - reward_scores: Computed rewards [B]
            - reward_mean: Mean reward
            - reward_std: Std of rewards
        """
        if self.reward_models is None:
            raise RuntimeError("Reward models not initialized. Call initialize_training_pipeline first.")
        
        logger.debug("Computing rewards from reward models")
        
        # Get final latents (after all denoising steps)
        # training_batch.latents is [B, num_steps+1, C, T, H, W]
        # We want the final latents at index -1: [B, C, T, H, W]
        final_latents = training_batch.latents[:, -1]  # [B, C, T, H, W]
        
        # Get VAE for decoding
        vae = self.get_module("vae")
        device = final_latents.device
        
        # Decode latents to videos
        # Apply VAE normalization (Wan VAE specific)
        latents_to_decode = final_latents.to(vae.dtype)
        
        # Wan VAE requires denormalization before decoding
        if hasattr(vae, 'config') and hasattr(vae.config, 'latents_mean') and hasattr(vae.config, 'latents_std'):
            z_dim = getattr(vae.config, 'z_dim', latents_to_decode.shape[1])
            latents_mean = (
                torch.tensor(vae.config.latents_mean, device=device, dtype=latents_to_decode.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents_std = (
                1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=latents_to_decode.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents_to_decode = latents_to_decode / latents_std + latents_mean
        elif hasattr(vae, 'latents_mean') and hasattr(vae, 'latents_std'):
            z_dim = latents_to_decode.shape[1]
            latents_mean = (
                torch.tensor(vae.latents_mean, device=device, dtype=latents_to_decode.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents_std = (
                1.0 / torch.tensor(vae.latents_std, device=device, dtype=latents_to_decode.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents_to_decode = latents_to_decode / latents_std + latents_mean
        
        # Decode using VAE
        with torch.no_grad():
            videos = vae.decode(latents_to_decode)
            # VAE.decode returns tensor directly (not tuple)
            # Postprocess video: convert from [-1, 1] to [0, 1]
            videos = (videos / 2 + 0.5).clamp(0, 1)
        
        # Get prompts for reward computation
        prompts = training_batch.input_kwargs.get("prompts") if training_batch.input_kwargs else None
        if prompts is None:
            raise ValueError("Prompts not found in training_batch.input_kwargs. Required for reward computation.")
        
        # Compute rewards using reward models
        # Note: reward_models.compute_reward expects videos [B, C, T, H, W] and prompts [B]
        reward_scores = self.reward_models.compute_reward(videos, prompts)
        
        # Apply KL reward penalty if configured
        # In FlowGRPO: rewards["avg"] = rewards["avg"] - kl_reward * kl
        kl_reward = getattr(self.training_args.rl_args, 'kl_reward', 0.0)
        if kl_reward > 0 and training_batch.kl is not None:
            # training_batch.kl is [B, num_steps], we need to aggregate across timesteps
            # FlowGRPO uses the mean KL across timesteps
            kl_penalty = training_batch.kl.mean(dim=1)  # [B]
            reward_scores = reward_scores - kl_reward * kl_penalty
        
        # Store reward scores
        training_batch.reward_scores = reward_scores
        
        # Compute reward statistics
        reward_stats = compute_reward_statistics(training_batch.reward_scores)
        training_batch.reward_mean = reward_stats["reward_mean"]
        training_batch.reward_std = reward_stats["reward_std"]
        
        logger.debug("Rewards computed: mean=%.3f, std=%.3f, kl_reward=%.3f",
                    reward_stats["reward_mean"],
                    reward_stats["reward_std"],
                    kl_reward)
        
        return training_batch

    def compute_values(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute value predictions using value model.

        Args:
            training_batch: Training batch

        Returns:
            Updated training_batch with value predictions
        """
        # GRPO doesn't use value models, so this method is a no-op
        # Keeping for API compatibility but returning early
        return training_batch

    def compute_advantages(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute advantages using per-prompt stat tracking and normalization.
        
        This method implements Step 6 of GRPO training:
        1. Uses PerPromptStatTracker for per-prompt normalization (if enabled)
        2. Supports global std option
        3. Computes normalized advantages from rewards
        4. Computes returns (advantages + values if value model is used)
        5. Stores advantages, returns, and statistics in TrainingBatch
        
        Ported from FlowGRPO's advantage computation to work with FastVideo's TrainingBatch.
        
        Args:
            training_batch: Training batch with reward_scores [B] and prompt_ids [B, seq_len]
        
        Returns:
            Updated training_batch with:
            - advantages: Computed advantages [B]
            - returns: TD returns [B] (advantages if no value model)
            - advantage_mean: Mean advantage
            - advantage_std: Std of advantages
        """
        if training_batch.reward_scores is None:
            raise ValueError("Rewards must be computed before advantages")
        
        if self.stat_tracker is None:
            raise RuntimeError("Stat tracker not initialized. Call initialize_training_pipeline first.")
        
        logger.debug("Computing advantages with per-prompt stat tracking")
        
        # Get prompts from prompt_ids (decode token IDs to strings)
        if training_batch.prompt_ids is not None:
            tokenizer = self.get_module("tokenizer")
            prompt_ids_np = training_batch.prompt_ids.cpu().numpy()
            prompts = tokenizer.batch_decode(prompt_ids_np, skip_special_tokens=True)
        elif training_batch.input_kwargs is not None and "prompts" in training_batch.input_kwargs:
            prompts = training_batch.input_kwargs["prompts"]
        else:
            raise ValueError(
                "Cannot find prompts for stat tracking. "
                "Need either prompt_ids or input_kwargs['prompts']"
            )
        
        # Get rewards
        rewards = training_batch.reward_scores
        
        # Check if per-prompt stat tracking is enabled
        per_prompt_stat_tracking = getattr(
            self.training_args.rl_args, 
            'per_prompt_stat_tracking', 
            True  # Default to True for GRPO
        )
        
        if per_prompt_stat_tracking:
            # Use PerPromptStatTracker for per-prompt normalization
            # This computes (reward - mean_per_prompt) / std_per_prompt
            advantages_np = self.stat_tracker.update(
                prompts=prompts,
                rewards=rewards,
                type='grpo'  # GRPO-style normalization
            )
            advantages = torch.as_tensor(advantages_np, device=rewards.device, dtype=rewards.dtype)
        else:
            # Global normalization: (reward - global_mean) / global_std
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        
        # Get values (use zeros if algorithm doesn't use value model)
        if training_batch.values is not None:
            values = training_batch.values
        else:
            values = torch.zeros_like(rewards)
        
        # Compute returns: returns = advantages + values
        # For GRPO, we typically don't use value models, so returns = advantages
        returns = advantages + values
        
        # Store in training batch
        training_batch.advantages = advantages
        training_batch.returns = returns
        training_batch.advantage_mean = advantages.mean().item()
        training_batch.advantage_std = advantages.std().item()
        
        logger.debug("Advantages computed: mean=%.3f, std=%.3f, per_prompt=%s",
                    training_batch.advantage_mean,
                    training_batch.advantage_std,
                    per_prompt_stat_tracking)
        
        return training_batch

    def _compute_log_prob_for_timestep(
        self,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None = None,
        guidance_scale: float = 4.5,
        return_dt_and_std_dev_t: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
        """
        Compute log probability for a given timestep using current transformer.
        
        This is similar to FlowGRPO's compute_log_prob function, adapted for FastVideo.
        It computes the log probability of next_latents given latents under the current model.
        
        Args:
            latents: Current latents [B, C, T, H, W]
            next_latents: Next latents (target) [B, C, T, H, W]
            timesteps: Timesteps [B]
            prompt_embeds: Prompt embeddings [B, seq_len, hidden_dim]
            negative_prompt_embeds: Negative prompt embeddings for CFG [B, seq_len, hidden_dim]
            guidance_scale: Classifier-free guidance scale
            return_dt_and_std_dev_t: If True, return dt and std_dev_t separately
        
        Returns:
            If return_dt_and_std_dev_t=True:
                (prev_sample, log_prob, prev_sample_mean, std_dev_t, dt)
            Otherwise:
                (prev_sample, log_prob, prev_sample_mean, std_dev_t * sqrt_dt)
        """
        scheduler = self.get_module("scheduler")
        transformer = self.get_module("transformer")
        
        # Prepare latent input
        latent_model_input = latents.to(transformer.dtype)
        timestep = timesteps.to(transformer.dtype)
        
        # Predict noise with transformer
        if guidance_scale > 1.0 and negative_prompt_embeds is not None:
            # Classifier-free guidance: concatenate negative and positive prompts
            # For CFG, we need to run transformer twice or concatenate inputs
            # FlowGRPO concatenates: [negative_embeds, positive_embeds]
            latent_model_input_cfg = torch.cat([latent_model_input] * 2)
            prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])
            timestep_cfg = timestep.repeat(2)
            
            noise_pred = transformer(
                hidden_states=latent_model_input_cfg,
                timestep=timestep_cfg,
                encoder_hidden_states=prompt_embeds_cfg,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)
            
            # Split and apply guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            # No CFG
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)
        
        # Compute log probability using SDE step
        # Use next_latents as prev_sample to compute log prob of the actual transition
        return sde_step_with_logprob(
            scheduler,
            noise_pred.float(),
            timesteps,
            latents.float(),
            prev_sample=next_latents.float(),
            return_dt_and_std_dev_t=return_dt_and_std_dev_t
        )
    def compute_log_prob(self, transformer, sample, j, embeds, negative_embeds):
        scheduler = self.get_module("scheduler")
        if self.training_args.rl_args.cfg:
            noise_pred_text = transformer(
                hidden_states=sample["latents"][:, j],
                timestep=sample["timesteps"][:, j],
                encoder_hidden_states=embeds,  # Should contain both neg and pos embeds
                return_dict=False,
            )[0]
            noise_pred_uncond = transformer(
                hidden_states=sample["latents"][:, j],
                timestep=sample["timesteps"][:, j],
                encoder_hidden_states=negative_embeds,
                return_dict=False,
            )[0]
            noise_pred = (
                noise_pred_uncond
                + self.training_args.rl_args.guidance_scale
                * (noise_pred_text - noise_pred_uncond)
            )
        else:
            noise_pred = transformer(
                hidden_states=sample["latents"][:, j],
                timestep=sample["timesteps"][:, j],
                encoder_hidden_states=embeds,
                return_dict=False,
            )[0]
        
        # compute the log prob of next_latents given latents under the current model
        prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
            scheduler,
            noise_pred.float(),
            sample["timesteps"][:, j],
            sample["latents"][:, j].float(),
            prev_sample=sample["next_latents"][:, j].float(),
            return_dt_and_std_dev_t=True
        )

        return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt
    def _compute_grpo_loss(
        self,
        training_batch: TrainingBatch
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute GRPO loss (policy loss + KL loss).
        
        This function implements the GRPO training objective:
        1. Recomputes log probabilities for current policy at each timestep
        2. Computes reference log probabilities with adapter disabled (if using LoRA)
        3. Computes policy loss with clipping
        4. Computes KL loss using reference model
        5. Returns total loss and metrics
        
        Ported from FlowGRPO's training loop to work with FastVideo's TrainingBatch.
        
        Args:
            training_batch: Training batch with:
                - latents: [B, num_steps+1, C, T, H, W] - latents at each step
                - timesteps: [B, num_steps] - timesteps used
                - log_probs: [B, num_steps] - old log probs from sampling
                - advantages: [B, num_steps] or [B] - advantages
                - prompt_embeds: [B, seq_len, hidden_dim] - prompt embeddings
                - negative_prompt_embeds: [B, seq_len, hidden_dim] - negative embeddings (optional)
        
        Returns:
            total_loss: Total loss for backward pass
            metrics: Dictionary with loss components and diagnostics
        """
        # Get configuration from RLArgs
        # Note: CLI arguments map to RLArgs fields:
        # --rl-policy-clip-range -> grpo_policy_clip_range (via dest)
        # --rl-kl-beta -> kl_beta (via dest)
        clip_range = self.training_args.rl_args.grpo_policy_clip_range
        kl_beta = self.training_args.rl_args.kl_beta
        guidance_scale = 4.5  # Hardcoded
        adv_clip_max = 10.0  # Hardcoded (advantage clipping)
        
        # Get data from training batch
        latents = training_batch.latents  # [B, num_steps+1, C, T, H, W]
        timesteps = training_batch.timesteps  # [B, num_steps]
        old_log_probs = training_batch.old_log_probs  # [B, num_steps]
        advantages = training_batch.advantages  # [B, num_steps] or [B]
        
        # Get prompt embeddings
        # If not stored, recompute from prompts
        if training_batch.prompt_embeds is not None:
            prompt_embeds = training_batch.prompt_embeds
        elif training_batch.encoder_hidden_states is not None:
            prompt_embeds = training_batch.encoder_hidden_states
        else:
            # Recompute prompt embeddings from prompts
            prompts = training_batch.input_kwargs.get("prompts") if training_batch.input_kwargs else None
            if prompts is None:
                raise ValueError("Cannot find prompts or prompt embeddings in training_batch")
            
            # Encode prompts
            text_encoder = self.get_module("text_encoder")
            tokenizer = self.get_module("tokenizer")
            device = next(self.transformer.parameters()).device
            
            # Normalize to list
            if isinstance(prompts, str):
                prompts = [prompts]
            
            # Tokenize and encode
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = text_encoder(
                    text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    output_hidden_states=True,
                )
                prompt_embeds = outputs.last_hidden_state.to(self.transformer.dtype)
        
        # Get negative prompt embeddings
        negative_prompt_embeds = training_batch.negative_prompt_embeds
        if negative_prompt_embeds is None and guidance_scale > 1.0:
            # Generate negative prompt embeddings if needed
            text_encoder = self.get_module("text_encoder")
            tokenizer = self.get_module("tokenizer")
            device = next(self.transformer.parameters()).device
            
            batch_size = prompt_embeds.shape[0]
            negative_prompts = [""] * batch_size
            
            neg_text_inputs = tokenizer(
                negative_prompts,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                neg_outputs = text_encoder(
                    neg_text_inputs["input_ids"],
                    attention_mask=neg_text_inputs["attention_mask"],
                    output_hidden_states=True,
                )
                negative_prompt_embeds = neg_outputs.last_hidden_state.to(self.transformer.dtype)
        
        # Handle advantages shape: if [B], expand to [B, num_steps]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1).expand(-1, timesteps.shape[1])
        
        batch_size, num_steps = timesteps.shape
        
        # Accumulate losses across timesteps
        policy_losses = []
        kl_losses = []
        clip_fractions = []
        importance_ratios = []
        approx_kls = []
        
        # Get transformer for reference model computation
        transformer = self.get_module("transformer")
        
        # Loop over timesteps
        for j in range(num_steps):
            # Get latents and next_latents for this timestep
            latents_j = latents[:, j]  # [B, C, T, H, W]
            next_latents_j = latents[:, j + 1]  # [B, C, T, H, W]
            timesteps_j = timesteps[:, j]  # [B]
            old_log_probs_j = old_log_probs[:, j]  # [B]
            advantages_j = advantages[:, j]  # [B]
            
            # Compute log probability with current policy
            prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = self._compute_log_prob_for_timestep(
                latents_j,
                next_latents_j,
                timesteps_j,
                prompt_embeds,
                negative_prompt_embeds,
                guidance_scale,
                return_dt_and_std_dev_t=True
            )
            
            # Compute reference log probability with adapter disabled (if using LoRA)
            if kl_beta > 0:
                with torch.no_grad():
                    if hasattr(transformer, 'disable_adapter'):
                        with transformer.disable_adapter():
                            _, _, prev_sample_mean_ref, std_dev_t_ref, dt_ref = self._compute_log_prob_for_timestep(
                                latents_j,
                                next_latents_j,
                                timesteps_j,
                                prompt_embeds,
                                negative_prompt_embeds,
                                guidance_scale,
                                return_dt_and_std_dev_t=True
                            )
                    else:
                        # No adapter to disable, use current model (shouldn't happen in practice)
                        prev_sample_mean_ref = prev_sample_mean.detach()
                        std_dev_t_ref = std_dev_t.detach()
                        dt_ref = dt.detach()
                
                # Compute KL loss: KL = (mean_diff)^2 / (2 * (std_dev_t * dt)^2)
                # FlowGRPO uses: kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * (std_dev_t * dt_ref) ** 2)
                # For videos [B, C, T, H, W], we average over all spatial/channel dims except batch
                # Note: std_dev_t and dt_ref are already broadcast to [B, 1, 1, 1, 1]
                kl_loss_j = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3, 4), keepdim=True) / (2 * (std_dev_t * dt_ref) ** 2 + 1e-8)
                kl_loss_j = kl_loss_j.mean()  # Average over batch dimension
                kl_losses.append(kl_loss_j)
            else:
                kl_losses.append(torch.tensor(0.0, device=log_prob.device))
            
            # GRPO policy loss computation
            # Clip advantages
            advantages_j_clipped = torch.clamp(advantages_j, -adv_clip_max, adv_clip_max)
            
            # Compute importance ratio
            ratio = torch.exp(log_prob - old_log_probs_j)
            
            # Clipped surrogate objective
            unclipped_loss = -advantages_j_clipped * ratio
            clipped_loss = -advantages_j_clipped * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            policy_loss_j = torch.maximum(unclipped_loss, clipped_loss).mean()
            policy_losses.append(policy_loss_j)
            
            # Compute diagnostics
            with torch.no_grad():
                # Clip fraction
                clip_fraction_j = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).float().mean()
                clip_fractions.append(clip_fraction_j)
                
                # Importance ratio
                importance_ratios.append(ratio.mean())
                
                # Approximate KL (using log prob difference)
                approx_kl_j = 0.5 * torch.mean((log_prob - old_log_probs_j) ** 2)
                approx_kls.append(approx_kl_j)
        
        # Average losses across timesteps
        policy_loss = torch.stack(policy_losses).mean()
        kl_loss = torch.stack(kl_losses).mean() if kl_beta > 0 else torch.tensor(0.0, device=policy_loss.device)
        
        # Total loss
        total_loss = policy_loss + kl_beta * kl_loss
        
        # Compute metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item() if kl_beta > 0 else 0.0,
            "total_loss": total_loss.item(),
            "clip_fraction": torch.stack(clip_fractions).mean().item(),
            "importance_ratio_mean": torch.stack(importance_ratios).mean().item(),
            "approx_kl": torch.stack(approx_kls).mean().item(),
        }
        
        return total_loss, metrics

    def train_one_step(self, text_encoders, tokenizers, sample_neg_prompt_embeds, optimizer: torch.optim.Optimizer, global_step: int, train_timesteps: list[int], train_neg_prompt_embeds: torch.Tensor, epoch: int) -> TrainingBatch:
        """
        Train one step using GRPO algorithm.

        This method orchestrates the GRPO training step:
        1. Collect trajectories with current policy
        2. Compute rewards
        3. Compute advantages
        4. Compute GRPO loss (policy loss with clipping + KL regularization)
        5. Update policy

        Args:
            training_batch: Current training batch

        Returns:
            Updated training_batch with loss and metrics
        """
        # # Check if we're in warmup phase (do SFT instead of RL)
        # if training_batch.current_timestep < self.training_args.rl_args.rl_warmup_steps:
        #     logger.debug("In warmup phase, using standard SFT training")
        #     return super().train_one_step(training_batch)
        samples = self.collect_trajectories(epoch, text_encoders, tokenizers, sample_neg_prompt_embeds)
         
         # Shuffle  Samples for training, assuming samples is of the form as in FlowGRPO code
        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == self.training_args.rl_args.num_steps
        for inner_epoch in range(self.training_args.rl_args.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=self.device) # Shuffle samples along batch dimension
            samples = {k: v[perm] for k, v in samples.items()}
            
            perms = torch.stack(
                    [
                        # torch.randperm(num_timesteps, device=accelerator.device)
                        torch.arange(num_timesteps, device=self.device)
                        for _ in range(total_batch_size)
                    ]
                )  # shuffle along time dimension independently for each sample
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=self.device)[:, None],
                        perms,
                    ]
            micoe_batch = total_batch_size // (self.training_args.rl_args.sample_num_batches_per_epoch * self.training_args.rl_args.sample_time_per_prompt) # Change this according to RLargs config

            samples_batched = {
                k: v.reshape(-1, micoe_batch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                    dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
                ]

            # train
            self.transformer.train() # Review this, if it's correct 
            info = defaultdict(list)
            for i, sample in tqdm(
                    list[tuple[int, dict[str, Any]]](enumerate(samples_batched)),
                    # desc=f"Epoch {epoch}.{inner_epoch}: training", # Change this according to how for loops are in workflow
                    position=0,
                    # disable=not accelerator.is_local_main_process,
                ):
                if self.training_args.rl_args.cfg:
                        # concat negative prompts to sample prompts to avoid two forward passes
                        embeds = sample["prompt_embeds"]
                        negative_embeds = train_neg_prompt_embeds[:len(sample["prompt_embeds"])]
                else:
                    embeds = sample["prompt_embeds"]
                    negative_embeds = None
                for j in tqdm(
                        train_timesteps,
                        desc="Timestep",
                        position=1,
                        leave=False,
                        # disable=not accelerator.is_local_main_process,
                    ):
                        # with accelerator.accumulate(self.transformer): # Review this line is it really self.transformer or self.get_module("transformer")?
                            # with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = self.compute_log_prob(self.transformer, sample, j, embeds, negative_embeds) # Review this line
                            if self.training_args.rl_args.kl_beta > 0:
                                with torch.no_grad():
                                    with self.transformer.disable_adapter(): # Review this as well.
                                        prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref, dt_ref = self.compute_log_prob(self.transformer, sample, j, embeds, negative_embeds) # Review this line check for input parameters

                            # grpo logic
                            advantages = torch.clamp(
                                sample["advantages"][:, j],
                                -self.training_args.rl_args.adv_clip_max,
                                self.training_args.rl_args.adv_clip_max,
                            )
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * torch.clamp(
                                ratio,
                                1.0 - self.training_args.rl_args.grpo_policy_clip_range,
                                1.0 + self.training_args.rl_args.grpo_policy_clip_range,
                            )
                            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            if self.training_args.rl_args.kl_beta > 0:
                                kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * (std_dev_t * dt_ref) ** 2)
                                kl_loss = torch.mean(kl_loss)
                                loss = policy_loss + self.training_args.rl_args.kl_beta * kl_loss
                            else:
                                loss = policy_loss

                            info["approx_kl"].append(
                                0.5
                                * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                            )
                            info["clipfrac"].append(
                                torch.mean(
                                    (
                                        torch.abs(ratio - 1.0) > self.training_args.rl_args.grpo_policy_clip_range
                                    ).float()
                                )
                            )
                            info["policy_loss"].append(policy_loss)
                            if self.training_args.rl_args.kl_beta > 0:
                                info["kl_loss"].append(kl_loss)

                            info["loss"].append(loss)

                            # backward pass
                            # accelerator.backward(loss)
                            loss.backward()
                            
                            # if accelerator.sync_gradients:
                            #     accelerator.clip_grad_norm_(
                            #         transformer.parameters(), self.training_args.max_grad_norm # Review this line
                            #     )
                            torch.nn.utils.clip_grad_norm_(
                                self.transformer.parameters(), self.training_args.max_grad_norm
                )
                            optimizer.step()
                            optimizer.zero_grad()
                            info = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    # if accelerator.sync_gradients:
                    #     # assert (j == train_timesteps[-1]) and (
                    #     #     i + 1
                    #     # ) % config.train.gradient_accumulation_steps == 0
                    #     # log training-related stuff
                    #     info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    #     info = accelerator.reduce(info, reduction="mean")
                    #     info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    #     accelerator.log(info, step=global_step)
                    #     global_step += 1
                    #     info = defaultdict(list)
            # if config.train.ema:
            #     ema.step(transformer_trainable_parameters, global_step)

    def set_trainable(self) -> None:
        """Set which parameters should be trainable."""
        # Policy (transformer) is trainable
        for param in self.transformer.parameters():
            param.requires_grad = True

        # GRPO doesn't use value models, so no value model training needed

        # Freeze reward models (they should not be trained)
        if self.reward_models is not None:
            for param in self.reward_models.parameters():
                param.requires_grad = False

        logger.info("Set trainable parameters for RL training")

    
    def train(self) -> None:    
            # basic Accelerate and logging setup
        
        # number of timesteps within each trajectory to train on
        num_train_timesteps = int(self.training_args.rl_args.num_steps * self.training_args.rl_args.timestep_fraction)
        train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
        text_encoders = [self.get_module("text_encoder")]
        tokenizers = [self.get_module("tokenizer")]
        neg_prompt_embed = self.compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=512, device=self.device)
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.training_args.rl_args.sample_train_batch_size, 1, 1)
        train_neg_prompt_embeds = neg_prompt_embed.repeat(self.training_args.train_batch_size * self.training_args.rl_args.sample_time_per_prompt, 1, 1)
        global_step = 0
        first_epoch = 0
        for epoch in range(first_epoch, self.training_args.num_epochs):
            self.train_one_step(text_encoders, tokenizers, sample_neg_prompt_embeds, self.optimizer, global_step, train_timesteps, train_neg_prompt_embeds, epoch)



def create_rl_pipeline(
    model_path: str,
    training_args: TrainingArgs
) -> RLPipeline:
    """
    Factory function to create RL pipeline.

    Args:
        model_path: Path to pretrained model
        training_args: Training arguments with RL configuration

    Returns:
        Initialized RLPipeline
    """
    pipeline = RLPipeline(model_path, training_args)
    pipeline.initialize_training_pipeline(training_args)
    return pipeline
