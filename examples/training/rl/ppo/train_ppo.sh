#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# PPO (Proximal Policy Optimization) Training Script
#
# This script trains a video generation model using standard PPO.
# Supports single-GPU and multi-GPU training via torchrun.
#
# Usage:
#   Single GPU:  NUM_GPUS=1 ./train_ppo.sh
#   Multi GPU:   NUM_GPUS=4 ./train_ppo.sh
#   8 GPU:       NUM_GPUS=8 ./train_ppo.sh

set -e

# Environment setup
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=${MASTER_PORT:-29501}

# Model and data paths (modify these for your setup)
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
DATA_DIR="${DATA_DIR:-data/rl_training_data/}"
VALIDATION_DATASET_FILE="${VALIDATION_DATASET_FILE:-$(dirname "$0")/validation.json}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/wan_ppo_training}"

# GPU configuration
NUM_GPUS="${NUM_GPUS:-4}"

echo "=== PPO Training Configuration ==="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Algorithm: PPO"
echo "=================================="

# Training arguments
training_args=(
    --tracker_project_name "wan_ppo_training"
    --output_dir "$OUTPUT_DIR"
    --max_train_steps 10000
    --train_batch_size 1
    --train_sp_batch_size 1
    --gradient_accumulation_steps 4
    --num_latent_t 20
    --num_height 480
    --num_width 832
    --num_frames 77
    --enable_gradient_checkpointing_type "full"
)

# Parallel arguments for multi-GPU
parallel_args=(
    --num_gpus $NUM_GPUS
    --sp_size $NUM_GPUS
    --tp_size 1
    --hsdp_replicate_dim 1
    --hsdp_shard_dim $NUM_GPUS
)

# Model arguments
model_args=(
    --model_path "$MODEL_PATH"
    --pretrained_model_name_or_path "$MODEL_PATH"
)

# Dataset arguments
dataset_args=(
    --data_path "$DATA_DIR"
    --dataloader_num_workers 2
)

# Validation arguments
validation_args=(
    --log_validation
    --validation_dataset_file "$VALIDATION_DATASET_FILE"
    --validation_steps 500
    --validation_sampling_steps "50"
    --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
    --learning_rate 3e-5
    --mixed_precision "bf16"
    --weight_only_checkpointing_steps 1000
    --training_state_checkpointing_steps 2000
    --weight_decay 1e-4
    --max_grad_norm 0.5
    --lr_scheduler "cosine"
    --lr_warmup_steps 100
)

# RL/PPO-specific arguments
rl_args=(
    --rl_mode
    --rl_algorithm "ppo"
    # Trajectory collection
    --rl_num_rollouts 4
    --rl_rollout_steps "20,30"
    --rl_noise_injection_min 10
    --rl_noise_injection_max 40
    --rl_num_denoising_steps 2
    # GAE advantage estimation
    --rl_gamma 0.99
    --rl_lambda 0.95
    --rl_use_gae
    --rl_normalize_advantages
    # PPO policy optimization
    --rl_policy_clip_range 0.2
    --rl_value_clip_range 0.2
    --rl_num_policy_epochs 1
    --rl_num_value_epochs 1
    --rl_target_kl 0.015
    --rl_entropy_coef 0.01
    --rl_value_loss_coef 0.5
    # Training schedule
    --rl_warmup_steps 500
    # Reward configuration
    --reward_model_types "dummy"
    --reward_weights "1.0"
    --value_model_share_backbone
)

# Miscellaneous arguments
miscellaneous_args=(
    --inference_mode False
    --checkpoints_total_limit 3
    --dit_precision "fp32"
    --seed 42
)

# Run training
echo "Starting PPO training with $NUM_GPUS GPU(s)..."

torchrun \
    --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    fastvideo/training/wan_rl_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${rl_args[@]}" \
    "${miscellaneous_args[@]}"

echo "PPO training complete!"
