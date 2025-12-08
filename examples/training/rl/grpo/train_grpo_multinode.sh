#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Multi-Node GRPO Training Script (SLURM)
#
# This script trains a video generation model using GRPO across multiple nodes.
# Designed for SLURM cluster environments.
#
# Usage:
#   sbatch train_grpo_multinode.sh

#SBATCH --job-name=grpo_train
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --exclusive

set -e

# Environment setup
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

# Multi-node configuration
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "=== Multi-Node GRPO Training ==="
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "World size: $WORLD_SIZE"
echo "================================"

# Model and data paths
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR="data/rl_training_data/"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
OUTPUT_DIR="checkpoints/wan_grpo_multinode"

# Training arguments
training_args=(
    --tracker_project_name "wan_grpo_multinode"
    --output_dir "$OUTPUT_DIR"
    --max_train_steps 20000
    --train_batch_size 1
    --train_sp_batch_size 1
    --gradient_accumulation_steps 2
    --num_latent_t 20
    --num_height 480
    --num_width 832
    --num_frames 77
    --enable_gradient_checkpointing_type "full"
)

# Parallel arguments for multi-node
parallel_args=(
    --num_gpus $WORLD_SIZE
    --sp_size $GPUS_PER_NODE
    --tp_size 1
    --hsdp_replicate_dim $NNODES
    --hsdp_shard_dim $GPUS_PER_NODE
)

# Model arguments
model_args=(
    --model_path "$MODEL_PATH"
    --pretrained_model_name_or_path "$MODEL_PATH"
)

# Dataset arguments
dataset_args=(
    --data_path "$DATA_DIR"
    --dataloader_num_workers 4
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
    --learning_rate 5e-6
    --mixed_precision "bf16"
    --weight_only_checkpointing_steps 1000
    --training_state_checkpointing_steps 2000
    --weight_decay 1e-4
    --max_grad_norm 1.0
    --lr_scheduler "cosine"
    --lr_warmup_steps 200
)

# RL/GRPO-specific arguments
rl_args=(
    --rl_mode
    --rl_algorithm "grpo"
    --rl_num_rollouts 4
    --rl_rollout_steps "20,30"
    --rl_noise_injection_min 10
    --rl_noise_injection_max 40
    --rl_num_denoising_steps 2
    --rl_gamma 0.99
    --rl_lambda 0.95
    --rl_normalize_advantages
    --rl_policy_clip_range 0.2
    --rl_value_clip_range 0.2
    --rl_target_kl 0.01
    --rl_entropy_coef 0.0
    --rl_value_loss_coef 0.5
    --rl_use_grpo_guard
    --rl_ratio_norm_correction
    --rl_gradient_reweighting
    --rl_max_importance_ratio 10.0
    --rl_warmup_steps 1000
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

# Launch distributed training
srun torchrun \
    --nnodes $NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    fastvideo/training/wan_rl_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${rl_args[@]}" \
    "${miscellaneous_args[@]}"

echo "Multi-node GRPO training complete!"
