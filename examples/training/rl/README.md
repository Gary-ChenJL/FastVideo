# RL Training Examples

This directory contains example scripts for training video generation models using Reinforcement Learning algorithms.

## Supported Algorithms

### GRPO (Group Relative Policy Optimization)
Location: `grpo/`

GRPO is a variant of PPO designed for generative models with additional safety mechanisms (GRPO-Guard):
- PPO-style clipped surrogate objective
- RatioNorm correction for importance sampling bias
- Gradient reweighting across denoising steps

**Best for:** General-purpose RL training with robust convergence.

### PPO (Proximal Policy Optimization)
Location: `ppo/`

Standard PPO implementation with:
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Value function clipping

**Best for:** Stable on-policy learning when you have a good reward model.

### DPO (Direct Preference Optimization)
Location: `dpo/`

Offline preference learning that doesn't require a reward model:
- Learns directly from preference pairs (chosen vs rejected)
- Implicit reward through KL-constrained optimization
- No online sampling required

**Best for:** When you have preference data but no explicit reward model.

## Quick Start

### Single GPU Training

```bash
# GRPO
cd grpo && NUM_GPUS=1 ./train_grpo.sh

# PPO
cd ppo && NUM_GPUS=1 ./train_ppo.sh

# DPO
cd dpo && NUM_GPUS=1 ./train_dpo.sh
```

### Multi-GPU Training

```bash
# 4 GPU GRPO training
cd grpo && NUM_GPUS=4 ./train_grpo.sh

# 8 GPU PPO training
cd ppo && NUM_GPUS=8 ./train_ppo.sh
```

### Multi-Node Training (SLURM)

```bash
# Submit GRPO multi-node job
cd grpo && sbatch train_grpo_multinode.sh
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NUM_GPUS` | Number of GPUs per node | 4 |
| `MODEL_PATH` | Path to pretrained model | Wan-AI/Wan2.1-T2V-1.3B-Diffusers |
| `DATA_DIR` | Path to training data | data/rl_training_data/ |
| `OUTPUT_DIR` | Checkpoint output directory | checkpoints/wan_*_training |
| `MASTER_PORT` | Distributed training port | 29500 |

### Key RL Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--rl_mode` | Enable RL training mode | False |
| `--rl_algorithm` | Algorithm to use (grpo/ppo/dpo) | grpo |
| `--rl_policy_clip_range` | PPO clipping range | 0.2 |
| `--rl_gamma` | Discount factor | 0.99 |
| `--rl_lambda` | GAE lambda | 0.95 |
| `--rl_target_kl` | Target KL for early stopping | 0.01 |
| `--rl_warmup_steps` | SFT warmup before RL | 500 |

### GRPO-Specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--rl_use_grpo_guard` | Enable GRPO-Guard safety | True |
| `--rl_ratio_norm_correction` | RatioNorm correction | True |
| `--rl_gradient_reweighting` | Gradient reweighting | True |
| `--rl_max_importance_ratio` | Max importance ratio | 10.0 |

## Data Preparation

### For GRPO/PPO
Requires standard video training data with text captions. The reward model will evaluate generated videos during training.

### For DPO
Requires preference data with chosen/rejected pairs. Each sample should contain:
- A prompt
- Chosen video (preferred output)
- Rejected video (non-preferred output)

## Monitoring

Training progress is logged to Weights & Biases. Key metrics to monitor:

- `policy_loss`: Should decrease over time
- `kl_divergence`: Should stay below `rl_target_kl`
- `clip_fraction`: Fraction of clipped ratios (high = may need lower learning rate)
- `reward_mean`: Average reward (should increase)
- `advantage_mean`: Should be ~0 after normalization

## Troubleshooting

### High KL Divergence
- Reduce learning rate
- Increase `rl_warmup_steps`
- Lower `rl_policy_clip_range`

### Training Instability
- Enable gradient checkpointing
- Reduce batch size
- Enable GRPO-Guard mechanisms

### OOM Errors
- Reduce `train_batch_size`
- Enable gradient checkpointing with `--enable_gradient_checkpointing_type "full"`
- Use `--value_model_share_backbone` to share backbone between policy and value
