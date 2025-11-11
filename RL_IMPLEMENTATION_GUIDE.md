# FastVideo RL/GRPO Pipeline Implementation Guide

**Created**: November 10, 2025
**Updated**: November 11, 2025
**Purpose**: Complete guide for implementing RL/GRPO pipeline in FastVideo
**Status**: Phase 1 Complete
**Scope**: VIDEO GENERATION ONLY (not image models)

---

## 🎯 IMPORTANT: VIDEO-ONLY SCOPE

This RL/GRPO implementation is **exclusively for VIDEO generation models**.

- ✅ **Supported**: FastVideo WAN, video diffusion models, T2V, I2V
- ❌ **NOT Supported**: Stable Diffusion 3.5, FLUX.1, any image-only models
- ✅ **Reward Models**: Video-specific (temporal coherence, motion quality, video-text alignment)
- ❌ **NOT Using**: PickScore, ImageReward, GenEval (image-only rewards)

See `RL_SCOPE.md` for complete scope documentation.

---

## Quick Navigation

This guide provides a complete roadmap for adding RL/GRPO capabilities to FastVideo. Four comprehensive reference documents are included:

### 1. **EXPLORATION_SUMMARY.md** - START HERE
**Executive summary with key findings and recommendations**
- Key architectural insights
- Current training infrastructure assessment
- Implementation roadmap (4 phases)
- Critical files to study (in order)
- Risk assessment & success criteria
- Estimated timeline: 2-4 weeks

### 2. **CODEBASE_STRUCTURE.md** - Full Technical Overview
**Complete architectural documentation**
- Project structure (directory layout)
- Training pipeline structure & class hierarchy
- Configuration system in detail
- Model architecture and registry
- Data loading & processing
- Existing RL/post-training capabilities
- Distributed training setup
- Checkpointing & state management
- 17 comprehensive sections

### 3. **KEY_FILES_REFERENCE.md** - Implementation Checklist
**Practical quick reference for developers**
- Core training infrastructure files with locations
- File dependency graph
- Training step flow trace
- Configuration modifications needed
- Critical patterns to follow
- Exact file locations and line numbers

### 4. **COMPONENT_APIS.md** - API Reference
**Function signatures and data structures**
- TrainingPipeline API
- TrainingBatch data structure
- TrainingArgs configuration
- Data loading API
- Scheduler & checkpointing APIs
- Logging/tracker API
- Utility functions
- Integration checklist

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
```
□ Create /fastvideo/training/rl_pipeline.py
  └─ Inherit from TrainingPipeline
  └─ ~400-500 LOC

□ Extend TrainingBatch in pipeline_batch_info.py
  └─ Add: rewards, advantages, log_probs, returns

□ Add RL parameters to TrainingArgs
  └─ RL hyperparameters (gamma, lambda, clip_range, etc.)

□ Create /fastvideo/training/reward_model.py
  └─ Wrapper for reward/value models
```

**Deliverable**: Skeleton RL pipeline that trains (even if just SFT baseline)

### Phase 2: Core RL Logic (Week 2-3)
```
□ Implement trajectory sampling
  └─ collect_trajectories() method

□ Implement advantage estimation
  └─ compute_gae() for GAE-lambda

□ Implement GRPO losses
  └─ policy_loss() and value_loss()

□ Override train_one_step() with RL logic
  └─ Alternating policy/value updates
```

**Deliverable**: RL pipeline that improves over baseline

### Phase 3: Integration & Validation (Week 3-4)
```
□ Add RL-specific logging metrics
  └─ policy_loss, value_loss, advantages stats

□ Implement policy comparison validation
  └─ Generate with old vs new policy

□ Multi-model checkpointing
  └─ Save policy & reward model separately

□ Create entry point script
  └─ Similar to wan_training_pipeline.py
```

**Deliverable**: Production-ready RL pipeline with full logging

### Phase 4: Optimization (As needed)
```
□ Parallel trajectory collection
□ Distributed advantage computation  
□ Memory optimization for long rollouts
□ VSA sparsity integration
```

---

## Starting Point: Minimal Example

The cleanest approach is to follow this pattern:

### Step 1: Minimal RL Pipeline Class
```python
# /fastvideo/training/rl_pipeline.py

from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.fastvideo_args import TrainingArgs

class RLPipeline(TrainingPipeline):
    """RL/GRPO training pipeline."""
    
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Create validation pipeline (same as WanTrainingPipeline)."""
        # Reuse pattern from wan_training_pipeline.py
        
    def train_one_step(self, training_batch):
        """Override with RL-specific logic."""
        # Collect trajectories
        # Compute advantages
        # Compute policy loss
        # Compute value loss
        # Return batch with metrics
```

### Step 2: Extend TrainingBatch
```python
# In pipeline_batch_info.py, add to TrainingBatch:

@dataclass
class TrainingBatch:
    # ... existing fields ...
    
    # RL-specific
    reward_scores: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    policy_loss: float = 0.0
    value_loss: float = 0.0
    advantage_mean: float = 0.0
```

### Step 3: Add RL Args
```python
# In TrainingArgs (fastvideo_args.py), add:

# RL parameters
rl_num_rollouts: int = 4
rl_lambda: float = 0.95  # GAE lambda
rl_gamma: float = 0.99   # Discount factor
rl_policy_cliprange: float = 0.2
rl_num_policy_updates: int = 5
rl_num_value_updates: int = 5
reward_model_path: str = ""
```

### Step 4: Entry Point
```python
# /fastvideo/training/rl_training_pipeline.py

from fastvideo.training.rl_pipeline import RLPipeline

def main(args):
    pipeline = RLPipeline.from_pretrained(
        args.model_path, 
        args=args
    )
    pipeline.train()

if __name__ == "__main__":
    # Parse args and call main() (same as wan_training_pipeline.py)
```

---

## Key Architectural Decisions

### 1. Model Architecture
```
Policy: WanTransformer3DModel (same as training)
Value/Reward: WanTransformer3DModel (can share weights or separate)
Scheduler: FlowMatchEulerDiscreteScheduler (reuse from existing)
```

### 2. Training Pattern
Follow `SelfForcingDistillationPipeline`:
- Alternating updates (policy, then value)
- Ratio-based scheduling (policy:value update ratio)
- Multi-GPU via sequence parallel + data parallel
- Gradient accumulation for long trajectories

### 3. Data Format
Reuse existing parquet format:
```python
{
    "video_latent": [B, C, T, H, W],      # VAE latents
    "prompt_embeds": [B, T_text, D],      # Text embeddings
    "attention_mask": [B, T_text],        # Text attention
}
# Can add: "reward_labels", "preference_pairs" (optional)
```

### 4. Distributed Strategy
```python
Sequence Parallel: sp_size > 1 (for large models)
Data Parallel: hsdp_replicate_dim > 1
Can collect rollouts in parallel across data parallel groups
```

---

## Testing Strategy

### Unit Tests
```python
# /fastvideo/tests/training/rl/

test_rl_pipeline.py
  - test_initialization()
  - test_gae_computation()
  - test_policy_loss()
  - test_value_loss()
  - test_train_one_step()

test_rl_distributed.py
  - test_distributed_rollouts()
  - test_distributed_advantage_computation()

test_rl_checkpointing.py
  - test_save_load_checkpoint()
```

### Integration Tests
```python
test_e2e_rl_training.py
  - Run full training loop
  - Check loss decreases
  - Check checkpoint saving
  - Check distributed communication
```

---

## Common Pitfalls to Avoid

1. **Data Format Mismatch**
   - RL reward values should match model's value range
   - Normalize advantages properly

2. **Distributed Synchronization**
   - Advantage computation must be synced across ranks
   - RNG state management crucial for reproducibility

3. **Memory Issues**
   - Store only necessary trajectory data
   - Clear intermediate buffers after use
   - Use gradient checkpointing

4. **Scheduler Compatibility**
   - Flow-match scheduler assumes diffusion model
   - May need to adapt or use standard LR schedulers
   - Consider warmup for RL (can be unstable early)

5. **Validation**
   - Generate samples with both old and new policy
   - Compare metrics fairly (same number of denoising steps)
   - Log improvements clearly

---

## File Checklist

### Files to Create
- [ ] `/fastvideo/training/rl_pipeline.py` (main class)
- [ ] `/fastvideo/training/rl_utils.py` (utility functions)
- [ ] `/fastvideo/training/reward_model.py` (model wrapper)
- [ ] `/fastvideo/training/rl_training_pipeline.py` (entry point)
- [ ] `/fastvideo/configs/training/rl_config.py` (config class)
- [ ] `/fastvideo/tests/training/rl/` (test files)

### Files to Modify
- [ ] `/fastvideo/fastvideo_args.py` (add RL args to TrainingArgs)
- [ ] `/fastvideo/pipelines/pipeline_batch_info.py` (extend TrainingBatch)
- [ ] `/fastvideo/training/__init__.py` (export RLPipeline)

### Files to Reference (Don't Modify)
- `/fastvideo/training/training_pipeline.py` (understand TrainingPipeline)
- `/fastvideo/training/distillation_pipeline.py` (understand two-model pattern)
- `/fastvideo/training/self_forcing_distillation_pipeline.py` (understand alternating training)
- `/fastvideo/training/training_utils.py` (reuse utility functions)

---

## API Integration Points

### Optimizer & Scheduler
```python
# Existing pattern - can reuse directly
self.policy_optimizer = torch.optim.AdamW(...)
self.value_optimizer = torch.optim.AdamW(...)
self.policy_scheduler = get_scheduler(...)  # From training_utils
self.value_scheduler = get_scheduler(...)
```

### Data Loading
```python
# Existing pattern - can reuse
self.train_dataloader = build_parquet_map_style_dataloader(...)
```

### Checkpointing
```python
# Existing pattern - can extend
ModelWrapper(self.policy)
ModelWrapper(self.value_model)
OptimizerWrapper(self.policy, self.policy_optimizer)
OptimizerWrapper(self.value_model, self.value_optimizer)
```

### Logging
```python
# Existing pattern - can extend with new metrics
tracker.log({
    "train/policy_loss": policy_loss,
    "train/value_loss": value_loss,
    "train/advantages_mean": advantages.mean(),
    "train/rewards_mean": rewards.mean(),
})
```

### Distributed
```python
# Existing utilities - can reuse
sp_group = get_sp_group()
world_group = get_world_group()
shard_latents_across_sp(latents, num_latent_t=...)
```

---

## Example Command Line

```bash
# Training script
python -m fastvideo.training.rl_training_pipeline \
  --model-path /path/to/pretrained \
  --data-path /path/to/training_data.parquet \
  --reward-model-path /path/to/reward_model \
  --max-train-steps 50000 \
  --learning-rate 1e-5 \
  --rl-gamma 0.99 \
  --rl-lambda 0.95 \
  --rl-num-rollouts 4 \
  --output-dir ./rl_checkpoints \
  --num-gpus 8 \
  --sp-size 2 \
  --hsdp-replicate-dim 4 \
  --log-validation \
  --validation-steps 1000
```

---

## Success Metrics

### Training Metrics to Monitor
- Policy loss decreasing over time
- Value loss decreasing over time
- Average advantages approaching 0 after normalization
- KL divergence between old and new policy (if tracked)
- Gradient norms stable (not exploding/vanishing)

### Validation Metrics
- Quality improvement in generated videos
- Consistency across rollout samples
- No mode collapse
- Prompt adherence improvement

### System Metrics
- GPU memory usage stable
- No distributed communication issues
- Checkpoints save correctly
- Training speed acceptable

---

## Reference Documents

All documentation is in the FastVideo root directory:

1. **EXPLORATION_SUMMARY.md** - Executive summary & roadmap
2. **CODEBASE_STRUCTURE.md** - Full architectural details
3. **KEY_FILES_REFERENCE.md** - Quick reference for developers
4. **COMPONENT_APIS.md** - API signatures & integration

---

## Support Resources

### Code Examples
- `wan_training_pipeline.py` - Minimal pipeline subclass
- `distillation_pipeline.py` - Two-model training pattern
- `self_forcing_distillation_pipeline.py` - Alternating training pattern
- `training_pipeline.py` - Base class with full implementation

### Key Functions
- `get_scheduler()` - LR scheduler creation
- `save_checkpoint()` - Checkpoint saving
- `load_checkpoint()` - Checkpoint loading
- `clip_grad_norm_while_handling_failing_dtensor_cases()` - Gradient management
- `shard_latents_across_sp()` - Distributed sharding
- `build_parquet_map_style_dataloader()` - Data loading

### Configuration
- `TrainingArgs` - All training hyperparameters
- `PipelineConfig` - Model configuration
- Model registry - Load any supported model

---

## Next Steps

1. Read **EXPLORATION_SUMMARY.md** (15 min)
2. Review **KEY_FILES_REFERENCE.md** (30 min)
3. Study key files in order:
   - `training_pipeline.py` (understand base class)
   - `self_forcing_distillation_pipeline.py` (understand pattern)
   - `wan_training_pipeline.py` (understand minimal subclass)
4. Create skeleton `/fastvideo/training/rl_pipeline.py`
5. Implement Phase 1: Foundation
6. Iterate through Phases 2-4

---

## Questions & Clarifications

### Q: Should we reuse or separate policy/value models?
**A**: Start with separate models for clarity. Can share backbone later if needed.

### Q: How to handle trajectory length variation?
**A**: Pad to max length, use attention masks. Standard practice.

### Q: How to integrate with existing VAE?
**A**: Keep VAE frozen. Only train policy and value models.

### Q: What about on-policy data distribution?
**A**: Collect fresh rollouts each training step. Can batch multiple rollouts.

### Q: How to handle backward pass through sampling?
**A**: Use log_prob computation from transformer output (not sampling operation).

---

## Contact & Updates

For questions about this implementation guide or FastVideo architecture:
- Refer to the embedded documentation
- Study the reference implementations (distillation, self-forcing)
- Check existing tests for patterns

---

**Happy implementing! The foundation is solid, and the infrastructure is comprehensive.**
**Estimated effort: 2-4 weeks to production-ready RL pipeline.**
