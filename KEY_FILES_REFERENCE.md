# FastVideo - Key Files Reference Guide

## CRITICAL FILES FOR RL PIPELINE IMPLEMENTATION

### Core Training Infrastructure
```
/home/user/FastVideo/fastvideo/
├── training/
│   ├── training_pipeline.py              # BASE CLASS - Start here!
│   │   ├── TrainingPipeline (abstract)
│   │   ├── initialize_training_pipeline()
│   │   ├── train_one_step()
│   │   └── train()
│   │
│   ├── training_utils.py                 # Utilities - ESSENTIAL
│   │   ├── get_scheduler()
│   │   ├── normalize_dit_input()
│   │   ├── shard_latents_across_sp()
│   │   ├── clip_grad_norm()
│   │   ├── load_checkpoint()
│   │   ├── save_checkpoint()
│   │   └── EMA_FSDP class
│   │
│   ├── checkpointing_utils.py            # Stateful saving
│   ├── trackers.py                       # Logging infrastructure
│   ├── activation_checkpoint.py          # Gradient checkpointing
│   │
│   └── [Reference implementations]
│       ├── distillation_pipeline.py      # Two-model approach
│       └── self_forcing_distillation_pipeline.py  # Alternating training

├── fastvideo_args.py                     # Configuration (MODIFY HERE)
│   ├── FastVideoArgs class (line 84)
│   ├── TrainingArgs class (line 641)
│   └── ExecutionMode enum

├── pipelines/
│   ├── pipeline_batch_info.py            # Data structures
│   │   ├── ForwardBatch
│   │   ├── TrainingBatch                 # EXTEND THIS
│   │   └── PipelineLoggingInfo
│   │
│   ├── lora_pipeline.py                  # Base inference pipeline
│   └── composed_pipeline_base.py

├── dataset/
│   ├── parquet_dataset_map_style.py      # Main dataloader
│   ├── validation_dataset.py             # Validation data
│   └── __init__.py                       # build_parquet_map_style_dataloader()

├── models/
│   ├── registry.py                       # Model loading
│   ├── dits/
│   │   ├── wanvideo.py                   # Main transformer
│   │   └── base.py
│   └── schedulers/
│       └── scheduling_flow_match_euler_discrete.py

├── configs/
│   ├── pipelines/
│   │   ├── wan.py                        # Config for Wan model
│   │   └── base.py
│   └── sample/
│       └── base.py

└── distributed/
    ├── parallel_state.py
    └── utils.py
```

---

## FILE DEPENDENCY GRAPH FOR RL PIPELINE

```
[Create RLPipeline]
    ↓
TrainingPipeline (abstract base)
    ↓ uses ↓ depends on
┌───────────────────────────────────────────┐
│  training_utils.py                        │
│  - Schedulers, checkpointing, loss utils  │
└───────────────────────────────────────────┘
    ↓ loads models from ↓
┌───────────────────────────────────────────┐
│  models/registry.py                       │
│  - Transformer, VAE, encoder registry     │
└───────────────────────────────────────────┘
    ↓ processes data from ↓
┌───────────────────────────────────────────┐
│  dataset/parquet_dataset_map_style.py     │
│  - TrainingBatch loading                  │
└───────────────────────────────────────────┘
    ↓ uses config from ↓
┌───────────────────────────────────────────┐
│  fastvideo_args.py (TrainingArgs)         │
│  - All hyperparameters                    │
└───────────────────────────────────────────┘
    ↓ manages state with ↓
┌───────────────────────────────────────────┐
│  training/checkpointing_utils.py          │
│  - Save/load model, optimizer, scheduler  │
└───────────────────────────────────────────┘
    ↓ logs to ↓
┌───────────────────────────────────────────┐
│  training/trackers.py                     │
│  - W&B, TensorBoard logging               │
└───────────────────────────────────────────┘
```

---

## EXAMPLE: TRACING THROUGH A TRAINING STEP

### File flow for: `pipeline.train_one_step(training_batch)`

1. **pipeline_batch_info.py** - TrainingBatch created
2. **training_pipeline.py::train_one_step()**
   - Calls `_prepare_training()` -> clears gradients
   - Calls `_get_next_batch()` -> gets data from dataloader
   - Calls `_normalize_dit_input()` 
   - Calls `_prepare_dit_inputs()`
3. **training_utils.py**
   - `normalize_dit_input()`
   - `shard_latents_across_sp()` -> distributed sharding
   - `get_scheduler()` lookups
4. **fastvideo_args.py**
   - Access training_args for timestep weighting, CFG rate
   - Access pipeline_config for model dimensions
5. **models/dits/wanvideo.py**
   - Forward pass of transformer
6. **training_pipeline.py**
   - `_transformer_forward_and_compute_loss()`
7. **training_utils.py**
   - `clip_grad_norm_while_handling_failing_dtensor_cases()`
   - Optimizer step

---

## SAMPLE CONFIGURATION MODIFICATIONS FOR RL

### In TrainingArgs (fastvideo_args.py at line 641+)

```python
@dataclasses.dataclass
class TrainingArgs(FastVideoArgs):
    # ... existing fields ...
    
    # NEW RL-specific fields to add:
    
    # Reward model
    reward_model_path: str = ""
    reward_model_learning_rate: float = 0.0
    reward_model_weight_decay: float = 0.0
    
    # Policy optimization
    rl_num_rollouts: int = 4
    rl_rollout_length: int = 1000  # denoising steps
    rl_lambda: float = 0.95  # GAE lambda
    rl_gamma: float = 0.99  # discount factor
    rl_num_policy_updates: int = 5  # Updates per rollout
    rl_num_value_updates: int = 5  # Updates per rollout
    rl_policy_cliprange: float = 0.2  # PPO clip range
    rl_value_cliprange: float = 0.2  # Value clip range
    
    # Advantage computation
    rl_normalize_advantages: bool = True
    rl_advantage_epsilon: float = 1e-8
    
    # Trajectory collection
    rl_collect_trajectory_batch_size: int = 1
    rl_parallel_rollouts: bool = False
```

---

## QUICK START: IMPLEMENTING A SIMPLE RL PIPELINE

### Step 1: Create `/fastvideo/training/rl_pipeline.py`
- Inherit from `TrainingPipeline`
- Implement `initialize_validation_pipeline()`
- Override `train_one_step()` with RL logic

### Step 2: Extend `/fastvideo/pipelines/pipeline_batch_info.py`
- Add RL-specific fields to `TrainingBatch`:
  ```python
  reward_score: torch.Tensor = None
  log_probs: torch.Tensor = None
  advantages: torch.Tensor = None
  returns: torch.Tensor = None
  policy_loss: float = 0.0
  value_loss: float = 0.0
  ```

### Step 3: Create `/fastvideo/training/rl_utils.py`
- Implement `compute_gae()` for advantage estimation
- Implement `compute_policy_loss()` 
- Implement `collect_trajectories()`

### Step 4: Update `/fastvideo/fastvideo_args.py`
- Add RL parameters to `TrainingArgs`
- Add validation in `check_fastvideo_args()`

### Step 5: Create entry point
- `/fastvideo/training/rl_training_pipeline.py`
- Similar pattern to `wan_training_pipeline.py`

---

## CRITICAL PATTERNS TO FOLLOW

### Pattern 1: Distributed Training
```python
# From training_pipeline.py
self.transformer.train()
self.optimizer.zero_grad()

# Gradient accumulation
for _ in range(self.training_args.gradient_accumulation_steps):
    # Forward pass
    # Backward pass
    loss.backward()

# Clip and step
grad_norm = clip_grad_norm(self.transformer, self.training_args.max_grad_norm)
self.optimizer.step()
self.lr_scheduler.step()
```

### Pattern 2: Checkpointing
```python
# From checkpointing_utils.py
ModelWrapper(self.transformer)
OptimizerWrapper(self.transformer, self.optimizer)
SchedulerWrapper(self.lr_scheduler)
RandomStateWrapper(self.noise_generator)

# All are torch.distributed.checkpoint.stateful.Stateful
# Can be passed to torch.distributed.checkpoint.save()
```

### Pattern 3: Logging
```python
# From training_pipeline.py
with self.tracker.timed("timing/optimizer_step"):
    self.optimizer.step()

# Log scalars
tracker.log({
    "train/loss": loss,
    "train/grad_norm": grad_norm,
    "train/learning_rate": lr
})
```

### Pattern 4: Configuration
```python
# Always use training_args for hyperparameters
if self.training_args.lora_training:
    # apply lora
    
num_warmup = self.training_args.lr_warmup_steps
max_steps = self.training_args.max_train_steps
```

---

## EXACT FILE LOCATIONS

| Task | File | Line/Function |
|------|------|---------------|
| Base training loop | `training/training_pipeline.py` | line 566 `def train()` |
| Single step logic | `training/training_pipeline.py` | line 506 `def train_one_step()` |
| Optimizer creation | `training/training_pipeline.py` | line 133 AdamW creation |
| Checkpoint saving | `training/training_utils.py` | `save_checkpoint()` |
| Dataloader setup | `training/training_pipeline.py` | line 179 `build_parquet_map_style_dataloader()` |
| Config args | `fastvideo_args.py` | line 641 `class TrainingArgs` |
| Batch structure | `pipelines/pipeline_batch_info.py` | line 203 `class TrainingBatch` |
| Schedulers | `models/schedulers/` | various scheduler files |
| Model loading | `models/registry.py` | Model registry dict |
| Logging | `training/trackers.py` | line 275 |

