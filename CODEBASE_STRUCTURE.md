# FastVideo Codebase Structure & Architecture Overview
## Comprehensive Analysis for RL/Post-Training Pipeline Implementation

### 1. PROJECT STRUCTURE

#### Main Directory Layout
```
/home/user/FastVideo/
├── fastvideo/                    # Main package
│   ├── training/                 # Training pipelines & utilities (7936 LOC)
│   ├── pipelines/                # Inference & preprocessing pipelines
│   ├── models/                   # Model architectures & components
│   ├── dataset/                  # Data loading & processing
│   ├── configs/                  # Configuration system
│   ├── attention/                # Attention backends (VSA, MoBA, STA)
│   ├── distributed/              # Distributed training utilities
│   ├── entrypoints/              # CLI & entry points
│   ├── fastvideo_args.py         # Main argument parsing (1100+ LOC)
│   └── [other utilities]
├── examples/                     # Inference examples
├── scripts/                      # Dataset prep & conversion scripts
├── csrc/                         # CUDA/compiled extensions
└── docs/                         # Documentation
```

---

### 2. TRAINING PIPELINE STRUCTURE

#### Core Training Files (in `/fastvideo/training/`)
| File | LOC | Purpose |
|------|-----|---------|
| `training_pipeline.py` | 902 | Base abstract training pipeline |
| `training_utils.py` | 1783 | Optimization, checkpointing, schedulers |
| `distillation_pipeline.py` | 1742 | DMD-style distillation (teacher-critic) |
| `self_forcing_distillation_pipeline.py` | 1246 | Self-forcing with alternating updates |
| `wan_training_pipeline.py` | 74 | Wan-specific training pipeline |
| `wan_i2v_training_pipeline.py` | 216 | Wan image-to-video training |
| `trackers.py` | 275 | Logging to W&B, tensorboard, etc. |
| `checkpointing_utils.py` | 118 | Checkpoint save/load utilities |
| `activation_checkpoint.py` | 90 | Gradient checkpointing |

#### Training Pipeline Class Hierarchy
```
TrainingPipeline (abstract)
├── LoRAPipeline (base)
│   ├── ComposedPipelineBase (inference pipeline)
│   └── Distillation-related
│
├── DistillationPipeline
│   └── SelfForcingDistillationPipeline
│
├── WanTrainingPipeline
├── WanI2VTrainingPipeline
└── ODE Causal Pipeline
```

#### Key Training Methods in TrainingPipeline

```python
def initialize_training_pipeline(training_args: TrainingArgs) -> None:
    # Sets up: device, distributed groups, optimizer, scheduler
    # Initializes data loaders, tracker, validation pipeline
    # Applies gradient checkpointing

def train_one_step(training_batch: TrainingBatch) -> TrainingBatch:
    # Gradient accumulation loop
    # Normalizes DIT inputs
    # Creates noisy model inputs
    # Shards latents across sequence parallelism
    # Builds attention metadata
    # Forward pass + loss computation
    # Clip grad norm, optimizer step

def train() -> None:
    # Main training loop
    # Progress tracking with tqdm
    # Step-based logging
    # Checkpoint saving/resuming
    # Validation sampling
```

---

### 3. CONFIGURATION SYSTEM

#### TrainingArgs (extends FastVideoArgs)
File: `/fastvideo/fastvideo_args.py` (class at line 641)

**Key Training Parameters:**
```python
# Data
data_path: str
dataloader_num_workers: int
training_cfg_rate: float  # Classifier-free guidance dropout

# Model dimensions
num_height, num_width, num_frames: int
num_latent_t: int
train_batch_size, train_sp_batch_size: int

# Optimization
learning_rate: float
max_train_steps: int
gradient_accumulation_steps: int
weight_decay: float
betas: str  # "0.9,0.999" format
enable_gradient_checkpointing_type: str | None

# Learning rate scheduling
lr_scheduler: str
lr_warmup_steps: int
lr_num_cycles: int
lr_power: float
min_lr_ratio: float

# Validation & logging
validation_dataset_file: str
validation_sampling_steps: str
validation_steps: float
log_validation: bool
trackers: list[str]
output_dir: str

# Distillation-specific
generator_update_interval: int
dfake_gen_update_ratio: int  # Gen vs critic training ratio
min_timestep_ratio, max_timestep_ratio: float
real_score_guidance_scale: float

# Self-forcing specific
num_frame_per_block: int
independent_first_frame: bool
enable_gradient_masking: bool
gradient_mask_last_n_frames: int
context_noise: int

# VSA sparsity decay
VSA_decay_rate: float
VSA_decay_interval_steps: int

# LoRA parameters
lora_rank: int | None
lora_training: bool
```

#### Pipeline Configuration
- Located in `/fastvideo/configs/pipelines/`
- Different configs for: Wan, HunyuanVideo, StepVideo, Cosmos
- Defines model, VAE, text encoder, scheduler, precision per component

---

### 4. MODEL ARCHITECTURE

#### Model Registry (`/fastvideo/models/registry.py`)
Supports multiple architectures:
- **DiT Models**: WanTransformer3DModel, HunyuanVideoTransformer3DModel, CausalWanTransformer3DModel, CosmosTransformer3DModel
- **Text Encoders**: T5, CLIP, Llama, UMT5
- **Image Encoders**: CLIP Vision
- **VAEs**: WanVAE, HunyuanVAE, StepVideoVAE
- **Schedulers**: FlowMatchEuler, UniPCMultistep, SelfForcingFlowMatch

#### DIT Model Files (`/fastvideo/models/dits/`)
```
wanvideo.py (34529 LOC)      - Main Wan architecture
causal_wanvideo.py (30465)   - Causal variant
hunyuanvideo.py (35795)      - HunyuanVideo architecture
cosmos.py (29288)            - Cosmos architecture
stepvideo.py (26835)         - StepVideo architecture
base.py (4733)               - Base DiT class
```

#### Schedulers (`/fastvideo/models/schedulers/`)
- Flow-Match based (continuous diffusion)
- SD3-style timestep weighting (logit_normal, mode)
- Self-forcing scheduler for critic training

---

### 5. DATA LOADING & PROCESSING

#### Dataset Implementations (`/fastvideo/dataset/`)
| File | Purpose |
|------|---------|
| `parquet_dataset_map_style.py` | Map-style dataloader for parquet files |
| `parquet_dataset_iterable_style.py` | Iterable-style for streaming |
| `preprocessing_datasets.py` | VideoCaptionMergedDataset, TextDataset |
| `validation_dataset.py` | ValidationDataset (CSV, JSON, Parquet, Arrow formats) |
| `latent_datasets.py` | Pre-encoded latent datasets |
| `transform.py` | Video transformations (crop, resize, normalize) |

#### Data Loading Pipeline
```python
build_parquet_map_style_dataloader(
    data_path,
    batch_size,
    parquet_schema,
    num_workers,
    cfg_rate,          # CFG dropout rate
    drop_last,
    text_padding_length,
    seed
) -> Tuple[Dataset, StatefulDataLoader]
```

#### Parquet Schema
- Located in `/fastvideo/dataset/dataloader/schema.py`
- `pyarrow_schema_t2v`: Text-to-video schema
- Includes: video latents, text embeddings, attention masks

---

### 6. DATA STRUCTURES FOR TRAINING

#### TrainingBatch (pipeline_batch_info.py, line 203)
```python
# Data from dataloader
latents: torch.Tensor
encoder_hidden_states: torch.Tensor    # Text embeddings
encoder_attention_mask: torch.Tensor
image_latents: torch.Tensor            # For I2V

# Transformer inputs (prepared)
noisy_model_input: torch.Tensor
timesteps: torch.Tensor
sigmas: torch.Tensor
noise: torch.Tensor

# Attention metadata
attn_metadata_vsa: AttentionMetadata
attn_metadata: AttentionMetadata

# Training outputs
loss: torch.Tensor
total_loss: float
grad_norm: float

# Distillation-specific
conditional_dict, unconditional_dict: dict
generator_loss, fake_score_loss: float
```

#### ForwardBatch (pipeline_batch_info.py, line 62)
Used in inference pipelines, contains embeddings, latents, generation parameters.

---

### 7. EXISTING RL/POST-TRAINING CAPABILITIES

#### Current Advanced Training Methods:

1. **Distillation Pipeline** (DMD - Diffusion Model Distillation)
   - Two-model approach: real_score_transformer (teacher) + fake_score_transformer (trainable)
   - Uses score-based guidance for training
   - Supports flow-matching with timestep shifting

2. **Self-Forcing Distillation** (SelfForcingDistillationPipeline)
   - Alternates between generator and critic training
   - Ratio-based updates: `dfake_gen_update_ratio` (default: 5)
   - Multi-block training with cache management
   - Uses critical frame sampling for efficiency

3. **Existing Score/Reward Components**
   - `real_score_guidance_scale`: 3.5 (default)
   - Score networks as separate transformers
   - Loss computation based on score predictions
   - No explicit reward models yet (RL-specific)

#### What's NOT Yet Implemented
- Explicit reward/value models
- Policy gradient methods (PPO, GRPO)
- Trajectory sampling & rollouts
- Advantage estimation
- Preference learning (RLHF-style)

---

### 8. DISTRIBUTED TRAINING SETUP

#### Parallelism Strategies
```python
# From fastvideo_args.py
tp_size: int          # Tensor parallelism
sp_size: int          # Sequence parallelism
hsdp_replicate_dim: int    # Data parallelism replicate
hsdp_shard_dim: int        # Data parallelism shard
num_gpus: int
```

#### Distributed Utilities (`/fastvideo/distributed/`)
- Parallel state management
- Communication groups (world, sp_group, dp_group)
- FSDP (Fully Sharded Data Parallel) integration
- Checkpoint distribution

#### Key Functions
```python
get_sp_group()          # Sequence parallel group
get_world_group()       # All ranks
get_sp_world_size()
get_world_size()
shard_latents_across_sp()  # Sequence sharding
```

---

### 9. CHECKPOINTING & STATE MANAGEMENT

#### Checkpoint Utilities (`/fastvideo/training/checkpointing_utils.py`)
```python
class ModelWrapper(Stateful)          # Model state dict
class OptimizerWrapper(Stateful)      # Optimizer state
class SchedulerWrapper(Stateful)      # LR scheduler
class RandomStateWrapper(Stateful)    # RNG state
```

#### Checkpoint Functions (training_utils.py)
```python
save_checkpoint(
    model, optimizer, scheduler, 
    dataloader, epoch, step, output_dir
)

load_checkpoint(
    model, rank, checkpoint_dir, 
    optimizer, dataloader, scheduler
)

save_distillation_checkpoint(...)  # For multi-model training
load_distillation_checkpoint(...)
```

#### Stateful DataLoader
- Uses `torchdata.stateful_dataloader.StatefulDataLoader`
- Resumable from arbitrary step
- Supports gradient accumulation synchronization

---

### 10. LOGGING & MONITORING

#### Tracker System (`/fastvideo/training/trackers.py`)
```python
class Trackers(Enum):
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"

initialize_trackers(
    tracker_names: list[str],
    experiment_name: str,
    config: dict,
    log_dir: str,
    run_name: str
) -> TrackerType
```

#### Key Logged Metrics
- Step-wise loss
- Gradient norm
- Step time
- Learning rate
- Validation metrics (if enabled)

---

### 11. MAIN ENTRY POINTS

#### CLI Entry Points (`/fastvideo/entrypoints/`)

**Main CLI**: `/fastvideo/entrypoints/cli/main.py`
- Subparser-based commands
- Generates frames/videos

**Training Launch** (from WanTrainingPipeline):
```python
from fastvideo.training.wan_training_pipeline import WanTrainingPipeline
pipeline = WanTrainingPipeline.from_pretrained(
    model_path, args=training_args
)
pipeline.train()
```

#### Example Usage Pattern
```bash
python -m fastvideo.training.wan_training_pipeline \
  --model-path model_weights \
  --data-path training_data.parquet \
  --output-dir ./checkpoints \
  --max-train-steps 10000 \
  --learning-rate 1e-4
```

---

### 12. VALIDATION & EVALUATION

#### Validation Pipeline
```python
def initialize_validation_pipeline(training_args: TrainingArgs):
    # Creates inference-mode pipeline (same architecture)
    # Uses same trained transformer weights
    # Generates videos at intervals for quality checks
```

#### ValidationDataset
- Supports multiple formats: CSV, JSON, Parquet, Arrow
- Distributed validation (DP-aware sampling)
- Can return: images, text prompts, video paths

#### Validation Sampling
```python
validation_steps: float         # How often to validate
validation_sampling_steps: str  # Denoising steps for validation
validation_guidance_scale: str  # Classifier-free guidance
log_validation: bool            # Whether to log outputs
```

---

### 13. KEY UTILITIES & HELPERS

#### Loss & Forward Functions (`training_utils.py`)
```python
normalize_dit_input()           # Input normalization
shard_latents_across_sp()       # Sequence parallel sharding
get_scheduler()                 # LR scheduler factory
compute_density_for_timestep_sampling()  # SD3-style weighting
clip_grad_norm_while_handling_failing_dtensor_cases()
```

#### Model Loading
```python
maybe_download_model()          # HF hub download
verify_model_config_and_directory()
is_vsa_available(), is_vmoba_available()
```

---

### 14. RECOMMENDED STRUCTURE FOR RL/GRPO PIPELINE

Based on codebase analysis, here's the recommended architecture:

#### New Components Needed
```
/fastvideo/training/
├── rl_pipeline.py              # Base RL training pipeline
├── grpo_pipeline.py            # GRPO-specific implementation
├── reward_models.py            # Reward/value model wrappers
├── rl_utils.py                 # RL-specific utilities
├── trajectory_sampling.py       # Rollout & trajectory collection
├── advantage_estimation.py      # GAE, returns computation
└── rl_trackers.py              # RL-specific logging

/fastvideo/configs/training/
├── rl_config.py                # RL-specific training config
└── reward_config.py            # Reward model config
```

#### Integration Points
1. **Extends TrainingPipeline**: Share optimizer, scheduler, checkpointing
2. **Uses TrainingBatch**: Add reward predictions, advantages, log-probs
3. **Training loop**: Similar to SelfForcingDistillationPipeline (alternating updates)
4. **Data loading**: Reuse existing parquet datasets
5. **Validation**: Generate samples with both old & new policy
6. **Checkpointing**: Save policy & reward model separately

#### Key Methods to Implement
```python
class RLPipeline(TrainingPipeline):
    def sample_trajectory()        # Collect rollouts
    def compute_rewards()          # Get reward scores
    def compute_advantages()       # GAE computation
    def compute_policy_loss()      # PPO/GRPO loss
    def compute_value_loss()       # Reward model loss
    def train_one_step()           # Override with RL logic
```

---

### 15. DEPENDENCY & LIBRARY ECOSYSTEM

Key libraries used:
- **PyTorch**: Main ML framework (DDP, FSDP, FSDP2)
- **Diffusers**: Scheduler implementations
- **Hugging Face**: Model hub, transformers
- **Einops**: Tensor reshaping
- **Torchdata**: Stateful data loading
- **PyArrow/Parquet**: Data format
- **Weights & Biases**: Experiment tracking
- **Tensorboard**: Alternative logging

---

### 16. CODEBASE STATISTICS

| Component | Files | Total LOC |
|-----------|-------|-----------|
| Training | 15+ | ~7,936 |
| Models | 6+ DITs | ~170,000+ |
| Pipelines | 15+ | ~50,000+ |
| Configs | 20+ | ~5,000+ |
| Dataset | 8 | ~2,000+ |
| Distributed | 5+ | ~2,000+ |

---

### 17. CRITICAL NOTES FOR RL IMPLEMENTATION

1. **Distributed Training**: FSDP2 with sequence parallelism is critical for large models
2. **Gradient Accumulation**: Already implemented, useful for RL
3. **Checkpointing**: Comprehensive system - leverage for RL state
4. **Data Format**: All training uses parquet with preprocessed latents/embeddings
5. **Scheduler**: Flow-matching specific - may need adaptation for RL
6. **VSA/Attention**: Sparsity scheduling available for efficiency
7. **Batch Structure**: TrainingBatch is extensible for advantage/reward data
8. **Validation**: Validation pipeline pattern exists - can be extended for policy comparison

