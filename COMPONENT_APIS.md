# FastVideo Component APIs & Interactions

## TRAINING PIPELINE API

### TrainingPipeline (Abstract Base Class)
File: `/fastvideo/training/training_pipeline.py`

```python
class TrainingPipeline(LoRAPipeline, ABC):
    
    # Initialization
    def __init__(
        self,
        model_path: str,
        fastvideo_args: TrainingArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None
    ) -> None:
        """Initialize training pipeline with model and args."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        args: TrainingArgs,
        **kwargs
    ) -> "TrainingPipeline":
        """Load model from pretrained weights."""
    
    # Setup (must implement in subclass)
    @abstractmethod
    def initialize_validation_pipeline(
        self, 
        training_args: TrainingArgs
    ):
        """Create validation pipeline."""
    
    # Core training methods
    def initialize_training_pipeline(
        self,
        training_args: TrainingArgs
    ):
        """Initialize optimizer, scheduler, dataloader, tracker."""
        # Sets: self.optimizer, self.lr_scheduler, self.train_dataloader
        # Sets: self.transformer, self.noise_scheduler
        # Initializes: self.tracker (logging)
    
    def train_one_step(
        self,
        training_batch: TrainingBatch
    ) -> TrainingBatch:
        """Execute single training step with gradient accumulation."""
        # Returns: updated TrainingBatch with loss and grad_norm
    
    def train(self) -> None:
        """Main training loop for max_train_steps."""
        # Handles: checkpointing, validation, logging
```

---

## TRAINING BATCH API

File: `/fastvideo/pipelines/pipeline_batch_info.py`

```python
@dataclass
class TrainingBatch:
    
    # Step tracking
    current_timestep: int = 0
    current_vsa_sparsity: float = 0.0
    
    # Data inputs (from dataloader)
    latents: torch.Tensor | None = None
    raw_latent_shape: torch.Tensor | None = None
    noise_latents: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None      # [B, T, D]
    encoder_attention_mask: torch.Tensor | None = None     # [B, T]
    preprocessed_image: torch.Tensor | None = None         # For I2V
    image_embeds: torch.Tensor | None = None
    image_latents: torch.Tensor | None = None
    
    # Prepared inputs
    noisy_model_input: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None                  # [B]
    sigmas: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    
    # Attention (distributed)
    attn_metadata_vsa: AttentionMetadata | None = None     # VSA metadata
    attn_metadata: AttentionMetadata | None = None         # Full attention
    
    # Loss computation inputs
    input_kwargs: dict[str, Any] | None = None
    
    # Outputs
    loss: torch.Tensor | None = None
    total_loss: float | None = None
    grad_norm: float | None = None
    
    # For distillation
    conditional_dict: dict[str, Any] | None = None
    unconditional_dict: dict[str, Any] | None = None
    generator_loss: float = 0.0
    fake_score_loss: float = 0.0
```

---

## TRAINING ARGS API

File: `/fastvideo/fastvideo_args.py`

```python
@dataclasses.dataclass
class TrainingArgs(FastVideoArgs):
    
    # === DATA ===
    data_path: str = ""                           # Path to training data
    dataloader_num_workers: int = 0              # Num workers for dataloader
    training_cfg_rate: float = 0.0               # CFG dropout probability
    
    # === MODEL DIMENSIONS ===
    num_height: int = 0                          # Latent height
    num_width: int = 0                           # Latent width
    num_frames: int = 0                          # Number of frames
    num_latent_t: int = 0                        # Sequence parallel split
    train_batch_size: int = 0                    # Global batch size
    train_sp_batch_size: int = 0                 # Per-GPU batch size
    
    # === OPTIMIZATION ===
    learning_rate: float = 0.0
    max_train_steps: int = 0
    gradient_accumulation_steps: int = 0         # For gradient accumulation
    weight_decay: float = 0.0
    betas: str = "0.9,0.999"                     # Adam betas
    max_grad_norm: float = 0.0                   # Gradient clipping
    
    # === LR SCHEDULING ===
    lr_scheduler: str = "constant"               # Scheduler type
    lr_warmup_steps: int = 0                     # Warmup steps
    lr_num_cycles: int = 0                       # For cosine scheduler
    lr_power: float = 0.0                        # For polynomial scheduler
    min_lr_ratio: float = 0.5                    # For cosine_with_min_lr
    
    # === VALIDATION ===
    validation_dataset_file: str = ""            # Val prompts/dataset
    validation_sampling_steps: str = ""          # Denoising steps
    validation_guidance_scale: str = ""          # CFG scale
    validation_steps: float = 0.0                # How often to validate
    log_validation: bool = False                 # Save validation outputs
    
    # === LOGGING ===
    trackers: list[str] = field(default_factory=list)      # ["wandb", ...]
    tracker_project_name: str = ""
    wandb_run_name: str = ""
    output_dir: str = ""
    
    # === CHECKPOINTING ===
    resume_from_checkpoint: str = ""             # Checkpoint dir to resume
    checkpoints_total_limit: int = 0             # Keep last N checkpoints
    
    # === DISTILLATION ===
    real_score_model_path: str = ""              # Teacher model
    fake_score_model_path: str = ""              # Critic/value model
    generator_update_interval: int = 5
    dfake_gen_update_ratio: int = 5              # Gen:critic update ratio
    min_timestep_ratio: float = 0.2
    max_timestep_ratio: float = 0.98
    real_score_guidance_scale: float = 3.5       # Score guidance strength
    fake_score_learning_rate: float = 0.0
    
    # === SELF-FORCING ===
    num_frame_per_block: int = 3                 # Frames per causal block
    independent_first_frame: bool = False
    enable_gradient_masking: bool = True
    gradient_mask_last_n_frames: int = 21
    same_step_across_blocks: bool = False
    last_step_only: bool = False
    context_noise: int = 0
    
    # === EFFICIENCY ===
    enable_gradient_checkpointing_type: str | None = None
    VSA_decay_rate: float = 0.01                 # Sparse attn decay
    VSA_decay_interval_steps: int = 1
    
    # === LORA ===
    lora_rank: int | None = None
    lora_training: bool = False
    
    # === OTHER ===
    seed: int | None = None
    mixed_precision: str = ""                    # "fp16", "bf16", ""
    master_weight_type: str = ""
```

---

## DATA LOADING API

File: `/fastvideo/dataset/__init__.py`

```python
def build_parquet_map_style_dataloader(
    data_path: str,                              # Path to .parquet files
    batch_size: int,                             # Global batch size
    parquet_schema: pyarrow.Schema | None = None,
    num_data_workers: int = 0,                   # DataLoader workers
    cfg_rate: float = 0.0,                       # CFG dropout
    drop_last: bool = True,
    text_padding_length: int = 512,              # Max text length
    seed: int | None = None,
) -> Tuple[
    VideoCaptionMergedDataset,
    StatefulDataLoader[dict]
]:
    """Build dataloader for training.
    
    Returns:
        dataset: The dataset (for state tracking)
        dataloader: Stateful dataloader yielding dicts with:
            - video_latent: [B, C, T, H, W]
            - prompt_embeds: [B, T_text, D]
            - attention_mask: [B, T_text]
    """
    
# Parquet schema (from dataset/dataloader/schema.py)
pyarrow_schema_t2v = pyarrow.schema([
    ("video_latent", ...) ,     # Video latents
    ("prompt_embeds", ...),     # Text embeddings
    ("attention_mask", ...),    # Text attention mask
])
```

---

## SCHEDULER API

File: `/fastvideo/training/training_utils.py`

```python
def get_scheduler(
    scheduler_name: str,                         # "constant", "linear", "cosine", etc.
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,                         # For cosine
    power: float = 1.0,                          # For polynomial
    min_lr_ratio: float = 0.0,                   # For cosine_with_min_lr
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler."""
```

---

## CHECKPOINTING API

File: `/fastvideo/training/training_utils.py`

```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: StatefulDataLoader,
    epoch: int,
    step: int,
    output_dir: str,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Save training checkpoint (model, optimizer, scheduler, dataloader state)."""

def load_checkpoint(
    model: torch.nn.Module,
    rank: int,
    checkpoint_dir: str,
    optimizer: torch.optim.Optimizer | None = None,
    dataloader: StatefulDataLoader | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    noise_generator: torch.Generator | None = None,
) -> int:
    """Load checkpoint. Returns resumed_step."""
```

---

## LOGGING/TRACKER API

File: `/fastvideo/training/trackers.py`

```python
class TrackerType(Protocol):
    
    def log(self, data: dict[str, Any]) -> None:
        """Log scalar/metric data."""
        # data = {"train/loss": 0.5, "train/lr": 1e-4}
    
    def save_image(
        self,
        name: str,
        image: PIL.Image.Image | torch.Tensor,
        step: int,
    ) -> None:
        """Log image."""
    
    def save_video(
        self,
        name: str,
        video: torch.Tensor,                      # [T, C, H, W] or [B, T, C, H, W]
        step: int,
        fps: int = 24,
    ) -> None:
        """Log video."""
    
    @contextmanager
    def timed(self, name: str):
        """Context manager for timing blocks."""
        # with tracker.timed("timing/forward_pass"):
        #     model()

def initialize_trackers(
    tracker_names: list[str],                    # ["wandb", "tensorboard"]
    experiment_name: str,
    config: dict[str, Any],
    log_dir: str,
    run_name: str | None = None,
) -> TrackerType:
    """Initialize trackers (W&B, TensorBoard, etc.)."""
```

---

## UTILITY FUNCTIONS

### Input/Output Processing
```python
# From training_utils.py

def normalize_dit_input(
    noisy_latents: torch.Tensor,                 # [B, C, T, H, W]
    timesteps: torch.Tensor,                     # [B]
    encoder_hidden_states: torch.Tensor,         # [B, T_text, D]
    encoder_attention_mask: torch.Tensor,        # [B, T_text]
) -> dict[str, torch.Tensor]:
    """Prepare inputs for DIT model."""

def pred_noise_to_pred_video(
    model_pred: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Convert model predictions to video latents."""

def shard_latents_across_sp(
    latents: torch.Tensor,                       # [B, C, T, H, W]
    num_latent_t: int,                           # Sequence parallel size
) -> torch.Tensor:
    """Shard latents across sequence parallel groups."""
```

### Gradient Management
```python
def clip_grad_norm_while_handling_failing_dtensor_cases(
    model: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    """Clip gradients, handle DTensor edge cases. Returns grad_norm."""

def compute_density_for_timestep_sampling(
    weighting_scheme: str,                       # "logit_normal", "mode"
    batch_size: int,
    generator: torch.Generator,
    logit_mean: float | None = None,
    logit_std: float | None = None,
    mode_scale: float | None = None,
) -> torch.Tensor:
    """Compute timestep sampling density (SD3-style)."""
```

---

## DISTRIBUTED UTILITIES

File: `/fastvideo/distributed/parallel_state.py`

```python
def get_world_group() -> ProcessGroup:
    """Get all ranks in training."""
    # Attributes: rank, world_size, local_rank

def get_sp_group() -> ProcessGroup:
    """Get sequence parallel group."""
    # Attributes: rank_in_group, world_size

def get_sp_world_size() -> int:
    """Get size of sequence parallel group."""

def get_sp_parallel_rank() -> int:
    """Get rank within SP group."""
```

---

## VALIDATION DATASET API

File: `/fastvideo/dataset/validation_dataset.py`

```python
class ValidationDataset(IterableDataset):
    
    def __init__(self, filename: str):
        """Load validation dataset from CSV, JSON, Parquet, or Arrow file.
        
        File format should contain fields:
            - prompt (or similar text field)
            - image_path, video_path (optional)
        """
    
    def __iter__(self):
        """Yields samples as dicts."""
        # Each sample: {"prompt": "...", "image": ..., ...}
```

---

## MODEL REGISTRY API

File: `/fastvideo/models/registry.py`

```python
# Model registry dicts

_TEXT_TO_VIDEO_DIT_MODELS = {
    "WanTransformer3DModel": ("dits", "wanvideo", "WanTransformer3DModel"),
    "HunyuanVideoTransformer3DModel": (...),
    "CausalWanTransformer3DModel": (...),
    "StepVideoModel": (...),
    "CosmosTransformer3DModel": (...),
}

_TEXT_ENCODER_MODELS = {
    "T5EncoderModel": ("encoders", "t5", "T5EncoderModel"),
    "CLIPTextModel": ("encoders", "clip", "CLIPTextModel"),
    # ...
}

_VAE_MODELS = {
    "AutoencoderKLWan": ("vaes", "wanvae", "AutoencoderKLWan"),
    "AutoencoderKLHunyuanVideo": (...),
    # ...
}

_SCHEDULERS = {
    "FlowMatchEulerDiscreteScheduler": (...),
    "SelfForcingFlowMatchScheduler": (...),
    # ...
}
```

---

## PIPELINE CONFIGURATION

File: `/fastvideo/configs/pipelines/base.py`

```python
@dataclass
class PipelineConfig:
    
    # DIT (Diffusion Transformer)
    dit_config: DiTConfig
    
    # VAE
    vae_config: VAEConfig
    vae_tiling: bool = False
    vae_sp: bool = False
    
    # Text encoders (can have multiple for multimodal)
    text_encoder_configs: tuple[EncoderConfig, ...]
    postprocess_text_funcs: tuple[Callable, ...]
    
    # Image encoder (for I2V)
    image_encoder_config: EncoderConfig | None = None
    
    # Scheduler
    flow_shift: float | None = None
    
    # Precision per component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...]
```

---

## INTEGRATION CHECKLIST FOR RL PIPELINE

New files to create:
- [ ] `/fastvideo/training/rl_pipeline.py` - Main RL pipeline class
- [ ] `/fastvideo/training/rl_utils.py` - RL-specific utilities (GAE, policy loss, etc.)
- [ ] `/fastvideo/training/reward_model.py` - Wrapper for reward models
- [ ] `/fastvideo/configs/training/rl_config.py` - RL-specific config class

Files to modify:
- [ ] `/fastvideo/fastvideo_args.py` - Add TrainingArgs.rl_* parameters
- [ ] `/fastvideo/pipelines/pipeline_batch_info.py` - Extend TrainingBatch
- [ ] `/fastvideo/training/__init__.py` - Export new RL classes

Integration points:
1. Inherit from `TrainingPipeline`
2. Use existing: optimizer, scheduler, checkpointing, distributed utilities, trackers
3. Extend: TrainingBatch with reward/advantage fields
4. Reuse: data loading, model loading, validation pipeline pattern
5. Follow: gradient accumulation, distributed training patterns

