# RL Parameter Compatibility Guide

**Created**: November 11, 2025
**Purpose**: Document which RL parameters are universal vs algorithm-specific

---

## 🎯 Current Parameters: Universal vs Algorithm-Specific

### ✅ **Universal Parameters** (PPO, GRPO, A2C, TRPO, etc.)

These work across most policy gradient RL algorithms:

| Parameter | PPO | GRPO | A2C | TRPO | Notes |
|-----------|-----|------|-----|------|-------|
| `rl_gamma` | ✅ | ✅ | ✅ | ✅ | Discount factor (universal) |
| `rl_lambda` | ✅ | ✅ | ✅ | ✅ | GAE lambda (Schulman et al.) |
| `rl_use_gae` | ✅ | ✅ | ✅ | ✅ | Generalized Advantage Estimation |
| `rl_normalize_advantages` | ✅ | ✅ | ✅ | ✅ | Standard practice |
| `rl_policy_clip_range` | ✅ | ✅ | ❌ | ❌ | PPO invented, GRPO adopted |
| `rl_value_clip_range` | ✅ | ✅ | ❌ | ❌ | PPO-style value clipping |
| `rl_target_kl` | ✅ | ✅ | ❌ | ✅ | KL constraint (PPO/TRPO) |
| `rl_entropy_coef` | ✅ | ✅ | ✅ | ✅ | Exploration bonus |
| `rl_value_loss_coef` | ✅ | ✅ | ✅ | ✅ | Value loss weight |
| `rl_num_policy_epochs` | ✅ | ✅ | ⚠️ | ✅ | PPO:3-10, GRPO:1, A2C:1 |
| `rl_warmup_steps` | ✅ | ✅ | ✅ | ✅ | Warm-start strategy |
| `reward_model_*` | ✅ | ✅ | ✅ | ✅ | Any RL can use rewards |

### 🎯 **Flow-GRPO Specific Parameters**

These are unique to Flow-GRPO's innovation (fast training on flow matching models):

| Parameter | Purpose | Why GRPO-specific |
|-----------|---------|-------------------|
| `rl_rollout_steps` | Random intermediate timesteps | Flow-GRPO-Fast innovation |
| `rl_noise_injection_min/max` | Timestep range for injection | Flow matching specific |
| `rl_use_sde_sampling` | SDE sampling strategy | Flow-GRPO-Fast approach |
| `rl_num_denoising_steps: 2` | Train on 1-2 steps only | Flow-GRPO-Fast (not full rollout) |
| `rl_use_grpo_guard` | Safety mechanisms | GRPO-Guard feature |
| `rl_ratio_norm_correction` | RatioNorm correction | GRPO-Guard specific |
| `rl_gradient_reweighting` | Reweight across denoising steps | Diffusion/flow specific |
| `rl_max_importance_ratio` | Clip extreme ratios | GRPO-Guard safety |

### 🔀 **Diffusion/Flow Model Specific** (Not in Standard RL)

These are necessary for diffusion/flow matching models but not in standard RL:

```python
# Standard RL (like Atari/MuJoCo) doesn't have these concepts:
- rl_noise_injection_*       # Diffusion timesteps
- rl_gradient_reweighting     # Across denoising steps
- rl_rollout_steps            # Intermediate sampling
- rl_num_denoising_steps      # Partial trajectories
```

Standard PPO for Atari would just use full environment rollouts, no "partial denoising steps".

---

## 📋 Algorithm Configurations

### PPO (Proximal Policy Optimization) Configuration

```python
# For standard PPO on video generation:
rl_algorithm = "ppo"
rl_gamma = 0.99
rl_lambda = 0.95
rl_use_gae = True
rl_normalize_advantages = True
rl_policy_clip_range = 0.2
rl_value_clip_range = 0.2
rl_num_policy_epochs = 4          # PPO typically 3-10
rl_num_value_epochs = 4
rl_target_kl = 0.01
rl_entropy_coef = 0.01            # Small exploration bonus
rl_value_loss_coef = 0.5

# PPO would do FULL rollouts, not partial denoising:
rl_num_denoising_steps = 50       # Full denoising trajectory
rl_use_sde_sampling = False       # Standard deterministic
rl_use_grpo_guard = False         # Not needed for PPO
rl_ratio_norm_correction = False
rl_gradient_reweighting = False
```

### GRPO (Group Relative Policy Optimization) Configuration

```python
# For Flow-GRPO (current default):
rl_algorithm = "grpo"
rl_gamma = 0.99
rl_lambda = 0.95
rl_use_gae = True
rl_normalize_advantages = True
rl_policy_clip_range = 0.2
rl_value_clip_range = 0.2
rl_num_policy_epochs = 1          # GRPO uses 1 (on-policy)
rl_num_value_epochs = 1
rl_target_kl = 0.01
rl_entropy_coef = 0.0             # GRPO doesn't need entropy bonus
rl_value_loss_coef = 0.5

# Flow-GRPO-Fast specific:
rl_num_denoising_steps = 2        # 1-2 steps only!
rl_rollout_steps = "20,30"        # Random intermediate
rl_noise_injection_min = 10
rl_noise_injection_max = 40
rl_use_sde_sampling = True        # SDE for stochasticity
rl_use_grpo_guard = True          # Safety mechanisms
rl_ratio_norm_correction = True
rl_gradient_reweighting = True
rl_max_importance_ratio = 10.0
```

### A2C (Advantage Actor-Critic) Configuration

```python
# For A2C on video generation:
rl_algorithm = "a2c"
rl_gamma = 0.99
rl_lambda = 0.95
rl_use_gae = True
rl_normalize_advantages = True
rl_policy_clip_range = None       # A2C doesn't clip
rl_value_clip_range = None
rl_num_policy_epochs = 1          # A2C is on-policy, 1 epoch
rl_num_value_epochs = 1
rl_target_kl = None               # A2C doesn't use KL
rl_entropy_coef = 0.01
rl_value_loss_coef = 0.5

# A2C would do full rollouts:
rl_num_denoising_steps = 50       # Full trajectory
rl_use_sde_sampling = False
rl_use_grpo_guard = False
```

---

## 🔧 Suggested Reorganization

To make it clearer which parameters are universal vs specific, here's a better structure:

```python
# ==================================================
# UNIVERSAL RL PARAMETERS (All Algorithms)
# ==================================================

# Core RL settings
rl_mode: bool = False
rl_algorithm: str = "grpo"  # "grpo", "ppo", "a2c", "dpo"

# Advantage estimation (GAE) - Universal
rl_gamma: float = 0.99
rl_lambda: float = 0.95
rl_use_gae: bool = True
rl_normalize_advantages: bool = True

# Policy optimization - Universal concepts
rl_policy_clip_range: float = 0.2      # Used by PPO, GRPO
rl_value_clip_range: float = 0.2       # Used by PPO, GRPO
rl_num_policy_epochs: int = 1          # Varies by algorithm
rl_num_value_epochs: int = 1
rl_target_kl: float = 0.01             # PPO, GRPO, TRPO
rl_entropy_coef: float = 0.0           # Exploration bonus
rl_value_loss_coef: float = 0.5        # Value loss weight

# Reward models - Universal
reward_model_paths: str = ""
reward_weights: str = ""
reward_model_types: str = ""
value_model_path: str = ""
value_model_share_backbone: bool = False

# Training schedule - Universal
rl_warmup_steps: int = 1000
rl_num_rollouts: int = 4
rl_collect_on_policy: bool = True
rl_policy_value_update_ratio: float = 1.0

# ==================================================
# DIFFUSION/FLOW SPECIFIC PARAMETERS
# ==================================================

# Trajectory collection for diffusion/flow models
diffusion_full_trajectory: bool = False     # True for PPO, False for GRPO-Fast
diffusion_num_denoising_steps: int = 2      # Full:50, GRPO-Fast:1-2
diffusion_use_sde_sampling: bool = True

# Flow-GRPO-Fast: Random intermediate sampling
grpo_fast_mode: bool = True                 # Enable Flow-GRPO-Fast
grpo_rollout_steps: str = "20,30"           # Random timesteps
grpo_noise_injection_min: int = 10
grpo_noise_injection_max: int = 40

# GRPO-Guard: Safety mechanisms
grpo_guard_enabled: bool = True
grpo_ratio_norm_correction: bool = True     # RatioNorm
grpo_gradient_reweighting: bool = True      # Across denoising steps
grpo_max_importance_ratio: float = 10.0
```

---

## 🎯 Recommendations

### Option 1: Keep Current Structure (Simpler)

**Pros**:
- Already implemented
- All params in one place
- Easy to understand

**Cons**:
- Mixes universal and specific params
- Naming can be confusing (some "rl_" params are GRPO-specific)

### Option 2: Reorganize with Prefixes (Better)

**Pros**:
- Clear what's universal vs specific
- `rl_*` = universal, `grpo_*` = GRPO-specific, `diffusion_*` = diffusion-specific
- Easier to add new algorithms

**Cons**:
- Requires refactoring Phase 1 code
- More params (though clearer)

### Option 3: Nested Config (Most Flexible)

```python
@dataclass
class RLConfig:
    # Universal RL params
    gamma: float = 0.99
    lambda_: float = 0.95
    # ...

@dataclass
class GRPOConfig:
    # GRPO-specific params
    fast_mode: bool = True
    rollout_steps: str = "20,30"
    # ...

@dataclass
class TrainingArgs:
    rl_config: RLConfig = field(default_factory=RLConfig)
    grpo_config: GRPOConfig = field(default_factory=GRPOConfig)
```

**Pros**:
- Best organization
- Type-safe
- Easy to add new algorithms

**Cons**:
- Most refactoring
- More complex

---

## 💡 My Recommendation

**For now (Phase 1)**: Keep current structure, but add better comments:

```python
# RL/GRPO-specific parameters
rl_mode: bool = False
rl_algorithm: str = "grpo"  # "grpo", "ppo", "a2c"

# === UNIVERSAL RL PARAMETERS (All Algorithms) ===
# Advantage estimation (GAE)
rl_gamma: float = 0.99                    # Used by: ALL
rl_lambda: float = 0.95                   # Used by: ALL
rl_use_gae: bool = True                   # Used by: ALL
rl_normalize_advantages: bool = True      # Used by: ALL

# Policy optimization
rl_policy_clip_range: float = 0.2         # Used by: PPO, GRPO
rl_value_clip_range: float = 0.2          # Used by: PPO, GRPO
rl_num_policy_epochs: int = 1             # Used by: ALL (PPO:3-10, GRPO:1)
rl_num_value_epochs: int = 1              # Used by: ALL
rl_target_kl: float = 0.01                # Used by: PPO, GRPO, TRPO
rl_entropy_coef: float = 0.0              # Used by: ALL
rl_value_loss_coef: float = 0.5           # Used by: ALL

# === FLOW-GRPO SPECIFIC PARAMETERS ===
# Flow-GRPO-Fast: Train on 1-2 denoising steps (not full rollout)
rl_num_denoising_steps: int = 2           # GRPO-Fast:1-2, PPO:50
rl_rollout_steps: str = "20,30"           # GRPO-Fast only
rl_noise_injection_min: int = 10          # GRPO-Fast only
rl_noise_injection_max: int = 40          # GRPO-Fast only
rl_use_sde_sampling: bool = True          # GRPO-Fast only

# GRPO-Guard: Safety mechanisms for GRPO
rl_use_grpo_guard: bool = True            # GRPO only
rl_ratio_norm_correction: bool = True     # GRPO-Guard
rl_gradient_reweighting: bool = True      # GRPO-Guard
rl_max_importance_ratio: float = 10.0     # GRPO-Guard
```

**For Phase 2+**: Consider Option 2 (prefix-based) when adding more algorithms.

---

## 📝 Summary Table

| Category | Parameters | Algorithms | Video-Specific? |
|----------|-----------|------------|-----------------|
| **GAE** | gamma, lambda, normalize | ALL | No |
| **Clipping** | policy_clip, value_clip | PPO, GRPO | No |
| **KL** | target_kl | PPO, GRPO, TRPO | No |
| **Entropy** | entropy_coef | ALL | No |
| **Fast Sampling** | rollout_steps, noise_injection | GRPO | Yes (diffusion) |
| **GRPO-Guard** | ratio_norm, gradient_reweight | GRPO | Yes (diffusion) |
| **Rewards** | reward_model_* | ALL | Yes (video) |

---

**Conclusion**: Current params are **MOSTLY universal with GRPO enhancements**. The GRPO-specific parts are clearly the Fast sampling and GRPO-Guard features. Everything else (GAE, clipping, KL, entropy) works for PPO/A2C/etc.
