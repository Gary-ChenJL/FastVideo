# Asynchronous RL Training Design for FastVideo

**Created**: November 11, 2025
**Status**: Design Proposal
**Target**: Phase 2 Implementation

---

## 🎯 Problem Statement

RL training has expensive sequential operations:
1. **Trajectory sampling** (multiple denoising steps)
2. **VAE decoding** (latents → videos)
3. **Reward computation** (potentially multiple models)

**Current Phase 1 Design**: All operations are **synchronous** → GPU idle time during reward computation.

**Goal**: Overlap sampling, decoding, reward computation, and training to maximize GPU utilization.

---

## 📐 Proposed Architecture

### Component 1: Trajectory Buffer

```python
class TrajectoryBuffer:
    """
    Thread-safe buffer for storing completed rollouts with computed rewards.

    This allows rollout collection to run asynchronously from training updates.
    """

    def __init__(self, max_size: int = 64):
        self.buffer: queue.Queue = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()

    def push(self, rollout: RolloutBatch):
        """Add a completed rollout with rewards to the buffer."""
        self.buffer.put(rollout, block=True)

    def sample(self, batch_size: int) -> list[RolloutBatch]:
        """Sample rollouts for training (blocking if buffer is empty)."""
        rollouts = []
        for _ in range(batch_size):
            rollout = self.buffer.get(block=True)
            rollouts.append(rollout)
        return rollouts

    def is_empty(self) -> bool:
        return self.buffer.empty()

    def size(self) -> int:
        return self.buffer.qsize()
```

### Component 2: Async Rollout Collector

```python
class AsyncRolloutCollector:
    """
    Asynchronous rollout collection in separate thread/process.

    Continuously collects rollouts with current policy and pushes
    to trajectory buffer.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        vae: nn.Module,
        reward_models: MultiRewardAggregator,
        trajectory_buffer: TrajectoryBuffer,
        device: str = "cuda:0"
    ):
        self.policy_model = policy_model
        self.vae = vae
        self.reward_models = reward_models
        self.trajectory_buffer = trajectory_buffer
        self.device = device

        self.running = False
        self.thread = None

    def start(self):
        """Start async rollout collection in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        logger.info("Started async rollout collector")

    def stop(self):
        """Stop async rollout collection."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=10.0)
        logger.info("Stopped async rollout collector")

    def _collection_loop(self):
        """Main loop for collecting rollouts."""
        while self.running:
            try:
                # 1. Sample trajectory with current policy
                with torch.no_grad():
                    rollout = self._sample_trajectory()

                # 2. Decode latents to videos (VAE)
                videos = self._decode_videos(rollout.latents)

                # 3. Compute rewards (potentially multiple models in parallel)
                rewards = self._compute_rewards_parallel(videos, rollout.prompts)

                # 4. Compute values
                values = self._compute_values(rollout)

                # 5. Compute advantages (GAE)
                advantages, returns = compute_gae(
                    rewards=rewards,
                    values=values,
                    next_values=values,  # For single-step
                    gamma=self.gamma,
                    lambda_=self.lambda_
                )

                # 6. Create complete rollout batch
                complete_rollout = RolloutBatch(
                    latents=rollout.latents,
                    prompts=rollout.prompts,
                    log_probs=rollout.log_probs,
                    rewards=rewards,
                    values=values,
                    advantages=advantages,
                    returns=returns
                )

                # 7. Push to buffer (blocks if buffer is full)
                self.trajectory_buffer.push(complete_rollout)

            except Exception as e:
                logger.error(f"Error in rollout collection: {e}")
                if not self.running:
                    break

    def _compute_rewards_parallel(
        self,
        videos: torch.Tensor,
        prompts: list[str]
    ) -> torch.Tensor:
        """
        Compute rewards from multiple models in parallel.

        This is a key optimization: instead of running reward models
        sequentially, we run them concurrently using ThreadPoolExecutor.
        """
        import concurrent.futures

        batch_size = videos.shape[0]

        # Option 1: Parallel across reward models (if multiple models)
        if len(self.reward_models.reward_models) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all reward computations
                futures = []
                for model in self.reward_models.reward_models:
                    future = executor.submit(
                        model.compute_reward,
                        videos,
                        prompts
                    )
                    futures.append(future)

                # Collect results
                individual_rewards = []
                for future in concurrent.futures.as_completed(futures):
                    reward = future.result()
                    individual_rewards.append(reward)

            # Aggregate with weights
            aggregated = sum(
                w * r for w, r in zip(
                    self.reward_models.reward_weights,
                    individual_rewards,
                    strict=False
                )
            )
            return aggregated

        # Option 2: Single reward model (no parallelism needed)
        else:
            return self.reward_models.compute_reward(videos, prompts)
```

### Component 3: Modified RLPipeline with Async Support

```python
class RLPipeline(TrainingPipeline):
    """
    RL pipeline with asynchronous rollout collection.
    """

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        # Initialize async components
        if training_args.rl_async_rollouts:
            self.trajectory_buffer = TrajectoryBuffer(
                max_size=training_args.rl_buffer_size
            )

            self.rollout_collector = AsyncRolloutCollector(
                policy_model=self.transformer,
                vae=self.get_module("vae"),
                reward_models=self.reward_models,
                trajectory_buffer=self.trajectory_buffer,
                device=str(self.device)
            )

            # Prefill buffer before training starts
            logger.info("Prefilling trajectory buffer...")
            self._prefill_buffer(num_rollouts=training_args.rl_buffer_prefill)

            # Start async collection
            self.rollout_collector.start()
            logger.info("Async rollout collection started")
        else:
            self.trajectory_buffer = None
            self.rollout_collector = None
            logger.info("Using synchronous rollout collection")

    def _prefill_buffer(self, num_rollouts: int):
        """
        Prefill trajectory buffer before starting training.

        This ensures we have rollouts ready when training starts.
        """
        with torch.no_grad():
            for _ in tqdm(range(num_rollouts), desc="Prefilling buffer"):
                # Synchronously collect initial rollouts
                rollout = self._collect_single_rollout_sync()
                self.trajectory_buffer.push(rollout)

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Train one step using async or sync rollout collection.
        """
        if self.training_args.rl_async_rollouts:
            return self._train_one_step_async(training_batch)
        else:
            return self._train_one_step_sync(training_batch)

    def _train_one_step_async(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Async training: Get rollouts from buffer (non-blocking for GPU).
        """
        training_batch = self._prepare_training(training_batch)

        for _ in range(self.training_args.gradient_accumulation_steps):
            # Get pre-computed rollout from buffer
            # This is FAST - no sampling/reward computation here!
            rollouts = self.trajectory_buffer.sample(batch_size=1)
            rollout = rollouts[0]

            # Directly use pre-computed values
            training_batch.latents = rollout.latents
            training_batch.log_probs = rollout.log_probs
            training_batch.old_log_probs = rollout.log_probs.clone()
            training_batch.reward_scores = rollout.rewards
            training_batch.values = rollout.values
            training_batch.old_values = rollout.values.clone()
            training_batch.advantages = rollout.advantages
            training_batch.returns = rollout.returns

            # Compute policy loss (GRPO)
            policy_loss, policy_info = compute_grpo_policy_loss(
                log_probs=training_batch.log_probs,
                old_log_probs=training_batch.old_log_probs,
                advantages=training_batch.advantages,
                clip_range=self.training_args.rl_policy_clip_range,
                use_ratio_norm=self.training_args.rl_ratio_norm_correction,
                max_importance_ratio=self.training_args.rl_max_importance_ratio
            )

            training_batch.policy_loss = policy_info["policy_loss"]
            training_batch.kl_divergence = policy_info["kl_divergence"]

            # Backward
            (policy_loss / self.training_args.gradient_accumulation_steps).backward()

            # Compute value loss
            value_loss, value_info = compute_value_loss(
                values=training_batch.values,
                returns=training_batch.returns,
                old_values=training_batch.old_values,
                clip_range=self.training_args.rl_value_clip_range
            )

            training_batch.value_loss = value_info["value_loss"]

            # Backward
            value_loss_scaled = value_loss * self.training_args.rl_value_loss_coef
            (value_loss_scaled / self.training_args.gradient_accumulation_steps).backward()

            training_batch.total_loss += (
                training_batch.policy_loss + training_batch.value_loss
            )

        # Update policy parameters
        # NOTE: Rollout collector will use updated policy for next rollouts
        training_batch = self._clip_grad_norm(training_batch)

        with self.tracker.timed("timing/optimizer_step"):
            self.optimizer.step()
            self.lr_scheduler.step()

            if self.value_optimizer is not None:
                self.value_optimizer.step()
                self.value_scheduler.step()

        return training_batch

    def _train_one_step_sync(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Synchronous training (current Phase 1 implementation).
        """
        # Use existing synchronous implementation
        return super().train_one_step(training_batch)
```

---

## 🚀 Distributed Async Rollouts

For multi-GPU training, we can parallelize rollout collection across data parallel ranks:

```python
class DistributedAsyncRolloutCollector:
    """
    Distributed rollout collection across multiple GPUs.

    Each data parallel rank runs its own rollout collector.
    Rollouts are collected in parallel and synchronized.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        vae: nn.Module,
        reward_models: MultiRewardAggregator,
        trajectory_buffer: TrajectoryBuffer,
        world_size: int,
        rank: int
    ):
        self.policy_model = policy_model
        self.vae = vae
        self.reward_models = reward_models
        self.trajectory_buffer = trajectory_buffer
        self.world_size = world_size
        self.rank = rank

        # Each rank has its own local buffer
        self.local_buffer = TrajectoryBuffer(max_size=32)

        # Start local collector
        self.local_collector = AsyncRolloutCollector(
            policy_model=policy_model,
            vae=vae,
            reward_models=reward_models,
            trajectory_buffer=self.local_buffer,
            device=f"cuda:{rank}"
        )

    def start(self):
        """Start distributed rollout collection."""
        self.local_collector.start()

        # Background thread to sync rollouts across ranks
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True
        )
        self.sync_thread.start()

    def _sync_loop(self):
        """
        Periodically sync rollouts across ranks.

        This ensures all ranks have access to rollouts from all other ranks,
        increasing diversity.
        """
        while self.local_collector.running:
            try:
                # Get rollout from local buffer
                if not self.local_buffer.is_empty():
                    rollout = self.local_buffer.sample(1)[0]

                    # Optionally: broadcast to other ranks
                    # (or just use local rollouts - simpler)

                    # Push to main buffer
                    self.trajectory_buffer.push(rollout)

                time.sleep(0.1)  # Check every 100ms

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
```

---

## 🔧 Configuration Parameters

Add to `TrainingArgs`:

```python
# Async rollout collection
rl_async_rollouts: bool = True  # Enable async rollout collection
rl_buffer_size: int = 64  # Trajectory buffer size
rl_buffer_prefill: int = 32  # Rollouts to collect before training starts
rl_num_collector_threads: int = 1  # Number of parallel collector threads
rl_parallel_reward_models: bool = True  # Compute rewards in parallel

# Distributed rollouts
rl_distributed_rollouts: bool = True  # Collect rollouts on all data parallel ranks
rl_sync_rollouts_interval: int = 10  # Sync interval for distributed rollouts
```

---

## 📊 Performance Benefits

**Estimated Speedup**:
- **Synchronous** (Phase 1):
  - Training: 100ms
  - Sampling: 500ms (5x training time!)
  - VAE decode: 200ms
  - Rewards: 200ms
  - **Total: 1000ms/step**

- **Asynchronous** (Phase 2):
  - Training: 100ms (GPU active)
  - Sampling/VAE/Rewards: Overlapped with training
  - **Total: ~150ms/step** (limited by buffer sync overhead)
  - **~6-7x speedup!**

**Memory Overhead**:
- Trajectory buffer: ~64 rollouts × latent size
- For video latents [B=1, C=16, T=21, H=64, W=64]:
  - Size per rollout: ~7MB (bf16)
  - Buffer total: ~450MB (acceptable)

---

## 🎯 Implementation Phases

### Phase 2A: Basic Async (Week 1)
- ✅ Implement `TrajectoryBuffer`
- ✅ Implement `AsyncRolloutCollector` (single thread)
- ✅ Modify `RLPipeline` to support async mode
- ✅ Add configuration flags

### Phase 2B: Parallel Rewards (Week 2)
- ✅ Parallel reward model computation (ThreadPoolExecutor)
- ✅ Optimize reward model device placement (CPU/GPU split)

### Phase 2C: Distributed Rollouts (Week 3)
- ✅ `DistributedAsyncRolloutCollector`
- ✅ Cross-rank rollout synchronization
- ✅ Load balancing across GPUs

### Phase 2D: Advanced Optimizations (Week 4)
- ✅ Prioritized experience replay (optional)
- ✅ Adaptive buffer sizing
- ✅ Multi-process rollout collection (instead of threading)

---

## 🔍 Comparison with flow_grpo

**flow_grpo approach**:
- Uses Ray for distributed rollouts
- Separate actor processes for sampling
- Central learner process for training
- Async by default

**FastVideo approach** (proposed):
- Uses threading/multiprocessing (simpler than Ray)
- Integrates with existing FSDP/distributed training
- Optional async (can fall back to sync)
- Leverages existing infrastructure (StatefulDataLoader, sequence parallel)

**Why not use Ray?**
- FastVideo doesn't currently use Ray
- Threading is simpler for initial implementation
- Can add Ray later if needed
- FSDP + sequence parallel already provides good distributed training

---

## 🧪 Testing Strategy

### Unit Tests
```python
def test_trajectory_buffer():
    buffer = TrajectoryBuffer(max_size=10)
    # Test push/sample/size operations

def test_async_collector():
    collector = AsyncRolloutCollector(...)
    collector.start()
    # Verify rollouts are collected
    collector.stop()

def test_parallel_reward_computation():
    # Verify rewards computed in parallel are equivalent to sequential
```

### Integration Tests
```python
def test_async_training_convergence():
    # Train with async rollouts
    # Verify convergence matches sync training

def test_distributed_rollouts():
    # Test with multiple GPUs
    # Verify rollouts synced correctly
```

---

## 🚨 Potential Issues & Solutions

### Issue 1: Stale Policy
**Problem**: Rollout collector uses slightly old policy parameters.

**Solution**:
- Acceptable for off-policy algorithms
- For strict on-policy (PPO), limit buffer size to 1-2 rollouts
- Add policy version tracking and discard too-old rollouts

### Issue 2: GIL in Python Threading
**Problem**: Python GIL limits threading parallelism.

**Solution**:
- Use `multiprocessing` instead of `threading` for collector
- Keep reward computation in C++ extensions (no GIL)
- VAE/model inference releases GIL (PyTorch ops)

### Issue 3: Memory Pressure
**Problem**: Large buffer can OOM for video.

**Solution**:
- Adaptive buffer sizing based on available memory
- Store compressed latents (not decoded videos)
- Only decode when computing rewards

### Issue 4: Synchronization Overhead
**Problem**: Frequent buffer access requires locks.

**Solution**:
- Use lock-free queue (e.g., `multiprocessing.Queue`)
- Batch buffer operations
- Minimize sync points

---

## 📚 References

- **flow_grpo**: Uses Ray for distributed async rollouts
- **CleanRL**: Simple async RL implementations
- **Stable-Baselines3**: VecEnv for parallel environments
- **RLlib**: Ray-based distributed RL framework

---

**Status**: Design Complete, Ready for Implementation in Phase 2
