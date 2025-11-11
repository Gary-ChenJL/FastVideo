# RL/GRPO Implementation Scope

**Created**: November 11, 2025
**Status**: Active Development

---

## 🎯 Project Scope

This RL/GRPO implementation is **exclusively focused on VIDEO generation models**, not image models.

### ✅ In Scope

1. **Video Generation Models**
   - FastVideo WAN models
   - Video diffusion/flow matching models
   - Text-to-video (T2V) tasks
   - Image-to-video (I2V) tasks

2. **Video-Specific Reward Models**
   - Temporal coherence rewards
   - Motion quality rewards
   - Video-text alignment rewards
   - Frame consistency rewards
   - Video aesthetic quality
   - Action/event understanding

3. **Video-Specific Validation**
   - Video quality metrics (FVD, etc.)
   - Temporal metrics
   - Multi-frame evaluation

### ❌ Out of Scope

1. **Image Generation Models**
   - Stable Diffusion 3.5 (image-only)
   - FLUX.1 (image-only)
   - Any text-to-image (T2I) models from flow_grpo

2. **Image-Only Reward Models**
   - PickScore (image aesthetic quality)
   - ImageReward
   - CLIP Score (single frame)
   - Any reward model that operates on individual frames only

3. **Image-Specific Examples**
   - No porting of SD3.5/FLUX examples from flow_grpo
   - No image generation validation

---

## 📋 Reward Model Strategy

### Video Reward Models to Implement

**Priority 1: Core Video Rewards**
1. **VideoScore** - Video aesthetic quality (multi-frame)
2. **VideoTextAlignment** - CLIP-based video-text similarity
3. **TemporalCoherence** - Frame-to-frame consistency
4. **MotionQuality** - Motion smoothness and realism

**Priority 2: Task-Specific Rewards**
5. **ActionRecognition** - Action/event understanding
6. **VideoOCR** - Text rendering quality in video (if applicable)
7. **HumanPreferenceVideo** - Video-specific human feedback

**Priority 3: Advanced Rewards**
8. **SemanticConsistency** - Object/scene consistency across frames
9. **CameraMotion** - Desired camera movement patterns
10. **VideoFVD** - Fréchet Video Distance for quality

### Reward Models NOT Being Ported

These image-only rewards from flow_grpo will **NOT** be implemented:
- ❌ PickScore (image-only aesthetic)
- ❌ ImageReward (image-only quality)
- ❌ GenEval (image-only compositional understanding)
- ❌ Single-frame CLIP Score
- ❌ Aesthetic Predictor v2 (image-only)

### Adaptation Strategy

For reward models that exist in both image and video forms:
- Use **video-specific versions** (e.g., CLIP video embeddings, not image)
- Aggregate over **temporal dimension** (not just spatial)
- Consider **motion and temporal coherence** as first-class metrics

---

## 🏗️ Architecture Implications

### Data Flow

```
Input: Text Prompt
  ↓
Policy Model (Video Transformer)
  ↓
Generated Video Latents [B, C, T, H, W]
  ↓
VAE Decode → Video Frames [B, T, C, H, W]
  ↓
Video Reward Models → Rewards [B]
  ↓
RL Training (GRPO)
```

### Key Differences from Image RL

1. **Temporal Dimension**: All operations must handle T (time) dimension
2. **Memory**: Videos require significantly more memory than images
3. **Rewards**: Must evaluate across multiple frames, not single frames
4. **Validation**: Generate and evaluate full video sequences

---

## 📊 Implementation Phases

### Phase 1: Foundation ✅ COMPLETE
- Core RL infrastructure
- TrainingBatch extensions
- RL parameters in TrainingArgs
- Base reward model interface (video-aware)
- RL utilities (GAE, GRPO loss)
- RLPipeline skeleton

### Phase 2: Video Reward Models 🔄 NEXT
- Implement VideoScore reward
- Implement VideoTextAlignment reward
- Implement TemporalCoherence reward
- Update MultiRewardAggregator for video
- Add video decoding pipeline

### Phase 3: RL Training Loop
- Trajectory collection with video rollouts
- Log probability extraction from video transformer
- Value model forward pass (video-aware)
- Complete train_one_step() implementation

### Phase 4: Validation & Logging
- Video generation validation
- FVD and video quality metrics
- Video logging to W&B/TensorBoard
- RL-specific video visualizations

### Phase 5: Optimization
- Distributed video rollouts
- Memory optimization for video
- Video caching strategies
- Gradient checkpointing for video sequences

---

## 🚫 What We're NOT Doing

To be absolutely clear, we will **NOT**:

1. Port any Stable Diffusion 3.5 examples from flow_grpo
2. Port any FLUX.1 examples from flow_grpo
3. Implement image-only reward models
4. Support text-to-image tasks
5. Validate on image benchmarks
6. Use single-frame rewards as primary signals

---

## 📝 Documentation Standards

All documentation, code comments, and examples should:
- Explicitly mention "video" when relevant
- Use video-specific terminology (frames, temporal, motion, etc.)
- Show video tensor shapes: `[B, C, T, H, W]` or `[B, T, C, H, W]`
- Reference video metrics (FVD, temporal coherence, etc.)
- Avoid image-only examples or references

---

## 🎬 Example Use Case

**Target Scenario**:
```
User provides: "A cat jumping over a fence"
Model generates: 2-second video (17 frames)
Rewards evaluate:
  - VideoTextAlignment: Does video match "cat jumping over fence"?
  - TemporalCoherence: Are frames smooth and consistent?
  - MotionQuality: Is the jumping motion realistic?
  - VideoScore: Overall video aesthetic quality?
Aggregated reward → Train policy with GRPO
```

**NOT Our Target**:
```
User provides: "A cat sitting on a fence"
Model generates: Single image
Rewards evaluate: PickScore, ImageReward, etc.
```

---

## ✅ Acceptance Criteria

Before considering the RL pipeline complete, it must:

1. ✅ Train on video generation models (FastVideo WAN)
2. ✅ Use video-specific reward models (temporal awareness)
3. ✅ Generate and validate full video sequences
4. ✅ Measure video quality metrics (FVD, etc.)
5. ✅ Handle temporal dimension throughout pipeline
6. ❌ NOT support image-only models
7. ❌ NOT use image-only reward models

---

## 🔗 References

**Video-Specific Resources**:
- FastVideo: Video generation architecture
- FVD (Fréchet Video Distance): Video quality metric
- CLIP Video Embeddings: Video-text alignment
- UCF-101, Kinetics: Video understanding datasets

**NOT Using** (Image-Only):
- flow_grpo SD3.5/FLUX examples
- PickScore, ImageReward
- Image-only CLIP embeddings
- COCO, ImageNet benchmarks

---

## 📧 Questions?

If there's any ambiguity about scope:
- **Default to VIDEO**: When in doubt, implement the video version
- **Temporal awareness**: If a component doesn't handle time dimension, it's probably wrong
- **Skip image examples**: Don't port image-only code from flow_grpo

---

**Last Updated**: November 11, 2025
**Scope Owner**: @Gary-ChenJL
