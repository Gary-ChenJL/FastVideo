# Adding a new model to FastVideo

This guide explains how to integrate a new model into FastVideo with minimal steps. It covers the three most common component types used by the pipelines: DiT transformers (text-to-video or image-to-video), encoders (text/image), and VAEs. You will:

- Create the model/config classes
- Register the model so loaders and pipelines can resolve it
- Verify loading from a Diffusers/Transformers checkpoint

Before starting, skim these existing implementations for concrete patterns:
- DiT models: fastvideo/models/dits/{wanvideo.py,hunyuanvideo.py,stepvideo.py}
- Encoders: fastvideo/models/encoders/{clip.py,stepllm.py,llama.py}
- VAEs: fastvideo/models/vaes/{wanvae.py,hunyuanvae.py,stepvideovae.py}
- Registry: fastvideo/models/registry.py
- Component loader: fastvideo/models/loader/component_loader.py
- Model configs: fastvideo/configs/models/**

## 1) Decide the component type

- DiT (transformer3D for video diffusion): subclass BaseDiT or CachableDiT
- Text encoder: subclass TextEncoder
- Image encoder: subclass ImageEncoder
- VAE: implement an nn.Module compatible with VAEConfig

Your upstream checkpoint format determines whether the loader is Transformers or Diffusers:
- Diffusers components must have a _class_name field in the exported config.json
- Transformers components are typically resolved by architectures in their config

## 2) Create config classes

All models in FastVideo have a pair of configs:
- ArchConfig: immutable architecture description (sharding, attention, param mapping)
- ModelConfig: user-facing, references the ArchConfig and adds runtime flags

Examples to follow:
- DiT: fastvideo/configs/models/dits/{wanvideo.py, hunyuanvideo.py, stepvideo.py}
- Encoders: fastvideo/configs/models/encoders/{clip.py, t5.py, llama.py}
- VAE: fastvideo/configs/models/vaes/{wanvae.py, hunyuanvae.py, stepvideovae.py}

Minimal steps:
- Add NewXxxArchConfig(…ArchConfig)
  - Fill in: hidden_size, num_attention_heads, num_channels_latents (for DiT)
  - If you need custom weight-name mapping or stacked weights, set param_names_mapping, stacked_params_mapping, etc.
- Add NewXxxConfig(…Config)
  - Provide default arch_config = NewXxxArchConfig()
  - Optionally add CLI args by overriding add_cli_args if needed

## 3) Implement the model class

- For DiT, subclass CachableDiT (recommended). You must:
  - Define required class variables on the class: _fsdp_shard_conditions, _compile_conditions, param_names_mapping, reverse_param_names_mapping, lora_param_names_mapping, and set _supported_attention_backends
  - In __init__(self, config: DiTConfig, hf_config: dict[str, Any]), initialize layers from config and hf_config
  - Implement forward(self, hidden_states, encoder_hidden_states, timestep, encoder_hidden_states_image=None, guidance=None, **kwargs)
  - If you want TeaCache acceleration, optionally override maybe_cache_states, should_skip_forward_for_cached_states, retrieve_cached_states
  - Ensure instance attrs like hidden_size, num_attention_heads, num_channels_latents are set

- For Text encoders, subclass TextEncoder. You must:
  - Set _supported_attention_backends class variable
  - Implement forward(...) to return BaseEncoderOutput
  - If loading stacked weights, use config.arch_config.stacked_params_mapping

- For Image encoders, subclass ImageEncoder and implement forward(pixel_values, **kwargs) -> BaseEncoderOutput

- For VAE, implement an nn.Module whose init matches your Config, and ensure it loads weights via safetensors (Diffusers format). See VAELoader for how it’s loaded.

Tip: Follow a similar structure to an existing model (e.g., WanTransformer3DModel) and copy over the minimal scaffolding to reduce mistakes.

## 4) Register the model in ModelRegistry

FastVideo resolves model classes via a central registry that maps external class names to internal modules/classes. Open fastvideo/models/registry.py and add your model to the appropriate mapping:

- Text-to-Video DiT: add to _TEXT_TO_VIDEO_DIT_MODELS
- Image-to-Video DiT: add to _IMAGE_TO_VIDEO_DIT_MODELS
- Text encoders: add to _TEXT_ENCODER_MODELS
- Image encoders: add to _IMAGE_ENCODER_MODELS
- VAEs: add to _VAE_MODELS
- Schedulers: add to _SCHEDULERS (rarely needed unless you add a new scheduler)

Entry format:
  "ExternalHFClassName": ("component_dir", "module_file_no_py", "InternalClassName")

Example for a new DiT named MyTransformer3DModel implemented at fastvideo/models/dits/myvideo.py:
  _TEXT_TO_VIDEO_DIT_MODELS = {
      ...,
      "MyTransformer3DModel": ("dits", "myvideo", "MyTransformer3DModel"),
  }

The registry will lazily import from fastvideo.models.<component_dir>.<module_file_no_py> and get the class InternalClassName.

Important: For Diffusers components, your exported config.json must contain "_class_name": "MyTransformer3DModel" so the loaders can resolve it.

## 5) Make configs/pipelines point to your model

Pipelines obtain architectures from the model’s own config at load time. Ensure your exported config.json (Diffusers) or Transformers config includes enough fields for update_model_arch to set:
- architectures (Transformers encoders)
- _class_name (Diffusers DiT/VAEs/Schedulers)
- other relevant structural hyperparameters

If you ship a new pipeline preset, add a PipelineConfig in fastvideo/configs/pipelines and include your new config defaults. See fastvideo/configs/pipelines/wan.py and stepvideo.py for reference.

## 6) Verify loading via ComponentLoader

- For DiT and VAE (Diffusers):
  - Ensure config.json has _class_name
  - Files: one or more *.safetensors weight files in the directory
  - TransformerLoader/VAELoader will resolve your class through ModelRegistry and call your init with (config=..., hf_config=...) for DiT or VAEConfig for VAEs

- For encoders (Transformers):
  - The loader reads the HF config and uses architectures to resolve the class via ModelRegistry
  - Implement load_weights and name mapping if your class needs custom weight layout

Run one of the example scripts under examples/inference/basic with your model checkpoint directory to validate.

## 7) Common pitfalls checklist

- _class_name missing in Diffusers config.json
- Model not added to the correct mapping in registry.py
- Incomplete class variables on DiT/TextEncoder subclasses (e.g., _supported_attention_backends)
- Weight names mismatch (fix via param_names_mapping or stacked_params_mapping in your ArchConfig)
- Device/offload expectations: review fastvideo_args flags for cpu_offload and FSDP inference

## 8) Minimal code skeletons

- DiT skeleton:

```python
from typing import Any
import torch
from fastvideo.models.dits.base import CachableDiT
from fastvideo.configs.models.dits.base import DiTConfig
from fastvideo.platforms import AttentionBackendEnum

class MyTransformer3DModel(CachableDiT):
    _fsdp_shard_conditions = []
    _compile_conditions = []
    param_names_mapping = {}
    reverse_param_names_mapping = {}
    lora_param_names_mapping = {}
    _supported_attention_backends = (
        AttentionBackendEnum.SLIDING_TILE_ATTN,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: DiTConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)
        # TODO: build layers using config/hf_config
        self.hidden_size = config.arch_config.hidden_size
        self.num_attention_heads = config.arch_config.num_attention_heads
        self.num_channels_latents = config.arch_config.num_channels_latents

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states, timestep: torch.LongTensor, encoder_hidden_states_image=None, guidance=None, **kwargs):
        # TODO: implement forward pass
        return hidden_states
```

- Text encoder skeleton:

```python
import torch
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.configs.models.encoders.base import TextEncoderConfig
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.configs.models.encoders import BaseEncoderOutput

class MyTextEncoder(TextEncoder):
    _supported_attention_backends = (
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: TextEncoderConfig):
        super().__init__(config)
        # TODO: build layers

    def forward(self, input_ids, position_ids=None, attention_mask=None, inputs_embeds=None, output_hidden_states=None, **kwargs) -> BaseEncoderOutput:
        # TODO: implement and return BaseEncoderOutput(hidden_states=..., last_hidden_state=...)
        raise NotImplementedError
```

## 9) Add your doc link to the Developer Guide (optional)

If this is for internal development, ensure the docs index includes this page under Developer Guide. In docs/source/index.md, under the Developer Guide toctree, add:

```
contributing/add_model
```

That’s it. With the registry system and component loaders, adding a new model is typically just 2–3 small code files plus a single-line registry entry, provided your checkpoint config.json exposes the expected metadata.
