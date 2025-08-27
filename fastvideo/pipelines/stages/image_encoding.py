# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import PIL
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.models.vision_utils import (get_default_height_width, normalize,
                                           numpy_to_pt, pil_to_numpy, resize)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ImageEncodingStage(PipelineStage):
    """
    Stage for encoding image prompts into embeddings for diffusion models.
    
    This stage handles the encoding of image prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, image_encoder, image_processor) -> None:
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary image encoder.
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt into image encoder hidden states.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with encoded prompt embeddings.
        """
        self.image_encoder = self.image_encoder.to(get_local_torch_device())

        image = batch.pil_image

        image_inputs = self.image_processor(
            images=image, return_tensors="pt").to(get_local_torch_device())
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.image_encoder(**image_inputs)
            image_embeds = outputs.last_hidden_state

        batch.image_embeds.append(image_embeds)

        if fastvideo_args.image_encoder_cpu_offload:
            self.image_encoder.to('cpu')

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        result = VerificationResult()
        result.add_check("pil_image", batch.pil_image, V.not_none)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify image encoding stage outputs."""
        result = VerificationResult()
        result.add_check("image_embeds", batch.image_embeds,
                         V.list_of_tensors_dims(3))
        return result


class ImageVAEEncodingStage(PipelineStage):
    """
    Stage for encoding pixel representations into latent space.
    
    This stage handles the encoding of pixel representations into the final
    input format (e.g., latents).
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
        self.vae: ParallelTiledVAE = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode pixel representations into latent space.

        Processes both original image (for I2V) and control video (if present).

        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.

        Returns:
            The batch with encoded outputs.
        """
        assert batch.height is not None and isinstance(batch.height, int)
        assert batch.width is not None and isinstance(batch.width, int)
        assert batch.num_frames is not None and isinstance(
            batch.num_frames, int)

        self.vae = self.vae.to(get_local_torch_device())

        latent_height = batch.height // self.vae.spatial_compression_ratio
        latent_width = batch.width // self.vae.spatial_compression_ratio

        if batch.pil_image is not None:
            batch = self._process_image_to_video(
                batch, fastvideo_args, latent_height, latent_width
            )

        # Control video processing
        if batch.video_latent is not None:
            batch = self._process_control_video(
                batch, fastvideo_args, batch.video_latent, latent_height, latent_width
            )

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")

        return batch

    def _process_image_to_video(
            self,
            batch: ForwardBatch,
            fastvideo_args: FastVideoArgs,
            latent_height: int,
            latent_width: int
    ) -> ForwardBatch:
        """Process single image for I2V (original logic)."""
        image = batch.pil_image
        image = self.preprocess(
            image,
            vae_scale_factor=self.vae.spatial_compression_ratio,
            height=batch.height,
            width=batch.width).to(get_local_torch_device(), dtype=torch.float32)

        image = image.unsqueeze(2)

        video_condition = torch.cat([
            image,
            image.new_zeros(image.shape[0], image.shape[1],
                            batch.num_frames - 1, batch.height, batch.width)
        ],
            dim=2)
        video_condition = video_condition.to(device=get_local_torch_device(),
                                             dtype=torch.float32)

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
                                       vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Encode Image
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            # if fastvideo_args.vae_sp:
            #     self.vae.enable_parallel()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        generator = batch.generator
        if generator is None:
            raise ValueError("Generator must be provided")
        latent_condition = self.retrieve_latents(encoder_output, generator)

        # Apply shifting if needed
        if (hasattr(self.vae, "shift_factor")
                and self.vae.shift_factor is not None):
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latent_condition -= self.vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.vae.scaling_factor.to(
                latent_condition.device, latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.vae.scaling_factor

        # Original I2V masking logic
        mask_lat_size = torch.ones(1, 1, batch.num_frames, latent_height,
                                   latent_width)
        mask_lat_size[:, :, list(range(1, batch.num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            dim=2,
            repeats=self.vae.temporal_compression_ratio)
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(1, -1,
                                           self.vae.temporal_compression_ratio,
                                           latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        batch.image_latent = torch.concat([mask_lat_size, latent_condition],
                                          dim=1)
        return batch

    def _process_control_video(
            self,
            batch: ForwardBatch,
            fastvideo_args: FastVideoArgs,
            control_video,
            latent_height: int,
            latent_width: int
    ) -> ForwardBatch:
        """Process control video (new functionality)."""
        # Prepare video tensor from control video
        video_condition = self._prepare_control_video_tensor(
            control_video, batch.num_frames, batch.height, batch.width
        ).to(get_local_torch_device(), dtype=torch.float32)

        # Setup VAE precision (same as original)
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Encode control video
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        generator = batch.generator
        if generator is None:
            raise ValueError("Generator must be provided")
        latent_condition = self.retrieve_latents(encoder_output, generator)

        if (hasattr(self.vae, "shift_factor")
                and self.vae.shift_factor is not None):
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latent_condition -= self.vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.vae.scaling_factor.to(
                latent_condition.device, latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.vae.scaling_factor

        mask_lat_size = self._create_control_mask(
            batch.num_frames, latent_height, latent_width, latent_condition.device
        )
        batch.video_latent = torch.concat([mask_lat_size, latent_condition], dim=1)

        return batch

    def _prepare_control_video_tensor(
            self,
            control_video,
            num_frames: int,
            height: int,
            width: int
    ) -> torch.Tensor:
        """
        Prepare video tensor from control video input.
        """
        if isinstance(control_video, list):
            processed_frames = []
            for i, frame in enumerate(control_video):
                if i >= num_frames:
                    break
                processed_frame = self.preprocess(
                    frame,
                    vae_scale_factor=self.vae.spatial_compression_ratio,
                    height=height,
                    width=width
                ).to(get_local_torch_device(), dtype=torch.float32)
                processed_frames.append(processed_frame)

            if processed_frames:
                video_tensor = torch.cat([f.unsqueeze(2) for f in processed_frames], dim=2)
            else:
                video_tensor = torch.zeros(1, 3, 0, height, width,
                                           device=get_local_torch_device(), dtype=torch.float32)

        elif isinstance(control_video, torch.Tensor):
            # Handle tensor input [batch, channels, frames, height, width]
            video_tensor = control_video.to(get_local_torch_device(), dtype=torch.float32)

            if video_tensor.shape[2] > num_frames:
                video_tensor = video_tensor[:, :, :num_frames]

        else:
            raise ValueError(f"Unsupported control_video type: {type(control_video)}. "
                             "Expected list of PIL Images or torch.Tensor.")

        # Pad with zeros if we have fewer frames than required
        current_frames = video_tensor.shape[2]
        if current_frames < num_frames:
            padding_frames = num_frames - current_frames
            zero_padding = torch.zeros(
                video_tensor.shape[0], video_tensor.shape[1],
                padding_frames, height, width,
                device=video_tensor.device, dtype=video_tensor.dtype
            )
            video_tensor = torch.cat([video_tensor, zero_padding], dim=2)

        return video_tensor

    def _create_control_mask(
            self,
            num_frames: int,
            latent_height: int,
            latent_width: int,
            device
    ) -> torch.Tensor:
        """
        Create control mask where all frames are valid conditioning.
        """
        # All frames are valid for control
        mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width)

        if hasattr(self.vae, 'temporal_compression_ratio') and self.vae.temporal_compression_ratio > 1:
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask,
                dim=2,
                repeats=self.vae.temporal_compression_ratio
            )
            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )
            mask_lat_size = mask_lat_size.view(1, -1,
                                               self.vae.temporal_compression_ratio,
                                               latent_height, latent_width)
            mask_lat_size = mask_lat_size.transpose(1, 2)

        return mask_lat_size.to(device)

    def retrieve_latents(self,
                         encoder_output: torch.Tensor,
                         generator: torch.Generator | None = None,
                         sample_mode: str = "sample"):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError(
                "Could not access latents of provided encoder_output")

    def preprocess(
            self,
            image: PIL.Image.Image,
            vae_scale_factor: int,
            height: int | None = None,
            width: int | None = None,
            resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> torch.Tensor:
        image = [image]

        height, width = get_default_height_width(image[0], vae_scale_factor,
                                                 height, width)
        image = [
            resize(i, height, width, resize_mode=resize_mode) for i in image
        ]
        image = pil_to_numpy(image)  # to np
        image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()
        # result.add_check("pil_image", batch.pil_image)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        result.add_check("image_latent", batch.image_latent,
                         [V.is_tensor, V.with_dims(5)])
        return result

class VideoVAEEncodingStage(ImageVAEEncodingStage):
    """
    Stage for encoding control video into latent space for diffusion models.

    Extends ImageVAEEncodingStage to handle control video frames instead of
    creating zero-padded video from a single image.
    """

    def forward(
            self,
            batch: ForwardBatch,
            fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode control video frames into latent space.

        Args:
            batch: The current batch information containing control_video frames.
            fastvideo_args: The inference arguments.

        Returns:
            The batch with encoded control latents.
        """
        # Validate required inputs (inherited validation logic)
        assert batch.height is not None and isinstance(batch.height, int)
        assert batch.width is not None and isinstance(batch.width, int)
        assert batch.num_frames is not None and isinstance(batch.num_frames, int)

        # Check for control video input
        control_video = getattr(batch, 'control_video', None)
        if control_video is None:
            logger.warning("No control_video found in batch. Skipping VideoVAEEncodingStage.")
            return batch

        self.vae = self.vae.to(get_local_torch_device())

        latent_height = batch.height // self.vae.spatial_compression_ratio
        latent_width = batch.width // self.vae.spatial_compression_ratio

        # Create video tensor from control video (this is the key difference)
        video_condition = self._prepare_video_tensor(
            control_video, batch.num_frames, batch.height, batch.width
        ).to(get_local_torch_device(), dtype=torch.float32)

        # Everything else follows the parent class logic
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
                                       vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Encode video using inherited VAE setup
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()

            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)

            encoder_output = self.vae.encode(video_condition)

        # Use inherited latent retrieval
        generator = batch.generator
        if generator is None:
            raise ValueError("Generator must be provided")
        latent_condition = self.retrieve_latents(encoder_output, generator)

        # Apply inherited VAE transformations (shifting and scaling)
        if (hasattr(self.vae, "shift_factor")
                and self.vae.shift_factor is not None):
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latent_condition -= self.vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.vae.scaling_factor.to(
                latent_condition.device, latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.vae.scaling_factor

        # Create control-specific mask (all frames are valid)
        mask_lat_size = self._create_control_mask(
            batch.num_frames, latent_height, latent_width, latent_condition.device
        )

        # Store as control latents instead of image latents
        batch.control_latents = torch.concat([mask_lat_size, latent_condition], dim=1)

        # Inherited cleanup
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")
        return batch

    def _prepare_video_tensor(
            self,
            control_video,
            num_frames: int,
            height: int,
            width: int
    ) -> torch.Tensor:
        """
        Prepare video tensor from control video input.

        This replaces the single image + zero padding logic from the parent class.
        """
        if isinstance(control_video, list):
            # Handle list of PIL Images
            processed_frames = []
            for i, frame in enumerate(control_video):
                if i >= num_frames:
                    break
                processed_frame = self.preprocess(  # Use inherited preprocess
                    frame,
                    vae_scale_factor=self.vae.spatial_compression_ratio,
                    height=height,
                    width=width
                ).to(get_local_torch_device(), dtype=torch.float32)
                processed_frames.append(processed_frame)

            # Stack frames and add temporal dimension
            if processed_frames:
                video_tensor = torch.cat([f.unsqueeze(2) for f in processed_frames], dim=2)
            else:
                # Fallback to zeros if no frames
                video_tensor = torch.zeros(1, 3, 0, height, width)

        elif isinstance(control_video, torch.Tensor):
            # Handle tensor input [batch, channels, frames, height, width]
            video_tensor = control_video.to(get_local_torch_device(), dtype=torch.float32)

            # Limit to num_frames
            if video_tensor.shape[2] > num_frames:
                video_tensor = video_tensor[:, :, :num_frames]

        else:
            raise ValueError(f"Unsupported control_video type: {type(control_video)}. "
                             "Expected list of PIL Images or torch.Tensor.")

        # Pad with zeros if we have fewer frames than required
        current_frames = video_tensor.shape[2]
        if current_frames < num_frames:
            padding_frames = num_frames - current_frames
            zero_padding = torch.zeros(
                video_tensor.shape[0], video_tensor.shape[1],
                padding_frames, height, width,
                device=video_tensor.device, dtype=video_tensor.dtype
            )
            video_tensor = torch.cat([video_tensor, zero_padding], dim=2)

        return video_tensor

    def _create_control_mask(
            self,
            num_frames: int,
            latent_height: int,
            latent_width: int,
            device
    ) -> torch.Tensor:
        """
        Create control mask for video latents.

        Unlike ImageVAEEncodingStage which masks out generated frames,
        control video has all frames as valid conditioning.
        """
        # All frames are valid for control (unlike I2V where only first frame is real)
        mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width)

        # Apply temporal compression handling if needed (inherited logic)
        if hasattr(self.vae, 'temporal_compression_ratio') and self.vae.temporal_compression_ratio > 1:
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask,
                dim=2,
                repeats=self.vae.temporal_compression_ratio
            )
            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )
            mask_lat_size = mask_lat_size.view(1, -1,
                                               self.vae.temporal_compression_ratio,
                                               latent_height, latent_width)
            mask_lat_size = mask_lat_size.transpose(1, 2)

        return mask_lat_size.to(device)

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify video VAE encoding stage inputs."""
        # Get base validation from parent
        result = super().verify_input(batch, fastvideo_args)

        # Add control video specific validation
        control_video = getattr(batch, 'control_video', None)
        if control_video is not None:
            if isinstance(control_video, list):
                result.add_check("control_video", control_video, V.list_not_empty)
            elif isinstance(control_video, torch.Tensor):
                result.add_check("control_video", control_video,
                                 [V.is_tensor, V.with_dims(5)])

        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify video VAE encoding stage outputs."""
        result = VerificationResult()

        # Verify control latents instead of image latents
        if hasattr(batch, 'control_latents') and batch.control_latents is not None:
            result.add_check("control_latents", batch.control_latents,
                             [V.is_tensor, V.with_dims(5)])

        return result