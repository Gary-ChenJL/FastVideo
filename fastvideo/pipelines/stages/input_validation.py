# SPDX-License-Identifier: Apache-2.0
"""
Input validation stage for diffusion pipelines.
"""

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import load_image, load_video, pil_to_numpy, numpy_to_pt, normalize, resize
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import (StageValidators,
                                                   VerificationResult)

logger = init_logger(__name__)

# Alias for convenience
V = StageValidators


class InputValidationStage(PipelineStage):
    """
    Stage for validating and preparing inputs for diffusion pipelines.
    
    This stage validates that all required inputs are present and properly formatted
    before proceeding with the diffusion process.
    """

    def _generate_seeds(self, batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs):
        """Generate seeds for the inference"""
        seed = batch.seed
        num_videos_per_prompt = batch.num_videos_per_prompt

        assert seed is not None
        seeds = [seed + i for i in range(num_videos_per_prompt)]
        batch.seeds = seeds
        # Peiyuan: using GPU seed will cause A100 and H100 to generate different results...
        batch.generator = [
            torch.Generator("cpu").manual_seed(seed) for seed in seeds
        ]

    def _process_control_video(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        """Process control video for video-to-video tasks using batch attributes and extract FPS."""

        # Load video and extract FPS using custom function
        pil_images, original_fps = load_video(batch.control_video_path)

        logger.info(f"Loaded video with {len(pil_images)} frames, original FPS: {original_fps}")

        # Get target parameters from batch
        target_fps = batch.fps
        target_num_frames = batch.num_frames
        target_height = batch.height
        target_width = batch.width

        # Handle FPS sampling using extracted original FPS
        if target_fps is not None and original_fps is not None:
            frame_skip = max(1, int(original_fps // target_fps))
            if frame_skip > 1:
                pil_images = pil_images[::frame_skip]
                effective_fps = original_fps / frame_skip
                logger.info(
                    f"Resampled video from {original_fps:.1f} fps to {effective_fps:.1f} fps (skip={frame_skip})")

        # Limit to target number of frames
        if target_num_frames is not None and len(pil_images) > target_num_frames:
            pil_images = pil_images[:target_num_frames]
            logger.info(f"Limited video to {target_num_frames} frames (from {len(pil_images)} total)")

        # Resize each PIL image to target dimensions
        resized_images = []
        for pil_img in pil_images:
            resized_img = resize(pil_img, target_height, target_width, resize_mode="default", resample="lanczos")
            resized_images.append(resized_img)

        # Convert PIL images to numpy array
        video_numpy = pil_to_numpy(resized_images)
        video_numpy = normalize(video_numpy)
        video_tensor = numpy_to_pt(video_numpy)

        # Rearrange to [C, T, H, W] and add batch dimension -> [B, C, T, H, W]
        input_video = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        batch.video_latent = input_video

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Validate and prepare inputs.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The validated batch information.
        """

        self._generate_seeds(batch, fastvideo_args)

        # Ensure prompt is properly formatted
        if batch.prompt is None and batch.prompt_embeds is None:
            raise ValueError(
                "Either `prompt` or `prompt_embeds` must be provided")

        # Ensure negative prompt is properly formatted if using classifier-free guidance
        if (batch.do_classifier_free_guidance and batch.negative_prompt is None
                and batch.negative_prompt_embeds is None):
            raise ValueError(
                "For classifier-free guidance, either `negative_prompt` or "
                "`negative_prompt_embeds` must be provided")

        # Validate height and width
        if batch.height is None or batch.width is None:
            raise ValueError(
                "Height and width must be provided. Please set `height` and `width`."
            )
        if batch.height % 8 != 0 or batch.width % 8 != 0:
            raise ValueError(
                f"Height and width must be divisible by 8 but are {batch.height} and {batch.width}."
            )

        # Validate number of inference steps
        if batch.num_inference_steps <= 0:
            raise ValueError(
                f"Number of inference steps must be positive, but got {batch.num_inference_steps}"
            )

        # Validate guidance scale if using classifier-free guidance
        if batch.do_classifier_free_guidance and batch.guidance_scale <= 0:
            raise ValueError(
                f"Guidance scale must be positive, but got {batch.guidance_scale}"
            )

        # for i2v, get image from image_path
        if batch.image_path is not None:
            if batch.image_path.endswith(".mp4"):
                image = load_video(batch.image_path)[0]
            else:
                image = load_image(batch.image_path)
            batch.pil_image = image

        # for v2v, get control video from video path
        if batch.control_video_path is not None:
            self._process_control_video(batch, fastvideo_args)

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify input validation stage inputs."""
        result = VerificationResult()
        result.add_check("seed", batch.seed, [V.not_none, V.positive_int])
        result.add_check("num_videos_per_prompt", batch.num_videos_per_prompt,
                         V.positive_int)
        result.add_check(
            "prompt_or_embeds", None, lambda _: V.string_or_list_strings(
                batch.prompt) or V.list_not_empty(batch.prompt_embeds))
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check(
            "guidance_scale", batch.guidance_scale, lambda x: not batch.
            do_classifier_free_guidance or V.positive_float(x))
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify input validation stage outputs."""
        result = VerificationResult()
        result.add_check("seeds", batch.seeds, V.list_not_empty)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        return result
