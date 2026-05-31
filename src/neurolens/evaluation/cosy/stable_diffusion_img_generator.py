import gc
import logging
import os
from pathlib import Path

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from PIL import Image

from neurolens.config import NeuroLensConfig
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.str_utils import validate_plain_path_component

from .cosy_config import CoSyConfig

logger = logging.getLogger(__name__)


class StableDiffusionImageGenerator:
    def __init__(
        self,
        config: NeuroLensConfig,
        cosy_config: CoSyConfig,
        path_configs: PathConfigs,
        device: str | torch.device | None,
    ) -> None:

        self.config = config
        self.cosy_config = cosy_config
        self.path_configs = path_configs

        self.device = device

        self.pipe = None

        validate_plain_path_component(self.cosy_config.stable_diffusion_model_identifier)
        self.generated_img_dir_path = self.path_configs.join(
            self.config.io.raw_data_dir_name,
            self.cosy_config.generated_imgs_dir_name.format(
                img_generator=self.cosy_config.stable_diffusion_model_identifier
            ),
        )
        self.generated_img_dir_path.mkdir(parents=True, exist_ok=True)

    def get_generated_image_path(self, prompt: str, index: int) -> Path:
        return (
            self.generated_img_dir_path
            / prompt
            / self.cosy_config.generated_img_filename.format(prompt=prompt, index=index)
        )

    def count_existing_prompt_imgs(self, prompt: str) -> int:

        index = 0
        while True:
            file_path = self.get_generated_image_path(prompt, index)
            if not os.path.isfile(file_path):
                break
            index += 1
        return index

    def _reload_pipe(self):

        # TODO: This safety checker seems to work for now.. but a better more clean solution should be found
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cosy_config.stable_diffusion_model_repo,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        self.pipe.enable_attention_slicing()

    def _unload_pipe(self) -> None:
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_images(self, prompt: str):

        validate_plain_path_component(prompt)

        existing_count = self.count_existing_prompt_imgs(prompt)
        remaining_count = self.cosy_config.generated_img_count - existing_count

        if remaining_count <= 0:
            return

        if self.pipe is None:
            self._reload_pipe()

        # TODO: batch generation?
        images = self.pipe(
            [prompt] * remaining_count,
        ).images

        for index in range(remaining_count):
            img_path = self.get_generated_image_path(prompt, index + existing_count)
            img_path.parent.mkdir(parents=True, exist_ok=True)
            images[index].save(img_path)

    def load_images(self, prompt: str) -> list[Image.Image]:

        images = []

        for index in range(self.cosy_config.generated_img_count):
            filepath = self.get_generated_image_path(prompt, index)

            if not os.path.isfile(filepath):
                raise FileNotFoundError(
                    f"Image file for prompt {prompt!r} and index {index!r} does not exist at {filepath}"
                )

            image = Image.open(filepath)

            # TODO: Related to the issue of stable diffusion and its safety checker..
            # ignoring completely black images, good enough for now!
            if not image.getbbox():
                continue

            images.append(image)

        if len(images) == 0:
            logger.warning(f"No valid image file for prompt {prompt!r} found! Maybe all images were black?")
        elif len(images) < self.cosy_config.generated_img_count:
            logger.warning(
                f"Only {len(images)} out of {self.cosy_config.generated_img_count} image files"
                f" for prompt {prompt!r} found! Some might be all black!"
            )

        return images
