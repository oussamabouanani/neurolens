import re
from abc import ABC, abstractmethod
from enum import StrEnum

import torch
from PIL import Image
from tqdm import tqdm

from neurolens.config import NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.target_model.neuron_data import BatchedNeuronData

from .utils import image_grid


class VLMTextGenerationMode(StrEnum):
    POS_ONLY = "pos_only"
    POS_NEG = "pos_neg"


class VLMWrapper(ABC):
    def __init__(
        self,
        identifier: str,
        config: NeuroLensConfig,
        img_dataset: ImageDataset,
        device: str | torch.device,
    ) -> None:

        self.config = config
        self.identifier = identifier
        self.img_dataset = img_dataset

        self.device = torch.device(device)

    def generated_text(self, neuron_data: BatchedNeuronData, mode: VLMTextGenerationMode) -> list[str]:

        generated_text = []

        pbar = tqdm(range(len(neuron_data)))
        for i in pbar:
            neuron_index = neuron_data.neuron_indices[i]
            pbar.set_description(f"Generating text for neuron {neuron_index}")

            pos_imgs = [
                self.img_dataset[int(index)] for index in neuron_data.pos_sample_indices[i][: self.config.vlm.img_count]
            ]
            neg_imgs = [
                self.img_dataset[int(index)] for index in neuron_data.neg_sample_indices[i][: self.config.vlm.img_count]
            ]

            pos_grid = image_grid(
                images=pos_imgs,
                size=self.config.vlm.grid_size,
                rows=self.config.vlm.grid_row_count,
            )

            if mode == VLMTextGenerationMode.POS_ONLY:
                grid = pos_grid
                generated_text += self.generate_pos_only_text(grid)

            elif mode == VLMTextGenerationMode.POS_NEG:
                neg_grid = image_grid(
                    images=neg_imgs,
                    size=self.config.vlm.grid_size,
                    rows=self.config.vlm.grid_row_count,
                )

                generated_text += self.generate_pos_neg_text(pos_grid, neg_grid)

            else:
                raise ValueError(f"Unknown text generation mode: {mode!r}")

        return generated_text

    def clean_output(self, response: str) -> list[str]:

        response = response.rstrip("\n")

        output: list[str] = []

        for text in response.split(","):
            clean_text = text.strip().lower().replace(" ", "_")
            clean_text = re.sub(r"[^A-Za-z0-9_-]+", "", clean_text)
            if len(clean_text) == 0 or len(clean_text) > self.config.vlm.max_text_length:
                continue
            output.append(clean_text)

        return output

    @abstractmethod
    def generate_pos_only_text(self, pos_grid: Image.Image) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def generate_pos_neg_text(self, pos_grid: Image.Image, neg_grid: Image.Image) -> list[str]:
        raise NotImplementedError
