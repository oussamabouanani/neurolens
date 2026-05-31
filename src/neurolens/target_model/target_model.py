from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from neurolens.utils.torch_utils import ImageProcessor, get_image_features


class TargetModel(ABC):
    def __init__(self, identifier: str, device: str | torch.device):

        self.identifier = identifier
        self.device = torch.device(device)

    @abstractmethod
    def get_img_processor(self) -> ImageProcessor:
        raise NotImplementedError

    @abstractmethod
    def get_total_neuron_count(self) -> int:
        raise NotImplementedError

    @torch.inference_mode()
    def get_activations(
        self,
        images: Image.Image | list[Image.Image],
    ) -> Float[Tensor, "n_imgs total_neuron_count"]:

        img_features = get_image_features(self.get_img_processor(), images)
        return self._get_activations(img_features.to(self.device))

    @abstractmethod
    def _get_activations(
        self,
        img_features: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch total_neuron_count"]:
        raise NotImplementedError
