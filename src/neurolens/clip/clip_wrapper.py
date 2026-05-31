import logging
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from neurolens.utils.torch_utils import (
    ImageProcessor,
    get_image_features,
    is_l2_normalized,
)

logger = logging.getLogger(__name__)


class CLIPWrapper(ABC):
    def __init__(self, identifier: str, device: str | torch.device):

        self.identifier = identifier
        self.device = torch.device(device)

    @abstractmethod
    def get_img_processor(self) -> ImageProcessor:
        raise NotImplementedError

    @abstractmethod
    def get_embd_dim(self) -> int:
        raise NotImplementedError

    @torch.inference_mode()
    def encode_image(
        self,
        images: Image.Image | list[Image.Image],
    ) -> Float[Tensor, "batch d_embd"]:

        img_embds = self._encode_image(get_image_features(self.get_img_processor(), images).to(self.device))
        self._verify_embds(img_embds)
        return img_embds

    @abstractmethod
    def _encode_image(
        self,
        img_features: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch d_embd"]:
        raise NotImplementedError

    @torch.inference_mode()
    def encode_text(self, text: str | list[str]) -> Float[Tensor, "batch d_embd"]:
        if isinstance(text, str):
            text_list = [text]
        elif isinstance(text, list) and len(text) == 0:
            raise ValueError("text must not be empty")
        elif isinstance(text, list) and all(isinstance(item, str) for item in text):
            text_list = text
        else:
            raise TypeError(f"[CLIP {self.identifier!r}] text must be a string or a list of strings")

        text_embds = self._encode_text(text_list)
        self._verify_embds(text_embds)
        return text_embds

    @abstractmethod
    def _encode_text(self, text_list: list[str]) -> Float[Tensor, "batch d_embd"]:
        raise NotImplementedError

    def _verify_embds(self, embds: Tensor) -> None:

        if embds.ndim != 2:
            raise ValueError(f"[CLIP {self.identifier!r}] image embeddings must have 2 dimensions")
        if not is_l2_normalized(embds, dim=1):
            raise ValueError(f"[CLIP {self.identifier!r}] image embeddings must be l2 normalized")
