from abc import ABC, abstractmethod

import torch
from einops import einsum
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from neurolens.utils.torch_utils import is_l2_normalized


class ImageTextModel(ABC):
    def __init__(self, identifier: str, device: str | torch.device):

        self.identifier = identifier
        self.device = torch.device(device)

    @abstractmethod
    def get_embd_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_img_embds(self, images: Image.Image | list[Image.Image]) -> Float[Tensor, "batch d_embd"]:
        raise NotImplementedError

    @abstractmethod
    def get_text_embds(self, text: str | list[str]) -> Float[Tensor, "batch d_embd"]:
        raise NotImplementedError

    def get_similarities(
        self,
        img: Image.Image | list[Image.Image] | Float[Tensor, "n_img d_embd"],
        text: str | list[str] | Float[Tensor, "n_text d_embd"],
    ) -> Float[Tensor, "n_img n_text"]:

        if isinstance(img, Image.Image) or isinstance(img, list):
            if isinstance(img, list):
                if len(img) == 0:
                    raise ValueError("image list can not be empty")
                elif not all(isinstance(img, Image.Image) for img in img):
                    raise TypeError("One of the elements in the list is not a PIL image list!")
            img_embds = self.get_img_embds(img)
        elif isinstance(img, Tensor):
            img_embds = img
        else:
            raise TypeError("img must be a single PIL image or a list of PIL images")

        if isinstance(text, str) or isinstance(text, list):
            if isinstance(text, list):
                if len(text) == 0:
                    raise ValueError("text list can not be empty")
                elif not all(isinstance(text, str) for text in text):
                    raise TypeError("One of the elements in the list is not a string list!")
            text_embds = self.get_text_embds(text)
        elif isinstance(text, Tensor):
            text_embds = text
        else:
            raise TypeError("text must be a string or a list of strings")

        if len(img_embds.shape) != 2 or img_embds.shape[1] != self.get_embd_dim():
            raise ValueError(f"img_embds must have 2 dimensions and {self.get_embd_dim()} elements")
        if not is_l2_normalized(img_embds, dim=1):
            raise ValueError("img_embds must be l2 normalized")

        if len(text_embds.shape) != 2 or text_embds.shape[1] != self.get_embd_dim():
            raise ValueError(f"text_embds must have 2 dimensions and {self.get_embd_dim()} elements")
        if not is_l2_normalized(text_embds, dim=1):
            raise ValueError("text_embds must be l2 normalized")

        return einsum(img_embds, text_embds, "n_img d_embd, n_text d_embd->n_img n_text")
