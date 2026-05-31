from jaxtyping import Float
from PIL import Image
from torch import Tensor

from neurolens.clip import CLIPWrapper

from .img_text_model import ImageTextModel


class CLIPImageTextModel(ImageTextModel):
    def __init__(self, clip_wrapper: CLIPWrapper):

        self.clip_wrapper = clip_wrapper

        super().__init__(
            identifier=f"{self.clip_wrapper.identifier}_img_text",
            device=clip_wrapper.device,
        )

    def get_embd_dim(self) -> int:
        return self.clip_wrapper.get_embd_dim()

    def get_img_embds(self, images: Image.Image | list[Image.Image]) -> Float[Tensor, "batch d_embd"]:
        return self.clip_wrapper.encode_image(images)

    def get_text_embds(self, text: str | list[str]) -> Float[Tensor, "batch d_embd"]:
        return self.clip_wrapper.encode_text(text)
