import open_clip
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from neurolens.clip import CLIPWrapper
from neurolens.img_text_model import CLIPImageTextModel
from neurolens.utils.torch_utils import ImageProcessor


class CLIPViTB32Wrapper(CLIPWrapper):
    _IDENTIFIER = "clip_vit_b_32"
    _HF_REPO = "hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"

    def __init__(self, device: str | torch.device):

        super().__init__(self._IDENTIFIER, device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self._HF_REPO)
        self.tokenizer = open_clip.get_tokenizer(self._HF_REPO)

        self.model = self.model.to(device).eval()

    def get_img_processor(self) -> ImageProcessor:
        return self.preprocess

    def get_embd_dim(self) -> int:
        return 512

    def _encode_image(
        self,
        img_features: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch d_embd"]:
        return F.normalize(self.model.encode_image(img_features), dim=-1, p=2)

    def _encode_text(self, text_list: list[str]) -> Float[Tensor, "batch d_embd"]:

        tokens = self.tokenizer(text_list).to(self.device)
        return F.normalize(self.model.encode_text(tokens), dim=-1, p=2)


class CLIPViTB32ImgTextModel(CLIPImageTextModel):
    def __init__(self, clip_wrapper: CLIPViTB32Wrapper):
        super().__init__(clip_wrapper)
