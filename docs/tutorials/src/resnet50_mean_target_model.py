import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)

from neurolens.target_model import TargetModel
from neurolens.utils.torch_utils import ImageProcessor


def get_mean_activations(output_list: list):

    def hook(module, inputs, output):
        # Expecting output to be [batch_size, num_channels, height, width]
        output_list.append(torch.mean(output, dim=(2, 3)))

    return hook


class ResNet50TargetModel(TargetModel):
    def __init__(self, layer: int, device: str | torch.device):

        super().__init__(f"resnet50-{layer}", device)

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.preprocess = weights.transforms()

        self.model = self.model.to(device).eval()

        self.target_layer = getattr(self.model, f"layer{layer}")

        self.neuron_count = self.get_activations(Image.new("RGB", (224, 224))).shape[1]

    def get_img_processor(self) -> ImageProcessor:
        return self.preprocess

    def get_total_neuron_count(self) -> int:
        return self.neuron_count

    def _get_activations(
        self,
        img_features: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch total_neuron_count"]:

        activations = []

        handle = self.target_layer.register_forward_hook(get_mean_activations(activations))

        try:
            with torch.inference_mode():
                self.model(img_features)
        finally:
            handle.remove()

        if len(activations) != 1:
            raise RuntimeError(f"Expected one activation tensor, got {len(activations)}")

        return activations[0]
