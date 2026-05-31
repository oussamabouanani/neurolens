import pytest
import torch
from PIL import Image

from neurolens.target_model import TargetModel
from neurolens.utils.torch_utils import ImageProcessor


def image_to_tensor_processor(image):
    return torch.zeros((3, image.height, image.width))


class CompleteTargetModel(TargetModel):
    def __init__(self, identifier, device):
        super().__init__(identifier, device)
        self.last_img_features = None

    def get_img_processor(self) -> ImageProcessor:
        return image_to_tensor_processor

    def get_total_neuron_count(self) -> int:
        return 4

    def _get_activations(self, img_features):
        self.last_img_features = img_features
        return torch.ones(
            (img_features.shape[0], self.get_total_neuron_count()),
            device=self.device,
        )


class MissingImageProcessorTargetModel(TargetModel):
    def get_total_neuron_count(self) -> int:
        return 4

    def _get_activations(self, img_features):
        return torch.ones((img_features.shape[0], 4), device=self.device)


class MissingNeuronCountTargetModel(TargetModel):
    def get_img_processor(self) -> ImageProcessor:
        return image_to_tensor_processor

    def _get_activations(self, img_features):
        return torch.ones((img_features.shape[0], 4), device=self.device)


class MissingActivationsTargetModel(TargetModel):
    def get_img_processor(self) -> ImageProcessor:
        return image_to_tensor_processor

    def get_total_neuron_count(self) -> int:
        return 4


def test_base_class_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="abstract class"):
        TargetModel("base", "cpu")


@pytest.mark.parametrize(
    "model_cls",
    [
        MissingImageProcessorTargetModel,
        MissingNeuronCountTargetModel,
        MissingActivationsTargetModel,
    ],
)
def test_subclasses_must_implement_every_public_operation(model_cls):
    with pytest.raises(TypeError, match="abstract class"):
        model_cls("broken", "cpu")


def test_constructor_stores_identifier_and_normalizes_string_device():
    model = CompleteTargetModel("target-id", "cpu")

    assert model.identifier == "target-id"
    assert model.device == torch.device("cpu")


def test_constructor_accepts_torch_device_instances():
    device = torch.device("cpu")

    model = CompleteTargetModel("target-id", device)

    assert model.device == device


def test_constructor_rejects_invalid_device_values():
    with pytest.raises(RuntimeError):
        CompleteTargetModel("target-id", "not-a-real-device")


def test_get_activations_processes_single_pil_image_before_encoding():
    model = CompleteTargetModel("target-id", "cpu")
    image = Image.new("RGB", (5, 7))

    activations = model.get_activations(image)

    assert activations.shape == (1, model.get_total_neuron_count())
    assert activations.device == model.device
    assert model.last_img_features.shape == (1, 3, 7, 5)


def test_get_activations_processes_pil_image_lists_before_encoding():
    model = CompleteTargetModel("target-id", "cpu")
    images = [Image.new("RGB", (5, 7)), Image.new("RGB", (5, 7))]

    activations = model.get_activations(images)

    assert activations.shape == (2, model.get_total_neuron_count())
    assert activations.device == model.device
    assert model.last_img_features.shape == (2, 3, 7, 5)


def test_get_activations_runs_in_inference_mode():
    class GradModeCheckingTargetModel(CompleteTargetModel):
        def _get_activations(self, img_features):
            assert not torch.is_grad_enabled()
            return super()._get_activations(img_features)

    model = GradModeCheckingTargetModel("target-id", "cpu")

    model.get_activations(Image.new("RGB", (5, 7)))


def test_get_activations_rejects_batched_tensor_inputs():
    model = CompleteTargetModel("target-id", "cpu")

    with pytest.raises(TypeError, match="single PIL image or a list of PIL images"):
        model.get_activations(torch.zeros((2, 3, 7, 5)))


def test_get_activations_rejects_empty_image_lists():
    model = CompleteTargetModel("target-id", "cpu")

    with pytest.raises(ValueError, match="image list can not be empty"):
        model.get_activations([])


def test_get_activations_rejects_mixed_image_and_tensor_lists():
    model = CompleteTargetModel("target-id", "cpu")

    with pytest.raises(TypeError, match="not a PIL image list"):
        model.get_activations([Image.new("RGB", (5, 7)), torch.zeros((3, 7, 5))])


class BadProcessorTargetModel(CompleteTargetModel):
    def get_img_processor(self) -> ImageProcessor:
        return lambda image: image


def test_get_activations_rejects_non_tensor_processor_outputs():
    model = BadProcessorTargetModel("target-id", "cpu")

    with pytest.raises(TypeError, match="expected Tensor as element 0"):
        model.get_activations(Image.new("RGB", (5, 7)))


class BadRankProcessorTargetModel(CompleteTargetModel):
    def get_img_processor(self) -> ImageProcessor:
        return lambda image: torch.zeros((7, 5))


def test_get_activations_rejects_processor_outputs_with_wrong_rank():
    model = BadRankProcessorTargetModel("target-id", "cpu")

    with pytest.raises(ValueError, match="image features must have 4 dimensions"):
        model.get_activations(Image.new("RGB", (5, 7)))
