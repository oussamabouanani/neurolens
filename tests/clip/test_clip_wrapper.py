import pytest
import torch
from PIL import Image

from neurolens.clip import CLIPWrapper
from neurolens.utils.torch_utils import ImageProcessor


def image_to_tensor_processor(image):
    return torch.zeros((3, image.height, image.width))


class CompleteCLIPWrapper(CLIPWrapper):
    def get_img_processor(self) -> ImageProcessor:
        return image_to_tensor_processor

    def get_embd_dim(self):
        return 2

    def _encode_image(self, img_features):
        return torch.nn.functional.normalize(
            torch.ones((img_features.shape[0], 2), device=self.device),
            dim=1,
        )

    def _encode_text(self, text_list):
        return torch.nn.functional.normalize(
            torch.ones((len(text_list), 2), device=self.device),
            dim=1,
        )


class MissingImageProcessorWrapper(CLIPWrapper):
    def get_embd_dim(self):
        return 2

    def _encode_image(self, img_features):
        return torch.nn.functional.normalize(
            torch.ones((img_features.shape[0], 2), device=self.device),
            dim=1,
        )

    def _encode_text(self, text_list):
        return torch.nn.functional.normalize(
            torch.ones((len(text_list), 2), device=self.device),
            dim=1,
        )


class MissingImageEncoderWrapper(CLIPWrapper):
    def get_img_processor(self) -> ImageProcessor:
        return image_to_tensor_processor

    def get_embd_dim(self):
        return 2

    def _encode_text(self, text_list):
        return torch.nn.functional.normalize(
            torch.ones((len(text_list), 2), device=self.device),
            dim=1,
        )


class MissingTextEncoderWrapper(CLIPWrapper):
    def get_img_processor(self) -> ImageProcessor:
        return image_to_tensor_processor

    def get_embd_dim(self):
        return 2

    def _encode_image(self, img_features):
        return torch.nn.functional.normalize(
            torch.ones((img_features.shape[0], 2), device=self.device),
            dim=1,
        )


class MissingEmbeddingDimWrapper(CLIPWrapper):
    def get_img_processor(self) -> ImageProcessor:
        return image_to_tensor_processor

    def _encode_image(self, img_features):
        return torch.nn.functional.normalize(
            torch.ones((img_features.shape[0], 2), device=self.device),
            dim=1,
        )

    def _encode_text(self, text_list):
        return torch.nn.functional.normalize(
            torch.ones((len(text_list), 2), device=self.device),
            dim=1,
        )


class BadProcessorWrapper(CompleteCLIPWrapper):
    def get_img_processor(self) -> ImageProcessor:
        return lambda images: images


def test_base_class_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="abstract class"):
        CLIPWrapper("base", "cpu")


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        MissingImageProcessorWrapper,
        MissingEmbeddingDimWrapper,
        MissingImageEncoderWrapper,
        MissingTextEncoderWrapper,
    ],
)
def test_subclasses_must_implement_every_public_operation(wrapper_cls):
    with pytest.raises(TypeError, match="abstract class"):
        wrapper_cls("broken", "cpu")


def test_constructor_stores_identifier_and_normalizes_string_device():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    assert wrapper.identifier == "clip-id"
    assert wrapper.device == torch.device("cpu")


def test_constructor_accepts_torch_device_instances():
    device = torch.device("cpu")

    wrapper = CompleteCLIPWrapper("clip-id", device)

    assert wrapper.device == device


def test_constructor_rejects_invalid_device_values():
    with pytest.raises(RuntimeError):
        CompleteCLIPWrapper("clip-id", "not-a-real-device")


def test_img_processor_accepts_single_image_and_returns_tensor():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")
    processor = wrapper.get_img_processor()
    image = Image.new("RGB", (5, 7))

    img_features = processor(image)

    assert isinstance(img_features, torch.Tensor)
    assert img_features.shape == (3, 7, 5)


def test_get_embd_dim_returns_embedding_dimension():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    assert wrapper.get_embd_dim() == 2


def test_encode_image_processes_pil_image_lists_before_encoding():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")
    images = [Image.new("RGB", (5, 7)), Image.new("RGB", (5, 7))]

    image_embeddings = wrapper.encode_image(images)

    assert image_embeddings.shape == (2, 2)
    assert image_embeddings.device == wrapper.device
    assert torch.allclose(torch.linalg.norm(image_embeddings, dim=1), torch.ones(2))


def test_encode_image_processes_single_pil_image_before_encoding():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")
    image = Image.new("RGB", (5, 7))

    image_embeddings = wrapper.encode_image(image)

    assert image_embeddings.shape == (1, 2)
    assert image_embeddings.device == wrapper.device
    assert torch.allclose(torch.linalg.norm(image_embeddings, dim=1), torch.ones(1))


def test_encode_image_rejects_batched_tensor_inputs():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    with pytest.raises(TypeError, match="single PIL image or a list of PIL images"):
        wrapper.encode_image(torch.zeros((2, 3, 7, 5)))


def test_encode_image_rejects_preprocessed_tensor_lists():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")
    image_features = [
        torch.full((3, 7, 5), 1.0),
        torch.full((3, 7, 5), 2.0),
    ]

    with pytest.raises(TypeError, match="not a PIL image list"):
        wrapper.encode_image(image_features)


def test_encode_image_rejects_empty_image_lists():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    with pytest.raises(ValueError, match="image list can not be empty"):
        wrapper.encode_image([])


def test_encode_image_rejects_mixed_image_and_tensor_lists():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    with pytest.raises(TypeError, match="not a PIL image list"):
        wrapper.encode_image([Image.new("RGB", (5, 7)), torch.zeros((3, 7, 5))])


def test_encode_image_rejects_non_tensor_processor_outputs():
    wrapper = BadProcessorWrapper("clip-id", "cpu")

    with pytest.raises(TypeError, match="expected Tensor as element 0"):
        wrapper.encode_image([Image.new("RGB", (5, 7))])


class BadRankProcessorWrapper(CompleteCLIPWrapper):
    def get_img_processor(self) -> ImageProcessor:
        return lambda image: torch.zeros((7, 5))


def test_encode_image_rejects_processor_outputs_with_wrong_rank():
    wrapper = BadRankProcessorWrapper("clip-id", "cpu")

    with pytest.raises(ValueError, match="image features must have 4 dimensions"):
        wrapper.encode_image(Image.new("RGB", (5, 7)))


class BadRankImageEncoderWrapper(CompleteCLIPWrapper):
    def _encode_image(self, img_features):
        return torch.ones((img_features.shape[0], 2, 1), device=self.device)


def test_encode_image_rejects_embeddings_with_wrong_rank():
    wrapper = BadRankImageEncoderWrapper("clip-id", "cpu")

    with pytest.raises(ValueError, match="embeddings must have 2 dimensions"):
        wrapper.encode_image(Image.new("RGB", (5, 7)))


class UnnormalizedImageEncoderWrapper(CompleteCLIPWrapper):
    def _encode_image(self, img_features):
        return torch.ones((img_features.shape[0], 2), device=self.device)


def test_encode_image_rejects_unnormalized_embeddings():
    wrapper = UnnormalizedImageEncoderWrapper("clip-id", "cpu")

    with pytest.raises(ValueError, match="embeddings must be l2 normalized"):
        wrapper.encode_image(Image.new("RGB", (5, 7)))


def test_encode_text_returns_tensor_on_wrapper_device():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    text_embeddings = wrapper.encode_text(["cat", "dog", "horse"])

    assert text_embeddings.shape == (3, 2)
    assert text_embeddings.device == wrapper.device
    assert torch.allclose(torch.linalg.norm(text_embeddings, dim=1), torch.ones(3))


def test_encode_text_accepts_single_string():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    text_embeddings = wrapper.encode_text("cat")

    assert text_embeddings.shape == (1, 2)
    assert text_embeddings.device == wrapper.device
    assert torch.allclose(torch.linalg.norm(text_embeddings, dim=1), torch.ones(1))


def test_encode_text_rejects_empty_text_lists():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    with pytest.raises(ValueError, match="text must not be empty"):
        wrapper.encode_text([])


def test_encode_text_rejects_non_string_items():
    wrapper = CompleteCLIPWrapper("clip-id", "cpu")

    with pytest.raises(TypeError, match="string or a list of strings"):
        wrapper.encode_text(["cat", 1])


class BadRankTextEncoderWrapper(CompleteCLIPWrapper):
    def _encode_text(self, text_list):
        return torch.ones((len(text_list), 2, 1), device=self.device)


def test_encode_text_rejects_embeddings_with_wrong_rank():
    wrapper = BadRankTextEncoderWrapper("clip-id", "cpu")

    with pytest.raises(ValueError, match="embeddings must have 2 dimensions"):
        wrapper.encode_text("cat")


class UnnormalizedTextEncoderWrapper(CompleteCLIPWrapper):
    def _encode_text(self, text_list):
        return torch.ones((len(text_list), 2), device=self.device)


def test_encode_text_rejects_unnormalized_embeddings():
    wrapper = UnnormalizedTextEncoderWrapper("clip-id", "cpu")

    with pytest.raises(ValueError, match="embeddings must be l2 normalized"):
        wrapper.encode_text("cat")
