import math

import pytest
import torch
from PIL import Image

from neurolens.img_text_model import ImageTextModel


class CompleteImageTextModel(ImageTextModel):
    def __init__(self, identifier="img-text-id", device="cpu"):
        super().__init__(identifier, device)
        self.last_images = None
        self.last_text = None

    def get_embd_dim(self) -> int:
        return 2

    def get_img_embds(self, images):
        self.last_images = images
        batch_len = len(images) if isinstance(images, list) else 1
        return torch.nn.functional.normalize(
            torch.ones((batch_len, self.get_embd_dim()), device=self.device),
            dim=1,
        )

    def get_text_embds(self, text):
        self.last_text = text
        batch_len = len(text) if isinstance(text, list) else 1
        return torch.nn.functional.normalize(
            torch.ones((batch_len, self.get_embd_dim()), device=self.device),
            dim=1,
        )


class MissingEmbeddingDimModel(ImageTextModel):
    def get_img_embds(self, images):
        return torch.ones((1, 2))

    def get_text_embds(self, text):
        return torch.ones((1, 2))


class MissingImageEmbeddingModel(ImageTextModel):
    def get_embd_dim(self) -> int:
        return 2

    def get_text_embds(self, text):
        return torch.ones((1, 2))


class MissingTextEmbeddingModel(ImageTextModel):
    def get_embd_dim(self) -> int:
        return 2

    def get_img_embds(self, images):
        return torch.ones((1, 2))


def test_base_class_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="abstract class"):
        ImageTextModel("base", "cpu")


@pytest.mark.parametrize(
    "model_cls",
    [MissingEmbeddingDimModel, MissingImageEmbeddingModel, MissingTextEmbeddingModel],
)
def test_subclasses_must_implement_every_public_operation(model_cls):
    with pytest.raises(TypeError, match="abstract class"):
        model_cls("broken", "cpu")


def test_constructor_stores_identifier_and_normalizes_device():
    model = CompleteImageTextModel("model-id", "cpu")

    assert model.identifier == "model-id"
    assert model.device == torch.device("cpu")


def test_constructor_rejects_invalid_device_values():
    with pytest.raises(RuntimeError):
        CompleteImageTextModel("model-id", "not-a-real-device")


def test_get_similarities_computes_tensor_dot_products():
    model = CompleteImageTextModel()
    img_embds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text_embds = torch.tensor([[1.0, 0.0], [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]])

    similarities = model.get_similarities(img_embds, text_embds)

    assert torch.allclose(
        similarities,
        torch.tensor([[1.0, 1.0 / math.sqrt(2)], [0.0, 1.0 / math.sqrt(2)]]),
    )


def test_get_similarities_routes_raw_images_and_text_through_embedding_methods():
    model = CompleteImageTextModel()
    images = [Image.new("RGB", (5, 7)), Image.new("RGB", (5, 7))]
    text = ["cat", "dog", "horse"]

    similarities = model.get_similarities(images, text)

    assert similarities.shape == (2, 3)
    assert torch.allclose(similarities, torch.ones((2, 3)))
    assert model.last_images == images
    assert model.last_text == text


def test_get_similarities_accepts_single_image_and_single_text():
    model = CompleteImageTextModel()
    image = Image.new("RGB", (5, 7))

    similarities = model.get_similarities(image, "cat")

    assert similarities.shape == (1, 1)
    assert torch.allclose(similarities, torch.ones((1, 1)))
    assert model.last_images == image
    assert model.last_text == "cat"


@pytest.mark.parametrize("img", [[], [Image.new("RGB", (5, 7)), torch.zeros(2)]])
def test_get_similarities_rejects_invalid_image_lists(img):
    model = CompleteImageTextModel()
    text_embds = torch.tensor([[1.0, 0.0]])

    with pytest.raises((TypeError, ValueError)):
        model.get_similarities(img, text_embds)


@pytest.mark.parametrize("text", [[], ["cat", 1]])
def test_get_similarities_rejects_invalid_text_lists(text):
    model = CompleteImageTextModel()
    img_embds = torch.tensor([[1.0, 0.0]])

    with pytest.raises((TypeError, ValueError)):
        model.get_similarities(img_embds, text)


def test_get_similarities_rejects_unsupported_image_input_type():
    model = CompleteImageTextModel()

    with pytest.raises(TypeError, match="img must"):
        model.get_similarities(123, torch.tensor([[1.0, 0.0]]))


def test_get_similarities_rejects_unsupported_text_input_type():
    model = CompleteImageTextModel()

    with pytest.raises(TypeError, match="text must"):
        model.get_similarities(torch.tensor([[1.0, 0.0]]), 123)


@pytest.mark.parametrize(
    "img_embds",
    [torch.ones((2,)), torch.ones((1, 3))],
)
def test_get_similarities_rejects_image_embeddings_with_wrong_shape(img_embds):
    model = CompleteImageTextModel()
    text_embds = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="img_embds must have 2 dimensions"):
        model.get_similarities(img_embds, text_embds)


@pytest.mark.parametrize(
    "text_embds",
    [torch.ones((2,)), torch.ones((1, 3))],
)
def test_get_similarities_rejects_text_embeddings_with_wrong_shape(text_embds):
    model = CompleteImageTextModel()
    img_embds = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="text_embds must have 2 dimensions"):
        model.get_similarities(img_embds, text_embds)


def test_get_similarities_rejects_non_normalized_image_embeddings():
    model = CompleteImageTextModel()

    with pytest.raises(ValueError, match="img_embds must be l2 normalized"):
        model.get_similarities(torch.tensor([[2.0, 0.0]]), torch.tensor([[1.0, 0.0]]))


def test_get_similarities_rejects_non_normalized_text_embeddings():
    model = CompleteImageTextModel()

    with pytest.raises(ValueError, match="text_embds must be l2 normalized"):
        model.get_similarities(torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]]))
