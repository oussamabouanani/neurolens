import torch
from PIL import Image

from neurolens.img_text_model import CLIPImageTextModel


class RecordingCLIPWrapper:
    identifier = "clip"
    device = torch.device("cpu")

    def __init__(self):
        self.image_inputs = []
        self.text_inputs = []

    def get_embd_dim(self):
        return 4

    def encode_image(self, images):
        self.image_inputs.append(images)
        batch_size = len(images) if isinstance(images, list) else 1
        return torch.ones((batch_size, self.get_embd_dim()))

    def encode_text(self, text):
        self.text_inputs.append(text)
        batch_size = len(text) if isinstance(text, list) else 1
        return torch.ones((batch_size, self.get_embd_dim())) * 2


def test_clip_image_text_model_delegates_metadata_and_embedding_dim():
    clip_wrapper = RecordingCLIPWrapper()

    model = CLIPImageTextModel(clip_wrapper)

    assert model.identifier == "clip_img_text"
    assert model.device == torch.device("cpu")
    assert model.get_embd_dim() == 4


def test_clip_image_text_model_delegates_image_and_text_embeddings():
    clip_wrapper = RecordingCLIPWrapper()
    model = CLIPImageTextModel(clip_wrapper)
    images = [Image.new("RGB", (5, 7)), Image.new("RGB", (5, 7))]

    img_embds = model.get_img_embds(images)
    text_embds = model.get_text_embds(["cat", "dog", "horse"])

    assert clip_wrapper.image_inputs == [images]
    assert clip_wrapper.text_inputs == [["cat", "dog", "horse"]]
    assert torch.equal(img_embds, torch.ones((2, 4)))
    assert torch.equal(text_embds, torch.ones((3, 4)) * 2)
