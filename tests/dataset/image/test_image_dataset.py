import pytest
from PIL import Image

from neurolens.dataset.image.image_dataset import ImageDataset


class DummyImageDataset(ImageDataset):
    def __init__(self):
        self.requested_indices = []
        super().__init__("dummy")

    def __len__(self):
        return 1

    def get_mean(self):
        return (0.1, 0.2, 0.3)

    def get_std(self):
        return (0.4, 0.5, 0.6)

    def __getitem__(self, idx):
        self.requested_indices.append(idx)
        return Image.new("RGB", (2, 3), color=(10, 20, 30))


def test_constructor_stores_identifier():
    dataset = DummyImageDataset()

    assert dataset.identifier == "dummy"


def test_getitem_returns_raw_pil_image_from_subclass():
    dataset = DummyImageDataset()

    image = dataset[7]

    assert dataset.requested_indices == [7]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (2, 3)


def test_get_mean_returns_dataset_mean():
    dataset = DummyImageDataset()

    assert dataset.get_mean() == (0.1, 0.2, 0.3)


def test_get_std_returns_dataset_std():
    dataset = DummyImageDataset()

    assert dataset.get_std() == (0.4, 0.5, 0.6)


def test_base_class_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="abstract class"):
        ImageDataset("base")
