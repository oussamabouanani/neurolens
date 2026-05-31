from dataclasses import dataclass

import torch
from PIL import Image

from neurolens.config import NeuroLensConfig, VLMConfig
from neurolens.vlm import VLMTextGenerationMode, VLMWrapper


class RecordingImageDataset:
    def __init__(self, images):
        self.images = images
        self.requested_indices = []

    def __getitem__(self, index):
        self.requested_indices.append(index)
        return self.images[index]


@dataclass
class DummyBatchedNeuronData:
    neuron_indices: list[int]
    pos_sample_indices: torch.Tensor
    neg_sample_indices: torch.Tensor

    def __len__(self):
        return len(self.neuron_indices)


class DummyVLMWrapper(VLMWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_only_grid_sizes = []
        self.pos_neg_grid_sizes = []

    def generate_pos_only_text(self, pos_grid):
        self.pos_only_grid_sizes.append(pos_grid.size)
        return self.clean_output("Cat!, , very long generated phrase, red car")

    def generate_pos_neg_text(self, pos_grid, neg_grid):
        self.pos_neg_grid_sizes.append((pos_grid.size, neg_grid.size))
        return ["contrastive_label"]


def test_clean_output_normalizes_filters_empty_and_long_text():
    wrapper = DummyVLMWrapper(
        identifier="dummy",
        config=NeuroLensConfig(vlm=VLMConfig(max_text_length=8)),
        img_dataset=RecordingImageDataset([]),
        device="cpu",
    )

    assert wrapper.clean_output(" Cat!, , red car, very long phrase \n") == [
        "cat",
        "red_car",
    ]


def test_generated_text_uses_configured_image_count_for_each_mode():
    images = [Image.new("RGB", (6, 6), (index, index, index)) for index in range(12)]
    dataset = RecordingImageDataset(images)
    wrapper = DummyVLMWrapper(
        identifier="dummy",
        config=NeuroLensConfig(
            vlm=VLMConfig(
                img_count=4,
                grid_row_count=2,
                grid_size=8,
                max_text_length=8,
            )
        ),
        img_dataset=dataset,
        device="cpu",
    )
    neuron_data = DummyBatchedNeuronData(
        neuron_indices=[7],
        pos_sample_indices=torch.tensor([[0, 1, 2, 3, 4, 5]]),
        neg_sample_indices=torch.tensor([[6, 7, 8, 9, 10, 11]]),
    )

    pos_only = wrapper.generated_text(neuron_data, mode=VLMTextGenerationMode.POS_ONLY)
    pos_neg = wrapper.generated_text(neuron_data, mode=VLMTextGenerationMode.POS_NEG)

    assert pos_only == ["cat", "red_car"]
    assert pos_neg == ["contrastive_label"]
    assert dataset.requested_indices == [
        0,
        1,
        2,
        3,
        6,
        7,
        8,
        9,
        0,
        1,
        2,
        3,
        6,
        7,
        8,
        9,
    ]
    assert wrapper.pos_only_grid_sizes == [(8, 8)]
    assert wrapper.pos_neg_grid_sizes == [((8, 8), (8, 8))]
