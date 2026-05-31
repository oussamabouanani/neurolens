from pathlib import Path

import pytest
import torch
from PIL import Image

from neurolens import DatasetConfig, IOConfig, NeuroLensConfig
from neurolens.target_model import (
    is_activations_precomputed,
    load_activations,
    precompute_activations,
)
from neurolens.target_model.target_model import TargetModel
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.torch_utils import ImageProcessor
from neurolens.utils.zarr_utils import zarr_save


def image_to_tensor_processor(image):
    return torch.zeros((3, image.height, image.width))


class IndexedImageDataset:
    def __init__(self, identifier, length):
        self.identifier = identifier
        self.images = [Image.new("RGB", (5, 7), color=(idx, idx, idx)) for idx in range(length)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class CountingTargetModel(TargetModel):
    def __init__(self, identifier="target-id", device="cpu"):
        super().__init__(identifier, device)
        self.batch_calls = []
        self.next_row = 0

    def get_img_processor(self) -> ImageProcessor:
        return image_to_tensor_processor

    def get_total_neuron_count(self) -> int:
        return 3

    def _get_activations(self, img_features):
        batch_len = img_features.shape[0]
        row_ids = torch.arange(
            self.next_row,
            self.next_row + batch_len,
            dtype=torch.float32,
            device=self.device,
        )
        self.next_row += batch_len
        self.batch_calls.append(batch_len)

        return torch.stack(
            [row_ids, row_ids + 10.0, row_ids + 20.0],
            dim=1,
        )


@pytest.fixture
def path_configs(tmp_path):
    return PathConfigs(
        NeuroLensConfig(
            io=IOConfig(root_data_dir_path=tmp_path, zarr_batch_save_count=1),
            dataset=DatasetConfig(precomp_batch_size=2),
        )
    )


def activations_path(path_configs, target_model, img_dataset) -> Path:
    return path_configs.data_precomp_target_activations_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
    )


def test_is_activations_precomputed_reflects_cache_existence(path_configs):
    target_model = CountingTargetModel()
    img_dataset = IndexedImageDataset("images", length=2)

    assert not is_activations_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )

    precompute_activations(
        config=path_configs.config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )

    assert is_activations_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )


def test_is_activations_precomputed_rejects_wrong_shape(path_configs):
    target_model = CountingTargetModel()
    img_dataset = IndexedImageDataset("images", length=2)

    zarr_save(
        activations_path(path_configs, target_model, img_dataset),
        shape=(len(img_dataset) + 1, target_model.get_total_neuron_count()),
        dtype=torch.float32,
        chunks=(1, target_model.get_total_neuron_count()),
    )

    assert not is_activations_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )


def test_load_activations_precomputes_missing_cache(path_configs):
    target_model = CountingTargetModel()
    img_dataset = IndexedImageDataset("images", length=3)

    activations = load_activations(
        config=path_configs.config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
        device="cpu",
    )

    assert is_activations_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )
    assert torch.equal(
        activations,
        torch.tensor(
            [
                [0.0, 10.0, 20.0],
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
            ]
        ),
    )


def test_load_activations_can_disable_missing_cache_precompute(path_configs):
    target_model = CountingTargetModel()
    img_dataset = IndexedImageDataset("images", length=3)

    with pytest.raises(Exception):
        load_activations(
            config=path_configs.config,
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
            precompute_if_missing=False,
        )

    assert not is_activations_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )


def test_precompute_activations_saves_expected_activation_rows(path_configs):
    target_model = CountingTargetModel()
    img_dataset = IndexedImageDataset("images", length=5)

    precompute_activations(
        config=path_configs.config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )

    expected = torch.tensor(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
            [3.0, 13.0, 23.0],
            [4.0, 14.0, 24.0],
        ]
    )

    assert target_model.batch_calls == [2, 2, 1]
    assert activations_path(path_configs, target_model, img_dataset).exists()
    assert torch.equal(
        load_activations(
            config=path_configs.config,
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
            device="cpu",
        ),
        expected,
    )


def test_load_activations_selects_requested_neuron_columns(path_configs):
    target_model = CountingTargetModel()
    img_dataset = IndexedImageDataset("images", length=3)

    precompute_activations(
        config=path_configs.config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )

    activations = load_activations(
        config=path_configs.config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
        device=torch.device("cpu"),
        neuron_indices=[0, 2],
    )

    assert torch.equal(
        activations,
        torch.tensor(
            [
                [0.0, 20.0],
                [1.0, 21.0],
                [2.0, 22.0],
            ]
        ),
    )


def test_load_activations_selects_requested_samples_and_neurons(path_configs):
    target_model = CountingTargetModel()
    img_dataset = IndexedImageDataset("images", length=5)

    precompute_activations(
        config=path_configs.config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
    )

    assert torch.equal(
        load_activations(
            config=path_configs.config,
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
            device="cpu",
            sample_indices=[0, 2, 4],
            neuron_indices=[1, 2],
        ),
        torch.tensor(
            [
                [10.0, 20.0],
                [12.0, 22.0],
                [14.0, 24.0],
            ]
        ),
    )
    assert torch.equal(
        load_activations(
            config=path_configs.config,
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
            device="cpu",
            sample_indices=(1, 4),
            neuron_indices=(0, 2),
        ),
        torch.tensor(
            [
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
            ]
        ),
    )


def test_load_activations_validates_expected_dataset_shape(path_configs):
    target_model = CountingTargetModel()
    original_dataset = IndexedImageDataset("images", length=3)
    resized_dataset = IndexedImageDataset("images", length=2)

    precompute_activations(
        config=path_configs.config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=original_dataset,
    )

    with pytest.raises(ValueError, match="does not match expected shape"):
        load_activations(
            config=path_configs.config,
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=resized_dataset,
            device="cpu",
        )


def test_precompute_activations_rejects_existing_cache_with_wrong_shape(path_configs):
    target_model = CountingTargetModel()
    original_dataset = IndexedImageDataset("images", length=3)
    resized_dataset = IndexedImageDataset("images", length=2)

    precompute_activations(
        config=path_configs.config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=original_dataset,
    )

    with pytest.raises(ValueError, match="does not match expected shape"):
        precompute_activations(
            config=path_configs.config,
            path_configs=path_configs,
            target_model=CountingTargetModel(),
            img_dataset=resized_dataset,
        )
