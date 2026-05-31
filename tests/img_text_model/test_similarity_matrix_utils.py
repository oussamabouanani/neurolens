import math

import pytest
import torch
from PIL import Image

from neurolens import DatasetConfig, IOConfig, NeuroLensConfig
from neurolens.dataset.text import TextDataset
from neurolens.img_text_model import ImageTextModel
from neurolens.img_text_model.dataset_utils import (
    precompute_img_embds,
    precompute_text_embds,
)
from neurolens.img_text_model.similarity_matrix_utils import (
    is_similarity_matrix_precomputed,
    load_similarity_matrix,
    precompute_similarity_matrix,
)
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.zarr_utils import zarr_save


class ImageDataset:
    identifier = "images"

    def __init__(self, length=3):
        self.images = [Image.new("RGB", (5, 7)) for _ in range(length)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class SimilarityImageTextModel(ImageTextModel):
    def __init__(self, identifier="clip-id", device="cpu"):
        super().__init__(identifier, device)
        self.next_img_row = 0

    def get_embd_dim(self) -> int:
        return 2

    def get_img_embds(self, images):
        batch_len = len(images) if isinstance(images, list) else 1
        rows = []
        for idx in range(self.next_img_row, self.next_img_row + batch_len):
            if idx % 3 == 0:
                rows.append([1.0, 0.0])
            elif idx % 3 == 1:
                rows.append([0.0, 1.0])
            else:
                rows.append([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
        self.next_img_row += batch_len
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def get_text_embds(self, text):
        text_list = text if isinstance(text, list) else [text]
        rows = []
        for next_text in text_list:
            if next_text in {"alpha", "synonym alpha"}:
                rows.append([1.0, 0.0])
            elif next_text in {"beta", "synonym beta"}:
                rows.append([0.0, 1.0])
            else:
                rows.append([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
        return torch.tensor(rows, dtype=torch.float32, device=self.device)


@pytest.fixture
def path_configs(tmp_path):
    return PathConfigs(
        NeuroLensConfig(
            io=IOConfig(root_data_dir_path=tmp_path, zarr_batch_save_count=1),
            dataset=DatasetConfig(precomp_batch_size=2),
        )
    )


def test_is_similarity_matrix_precomputed_reflects_cache_existence(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    assert not is_similarity_matrix_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    precompute_img_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
    precompute_text_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )
    precompute_similarity_matrix(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    assert is_similarity_matrix_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )


def test_is_similarity_matrix_precomputed_rejects_wrong_shape(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])
    sim_mat_file_path = path_configs.data_precomp_imgtext_sim_matrix_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    zarr_save(
        sim_mat_file_path,
        shape=(len(img_dataset) + 1, len(text_dataset)),
        dtype=torch.float32,
        chunks=(1, len(text_dataset)),
    )

    assert not is_similarity_matrix_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )


def test_load_similarity_matrix_precomputes_missing_cache(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    sim_matrix = load_similarity_matrix(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
        device="cpu",
    )

    assert is_similarity_matrix_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )
    assert torch.allclose(
        sim_matrix,
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],
            ]
        ),
    )


def test_load_similarity_matrix_can_disable_missing_cache_precompute(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    with pytest.raises(Exception):
        load_similarity_matrix(
            config=path_configs.config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            text_dataset=text_dataset,
            device="cpu",
            precompute_if_missing=False,
        )

    assert not is_similarity_matrix_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )


def test_precompute_similarity_matrix_precomputes_missing_text_embeddings(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    precompute_img_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )

    precompute_similarity_matrix(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    assert is_similarity_matrix_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )


def test_precompute_similarity_matrix_precomputes_missing_image_embeddings(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    precompute_text_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )

    precompute_similarity_matrix(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    assert is_similarity_matrix_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )


def test_precompute_similarity_matrix_saves_expected_similarities(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    precompute_img_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
    precompute_text_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )
    precompute_similarity_matrix(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    assert torch.allclose(
        load_similarity_matrix(
            config=path_configs.config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            text_dataset=text_dataset,
            device="cpu",
        ),
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],
            ]
        ),
    )


def test_precompute_similarity_matrix_uses_configured_templates_when_requested(
    tmp_path,
):
    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(root_data_dir_path=tmp_path, zarr_batch_save_count=1),
            dataset=DatasetConfig(
                precomp_batch_size=2,
                templates=["{}", "synonym {}"],
                similarity_matrix_use_vanilla_template=False,
            ),
        )
    )
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    precompute_img_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
    precompute_text_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )
    precompute_similarity_matrix(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    assert torch.allclose(
        load_similarity_matrix(
            config=path_configs.config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            text_dataset=text_dataset,
            device="cpu",
        ),
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],
            ]
        ),
    )


def test_load_similarity_matrix_selects_requested_image_and_text_indices(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset()
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    precompute_img_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
    precompute_text_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )
    precompute_similarity_matrix(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    assert torch.allclose(
        load_similarity_matrix(
            config=path_configs.config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            text_dataset=text_dataset,
            device="cpu",
            img_indices=[0, 2],
            text_indices=1,
        ),
        torch.tensor([[0.0], [1.0 / math.sqrt(2)]]),
    )


def test_load_similarity_matrix_validates_expected_dataset_shape(path_configs):
    img_text_model = SimilarityImageTextModel()
    img_dataset = ImageDataset(length=3)
    resized_img_dataset = ImageDataset(length=2)
    text_dataset = TextDataset("labels", ["alpha", "beta"])

    precompute_img_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
    precompute_text_embds(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )
    precompute_similarity_matrix(
        config=path_configs.config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    with pytest.raises(ValueError, match="does not match expected shape"):
        load_similarity_matrix(
            config=path_configs.config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=resized_img_dataset,
            text_dataset=text_dataset,
            device="cpu",
        )
