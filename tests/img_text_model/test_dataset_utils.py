import pytest
import torch
from PIL import Image

from neurolens import DatasetConfig, IOConfig, NeuroLensConfig
from neurolens.dataset.text import TextDataset
from neurolens.img_text_model import ImageTextModel
from neurolens.img_text_model.dataset_utils import (
    is_img_embds_precomputed,
    is_text_embds_precomputed,
    load_img_embds,
    load_text_embds,
    load_text_embds_avg_templates,
    precompute_img_embds,
    precompute_text_embds,
)
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.zarr_utils import zarr_save


class TemplateAwareImageTextModel(ImageTextModel):
    def __init__(self, identifier="clip-id", device="cpu"):
        super().__init__(identifier, device)
        self.text_batches = []
        self.next_img_row = 0

    def get_embd_dim(self) -> int:
        return 3

    def get_img_embds(self, images):
        batch_len = len(images) if isinstance(images, list) else 1
        row_ids = torch.arange(
            self.next_img_row,
            self.next_img_row + batch_len,
            dtype=torch.float32,
            device=self.device,
        )
        self.next_img_row += batch_len

        return torch.stack(
            [row_ids, row_ids + 10.0, row_ids + 20.0],
            dim=1,
        )

    def get_text_embds(self, text):
        text_list = text if isinstance(text, list) else [text]
        self.text_batches.append(text_list)

        return torch.tensor(
            [
                [
                    float(len(next_text)),
                    float(next_text.startswith("a photo of ")),
                    float(next_text.endswith(".")),
                ]
                for next_text in text_list
            ],
            dtype=torch.float32,
            device=self.device,
        )


@pytest.fixture
def path_configs(tmp_path):
    return PathConfigs(NeuroLensConfig(io=IOConfig(root_data_dir_path=tmp_path, zarr_batch_save_count=1)))


def config_for(path_configs, *, templates=None):
    return NeuroLensConfig(
        io=path_configs.config.io,
        dataset=DatasetConfig(
            templates=["{}"] if templates is None else templates,
            precomp_batch_size=2,
        ),
    )


class ImageDataset:
    identifier = "images"

    def __init__(self):
        self.images = [Image.new("RGB", (5, 7)) for _ in range(3)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def test_is_img_embds_precomputed_rejects_wrong_shape(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    img_dataset = ImageDataset()
    embds_file_path = path_configs.data_precomp_imgtext_img_embds_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )

    zarr_save(
        embds_file_path,
        shape=(len(img_dataset) - 1, img_text_model.get_embd_dim()),
        dtype=torch.float32,
        chunks=(1, img_text_model.get_embd_dim()),
    )

    assert not is_img_embds_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )


def test_load_img_embds_precomputes_missing_cache(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    img_dataset = ImageDataset()

    embds = load_img_embds(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        device="cpu",
    )

    assert is_img_embds_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
    assert torch.equal(
        embds,
        torch.tensor(
            [
                [0.0, 10.0, 20.0],
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
            ]
        ),
    )


def test_load_img_embds_can_disable_missing_cache_precompute(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    img_dataset = ImageDataset()

    with pytest.raises(Exception):
        load_img_embds(
            config=config_for(path_configs),
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            precompute_if_missing=False,
        )

    assert not is_img_embds_precomputed(
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )


def test_load_img_embds_selects_requested_image_rows(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    img_dataset = ImageDataset()

    precompute_img_embds(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )

    embds = load_img_embds(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        device="cpu",
        img_indices=[0, 2],
    )

    assert torch.equal(
        embds,
        torch.tensor(
            [
                [0.0, 10.0, 20.0],
                [2.0, 12.0, 22.0],
            ]
        ),
    )


def test_is_text_embds_precomputed_requires_all_templates(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    text_dataset = TextDataset("labels", ["zebra", "apple"])

    assert not is_text_embds_precomputed(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        templates=["{}", "a photo of {}"],
        text_dataset=text_dataset,
    )

    precompute_text_embds(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )

    assert is_text_embds_precomputed(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )
    assert not is_text_embds_precomputed(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        templates=["{}", "a photo of {}"],
        text_dataset=text_dataset,
    )


def test_is_text_embds_precomputed_rejects_wrong_shape(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    text_dataset = TextDataset("labels", ["zebra", "apple"])
    embds_file_path = path_configs.data_precomp_imgtext_text_embds_file_path(
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        template="{}",
    )

    zarr_save(
        embds_file_path,
        shape=(len(text_dataset) + 1, img_text_model.get_embd_dim()),
        dtype=torch.float32,
        chunks=(1, img_text_model.get_embd_dim()),
    )

    assert not is_text_embds_precomputed(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        templates="{}",
    )


def test_load_text_embds_precomputes_missing_templates(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    text_dataset = TextDataset("labels", ["zebra", "apple"])

    embds = load_text_embds(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        templates=["{}", "a photo of {}"],
        device="cpu",
    )

    assert is_text_embds_precomputed(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        templates=["{}", "a photo of {}"],
    )
    assert embds.shape == (2, 2, 3)
    assert img_text_model.text_batches == [
        ["apple", "zebra"],
        ["a photo of apple", "a photo of zebra"],
    ]


def test_load_text_embds_can_disable_missing_cache_precompute(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    text_dataset = TextDataset("labels", ["zebra", "apple"])

    with pytest.raises(Exception):
        load_text_embds(
            config=config_for(path_configs),
            path_configs=path_configs,
            img_text_model=img_text_model,
            text_dataset=text_dataset,
            templates="{}",
            precompute_if_missing=False,
        )

    assert not is_text_embds_precomputed(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        templates="{}",
    )


def test_precompute_text_embds_applies_each_template(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    text_dataset = TextDataset("labels", ["zebra", "apple"])

    precompute_text_embds(
        config=config_for(path_configs, templates=["{}", "a photo of {}"]),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )

    assert img_text_model.text_batches == [
        ["apple", "zebra"],
        ["a photo of apple", "a photo of zebra"],
    ]
    assert text_dataset[0] == "apple"


def test_load_text_embds_returns_template_text_embedding_stack(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    text_dataset = TextDataset("labels", ["zebra", "apple"])

    precompute_text_embds(
        config=config_for(path_configs, templates=["{}", "a photo of {}"]),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )

    embds = load_text_embds(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        templates=["{}", "a photo of {}"],
        device="cpu",
    )

    assert embds.shape == (2, 2, 3)
    assert torch.equal(
        embds,
        torch.tensor(
            [
                [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[16.0, 1.0, 0.0], [16.0, 1.0, 0.0]],
            ]
        ),
    )


def test_load_text_embds_selects_requested_text_rows(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    text_dataset = TextDataset("labels", ["cat", "horse", "ox"])

    precompute_text_embds(
        config=config_for(path_configs, templates=["{}", "a photo of {}"]),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )

    embds = load_text_embds(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        templates=["{}", "a photo of {}"],
        device="cpu",
        text_indices=[0, 2],
    )

    assert embds.shape == (2, 2, 3)
    assert torch.equal(
        embds,
        torch.tensor(
            [
                [[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[14.0, 1.0, 0.0], [13.0, 1.0, 0.0]],
            ]
        ),
    )


def test_load_text_embds_avg_templates_averages_template_dimension(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    text_dataset = TextDataset("labels", ["zebra", "apple"])

    precompute_text_embds(
        config=config_for(path_configs, templates=["{}", "a photo of {}"]),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
    )

    embds = load_text_embds_avg_templates(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        templates=["{}", "a photo of {}"],
        device="cpu",
    )

    assert embds.shape == (2, 3)
    assert torch.equal(
        embds,
        torch.tensor([[10.5, 0.5, 0.0], [10.5, 0.5, 0.0]]),
    )


def test_load_text_embds_validates_expected_dataset_shape(path_configs):
    img_text_model = TemplateAwareImageTextModel()
    original_dataset = TextDataset("labels", ["zebra", "apple", "banana"])
    resized_dataset = TextDataset("labels", ["zebra", "apple"])

    precompute_text_embds(
        config=config_for(path_configs),
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=original_dataset,
    )

    with pytest.raises(ValueError, match="does not match expected shape"):
        load_text_embds(
            config=config_for(path_configs),
            path_configs=path_configs,
            img_text_model=img_text_model,
            text_dataset=resized_dataset,
            templates="{}",
            device="cpu",
        )
