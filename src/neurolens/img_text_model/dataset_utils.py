import logging
import os

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from neurolens.config import NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.dataset.text import TextDataset
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.precompute_utils import precompute_rows
from neurolens.utils.str_utils import validate_text_template
from neurolens.utils.zarr_utils import get_zarr_shape, zarr_load_to_torch

from .img_text_model import ImageTextModel

logger = logging.getLogger(__name__)


def load_img_embds(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
    device: str | torch.device | None = None,
    img_indices: int | list[int] | tuple[int, int] | None = None,
    precompute_if_missing: bool = True,
):

    if device is None:
        device = img_text_model.device

    embds_file_path = path_configs.data_precomp_imgtext_img_embds_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )

    if (
        not is_img_embds_precomputed(
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
        )
        and precompute_if_missing
    ):
        precompute_img_embds(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
        )

    return zarr_load_to_torch(
        store=embds_file_path,
        validate_shape=(len(img_dataset), img_text_model.get_embd_dim()),
        device=device,
        rows=img_indices,
    )


@torch.inference_mode()
def precompute_img_embds(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
):

    batch_save_count = config.io.zarr_batch_save_count
    batch_size = config.dataset.precomp_batch_size

    embds_file_path = path_configs.data_precomp_imgtext_img_embds_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )

    dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, collate_fn=list)

    precompute_rows(
        save_path=embds_file_path,
        dataloader=dataloader,
        total_rows=len(img_dataset),
        compute_batch=lambda image_batch: img_text_model.get_img_embds(
            images=image_batch,
        ),
        batch_save_count=batch_save_count,
        desc=f"[img text model {img_text_model.identifier}, image dataset {img_dataset.identifier}]"
        " Precomputing embeddings",
        log_saved_chunk=lambda chunk_idx: (
            f"[img text model {img_text_model.identifier}, image dataset {img_dataset.identifier}]"
            f"Precomputed and saved embds for image chunk {chunk_idx=:02d}"
        ),
    )


def is_img_embds_precomputed(
    *,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
) -> bool:
    embds_file_path = path_configs.data_precomp_imgtext_img_embds_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
    if not os.path.exists(embds_file_path):
        return False

    return get_zarr_shape(embds_file_path) == (
        len(img_dataset),
        img_text_model.get_embd_dim(),
    )


def load_text_embds_avg_templates(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    text_dataset: TextDataset,
    device: str | torch.device | None = None,
    templates: str | list[str] | None = None,
    text_indices: int | list[int] | tuple[int, int] | None = None,
) -> Float[Tensor, "n_text d_embd"]:

    return torch.mean(
        load_text_embds(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            text_dataset=text_dataset,
            device=device,
            text_indices=text_indices,
            templates=templates,
        ),
        dim=0,
    )


def load_text_embds(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    text_dataset: TextDataset,
    device: str | torch.device | None = None,
    templates: str | list[str] | None = None,
    text_indices: int | list[int] | tuple[int, int] | None = None,
    precompute_if_missing: bool = True,
) -> Float[Tensor, "n_templates n_text d_embd"]:

    if device is None:
        device = img_text_model.device

    if templates is None:
        templates = config.dataset.templates

    templates = validate_text_template(templates)

    if (
        not is_text_embds_precomputed(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            text_dataset=text_dataset,
            templates=templates,
        )
        and precompute_if_missing
    ):
        precompute_text_embds(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            text_dataset=text_dataset,
            templates=templates,
        )

    if text_indices is None:
        n_text = len(text_dataset)
    elif isinstance(text_indices, int):
        n_text = 1
    elif isinstance(text_indices, list):
        n_text = len(text_indices)
    else:
        n_text = text_indices[1] - text_indices[0]

    text_embds: Float[Tensor, "n_templates n_text d_embd"] = torch.zeros(
        len(templates), n_text, img_text_model.get_embd_dim(), device=device
    )

    for template_idx, next_template in enumerate(templates):
        embds_file_path = path_configs.data_precomp_imgtext_text_embds_file_path(
            img_text_model=img_text_model,
            text_dataset=text_dataset,
            template=next_template,
        )
        text_embds[template_idx, :, :] = zarr_load_to_torch(
            store=embds_file_path,
            validate_shape=(len(text_dataset), img_text_model.get_embd_dim()),
            device=device,
            rows=text_indices,
        )

    return text_embds


@torch.inference_mode()
def precompute_text_embds(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    text_dataset: TextDataset,
    templates: str | list[str] | None = None,
):

    batch_save_count = config.io.zarr_batch_save_count
    batch_size = config.dataset.precomp_batch_size

    if templates is None:
        templates = config.dataset.templates
    templates = validate_text_template(templates)

    for next_template in templates:
        embds_file_path = path_configs.data_precomp_imgtext_text_embds_file_path(
            img_text_model=img_text_model,
            text_dataset=text_dataset,
            template=next_template,
        )
        text_dataset.set_template(next_template)

        dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, collate_fn=list)

        precompute_rows(
            save_path=embds_file_path,
            dataloader=dataloader,
            total_rows=len(text_dataset),
            compute_batch=lambda text_batch: img_text_model.get_text_embds(
                text=text_batch,
            ),
            batch_save_count=batch_save_count,
            desc=f"[img text model {img_text_model.identifier}, text dataset {text_dataset.identifier}]"
            f"Precomputing embds for template {next_template.format('text')!r}",
            log_saved_chunk=lambda chunk_idx: (
                f"[img text model {img_text_model.identifier}, text dataset {text_dataset.identifier}]"
                f"Precomputed and saved embds for text chunk {chunk_idx=:02d}"
            ),
        )

    text_dataset.reset_template()


def is_text_embds_precomputed(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    text_dataset: TextDataset,
    templates: str | list[str] | None = None,
):
    if templates is None:
        templates = config.dataset.templates
    templates = validate_text_template(templates)

    for next_template in templates:
        embds_file_path = path_configs.data_precomp_imgtext_text_embds_file_path(
            img_text_model=img_text_model,
            text_dataset=text_dataset,
            template=next_template,
        )
        if not os.path.exists(embds_file_path):
            return False

        if get_zarr_shape(embds_file_path) != (
            len(text_dataset),
            img_text_model.get_embd_dim(),
        ):
            return False

    return True
