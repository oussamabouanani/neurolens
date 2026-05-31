import logging
import os

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from neurolens.config import NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.dataset.text import TextDataset
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.precompute_utils import precompute_rows
from neurolens.utils.zarr_utils import get_zarr_shape, zarr_load_to_torch

from .dataset_utils import load_img_embds, load_text_embds_avg_templates
from .img_text_model import ImageTextModel

logger = logging.getLogger(__name__)


def load_similarity_matrix(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
    text_dataset: TextDataset,
    device: str | torch.device,
    img_indices: int | list[int] | tuple[int, int] | None = None,
    text_indices: int | list[int] | tuple[int, int] | None = None,
    precompute_if_missing: bool = True,
):

    sim_mat_file_path = path_configs.data_precomp_imgtext_sim_matrix_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    if (
        not is_similarity_matrix_precomputed(
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            text_dataset=text_dataset,
        )
        and precompute_if_missing
    ):
        precompute_similarity_matrix(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            text_dataset=text_dataset,
        )

    return zarr_load_to_torch(
        store=sim_mat_file_path,
        validate_shape=(len(img_dataset), len(text_dataset)),
        device=device,
        rows=img_indices,
        columns=text_indices,
    )


@torch.inference_mode()
def precompute_similarity_matrix(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
    text_dataset: TextDataset,
):

    batch_save_count = config.io.zarr_batch_save_count
    batch_size = config.dataset.precomp_batch_size

    if config.dataset.similarity_matrix_use_vanilla_template:
        templates = config.dataset.vanilla_template
    else:
        templates = config.dataset.templates

    text_embds: Float[Tensor, "n_text d_embd"] = load_text_embds_avg_templates(
        config=config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        text_dataset=text_dataset,
        templates=templates,
    )

    img_embds: Float[Tensor, "n_img d_embd"] = load_img_embds(
        config=config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )

    sim_mat_file_path = path_configs.data_precomp_imgtext_sim_matrix_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )

    dataloader = DataLoader(TensorDataset(img_embds), batch_size=batch_size, shuffle=False)

    precompute_rows(
        save_path=sim_mat_file_path,
        dataloader=dataloader,
        total_rows=len(img_dataset),
        compute_batch=lambda img_embds_batch: img_text_model.get_similarities(
            # because TensorDataset returns a batch of size 1
            img=img_embds_batch[0],
            text=text_embds,
        ),
        batch_save_count=batch_save_count,
        desc=f"[img text model {img_text_model.identifier}, image dataset {img_dataset.identifier},"
        f" text dataset {text_dataset.identifier}]"
        " Precomputing similarity matrix ",
        log_saved_chunk=lambda chunk: (
            f"[img text model {img_text_model.identifier}, image dataset {img_dataset.identifier},"
            f" text dataset {text_dataset.identifier}]"
            f" Precomputed and saved similarity matrix for image chunk {chunk=:02d}"
        ),
    )


def is_similarity_matrix_precomputed(
    *,
    path_configs: PathConfigs,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
    text_dataset: TextDataset,
):
    sim_mat_file_path = path_configs.data_precomp_imgtext_sim_matrix_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    )
    if not os.path.exists(sim_mat_file_path):
        return False

    return get_zarr_shape(sim_mat_file_path) == (
        len(img_dataset),
        len(text_dataset),
    )
