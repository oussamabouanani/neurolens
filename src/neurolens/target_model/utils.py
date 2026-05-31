import logging
import os

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from neurolens.config import NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.precompute_utils import precompute_rows
from neurolens.utils.zarr_utils import get_zarr_shape, zarr_load_to_torch

from .target_model import TargetModel

logger = logging.getLogger(__name__)


def load_activations(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_dataset: ImageDataset,
    device: str | torch.device | None = None,
    sample_indices: int | list[int] | tuple[int, int] | None = None,
    neuron_indices: int | list[int] | tuple[int, int] | None = None,
    precompute_if_missing: bool = True,
) -> Float[Tensor, "n_img n_neuron"]:

    if device is None:
        device = target_model.device

    activations_file_path = path_configs.data_precomp_target_activations_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
    )

    if (
        not is_activations_precomputed(
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
        )
        and precompute_if_missing
    ):
        precompute_activations(
            config=config,
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
        )

    return zarr_load_to_torch(
        store=activations_file_path,
        validate_shape=(len(img_dataset), target_model.get_total_neuron_count()),
        device=device,
        rows=sample_indices,
        columns=neuron_indices,
    )


@torch.inference_mode()
def precompute_activations(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_dataset: ImageDataset,
):

    batch_save_count = config.io.zarr_batch_save_count
    batch_size = config.dataset.precomp_batch_size

    activations_file_path = path_configs.data_precomp_target_activations_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
    )

    dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, collate_fn=list)

    precompute_rows(
        save_path=activations_file_path,
        dataloader=dataloader,
        total_rows=len(img_dataset),
        compute_batch=lambda image_batch: target_model.get_activations(images=image_batch),
        batch_save_count=batch_save_count,
        desc=f"[target model {target_model.identifier}, image dataset {img_dataset.identifier}]"
        " Precomputing activations",
        log_saved_chunk=lambda chunk_idx: (
            f"[target model {target_model.identifier}, image dataset {img_dataset.identifier}]"
            " Precomputed and saved activations for image chunk {chunk_idx=:02d}"
        ),
    )


def is_activations_precomputed(
    *,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_dataset: ImageDataset,
):
    activations_file_path = path_configs.data_precomp_target_activations_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
    )
    if not os.path.exists(activations_file_path):
        return False

    return get_zarr_shape(activations_file_path) == (
        len(img_dataset),
        target_model.get_total_neuron_count(),
    )
