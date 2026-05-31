import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

import torch
from jaxtyping import Float, Int
from torch import Tensor

from neurolens.config import NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.img_text_model import ImageTextModel, load_img_embds
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.torch_utils import is_l2_normalized

logger = logging.getLogger(__name__)


class NeuronDataSampleType(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True, slots=True, eq=False)
class NeuronData:
    _FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "neuron_idx",
        "indices",
        "activation_values",
        "similarity_values",
        "probe_dataset_overall_mean",
    )

    _TENSOR_FIELDS: ClassVar[tuple[str, ...]] = (
        "indices",
        "activation_values",
        "similarity_values",
    )

    neuron_idx: int
    indices: Int[Tensor, " n_sample"]
    activation_values: Float[Tensor, " n_sample"]
    similarity_values: Float[Tensor, " n_sample"]
    probe_dataset_overall_mean: float

    @property
    def device(self) -> torch.device:
        devices = {getattr(self, field_name).device for field_name in self._TENSOR_FIELDS}

        if len(devices) != 1:
            raise RuntimeError(f"NeuronData tensors are on multiple devices: {devices}")

        return next(iter(devices))

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool = False,
    ) -> Self:
        device = torch.device(device)

        new_obj = object.__new__(type(self))

        for field_name in self._FIELD_NAMES:
            value = getattr(self, field_name)

            if field_name in self._TENSOR_FIELDS:
                value = value.to(device, non_blocking=non_blocking)

            object.__setattr__(new_obj, field_name, value)

        return new_obj


@dataclass(frozen=True, slots=True, init=False)
class BatchedNeuronData:
    _FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "neuron_indices",
        "pos_sample_indices",
        "neg_sample_indices",
        "activation_values",
        "neg_activation_values",
        "similarity_values",
        "probe_dataset_overall_mean",
        "pos_embds",
        "neg_embds",
    )

    _TENSOR_FIELDS: ClassVar[tuple[str, ...]] = (
        "pos_sample_indices",
        "neg_sample_indices",
        "activation_values",
        "neg_activation_values",
        "similarity_values",
        "pos_embds",
        "neg_embds",
    )

    neuron_indices: list[int]
    pos_sample_indices: Int[Tensor, "batch n_sample"]
    neg_sample_indices: Int[Tensor, "batch n_sample"]
    activation_values: Float[Tensor, "batch n_sample"]
    neg_activation_values: Float[Tensor, "batch n_sample"]
    similarity_values: Float[Tensor, "batch n_sample"]
    probe_dataset_overall_mean: tuple[float, ...]

    pos_embds: Float[Tensor, "batch n_sample d_embd"]
    neg_embds: Float[Tensor, "batch n_sample d_embd"]

    def __init__(
        self,
        *,
        config: NeuroLensConfig,
        pos_neuron_data_list: list[NeuronData],
        neg_neuron_data_list: list[NeuronData],
        path_configs: PathConfigs,
        img_text_model: ImageTextModel,
        img_dataset: ImageDataset,
        device: str | torch.device | None = None,
    ):
        sample_count = config.evaluation.sample_count

        if len(pos_neuron_data_list) != len(neg_neuron_data_list):
            raise ValueError(
                "pos_neuron_data_list and neg_neuron_data_list must have the same "
                f"length, but got {len(pos_neuron_data_list)} and "
                f"{len(neg_neuron_data_list)}"
            )

        neuron_indices = list(neuron_data.neuron_idx for neuron_data in pos_neuron_data_list)

        pos_sample_indices = torch.stack(
            [neuron_data.indices for neuron_data in pos_neuron_data_list],
            dim=0,
        )[:, :sample_count]

        neg_sample_indices = torch.stack(
            [neuron_data.indices for neuron_data in neg_neuron_data_list],
            dim=0,
        )[:, :sample_count]

        activation_values = torch.stack(
            [neuron_data.activation_values for neuron_data in pos_neuron_data_list],
            dim=0,
        )[:, :sample_count]

        neg_activation_values = torch.stack(
            [neuron_data.activation_values for neuron_data in neg_neuron_data_list],
            dim=0,
        )[:, :sample_count]

        similarity_values = torch.stack(
            [neuron_data.similarity_values for neuron_data in neg_neuron_data_list],
            dim=0,
        )[:, :sample_count]

        probe_dataset_overall_mean = tuple(
            neuron_data.probe_dataset_overall_mean for neuron_data in pos_neuron_data_list
        )

        all_embds = load_img_embds(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            device=device,
        )

        pos_embds = torch.stack(
            [all_embds[pos_sample_indices[i]] for i in range(len(neuron_indices))],
            dim=0,
        )

        neg_embds = torch.stack(
            [all_embds[neg_sample_indices[i]] for i in range(len(neuron_indices))],
            dim=0,
        )

        if not is_l2_normalized(pos_embds, dim=-1):
            raise ValueError("pos_embds in batch are not l2 normalized")

        if not is_l2_normalized(neg_embds, dim=-1):
            raise ValueError("neg_embds in batch are not l2 normalized")

        object.__setattr__(self, "neuron_indices", neuron_indices)
        object.__setattr__(self, "pos_sample_indices", pos_sample_indices)
        object.__setattr__(self, "neg_sample_indices", neg_sample_indices)
        object.__setattr__(self, "activation_values", activation_values)
        object.__setattr__(self, "neg_activation_values", neg_activation_values)
        object.__setattr__(self, "similarity_values", similarity_values)
        object.__setattr__(
            self,
            "probe_dataset_overall_mean",
            probe_dataset_overall_mean,
        )
        object.__setattr__(self, "pos_embds", pos_embds)
        object.__setattr__(self, "neg_embds", neg_embds)

    @property
    def device(self) -> torch.device:
        devices = {getattr(self, field_name).device for field_name in self._TENSOR_FIELDS}

        if len(devices) != 1:
            raise RuntimeError(f"BatchedNeuronData tensors are on multiple devices: {devices}")

        return next(iter(devices))

    def __len__(self) -> int:
        return len(self.neuron_indices)

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool = False,
    ) -> Self:
        device = torch.device(device)

        new_obj = object.__new__(type(self))

        for field_name in self._FIELD_NAMES:
            value = getattr(self, field_name)

            if field_name in self._TENSOR_FIELDS:
                value = value.to(device, non_blocking=non_blocking)

            object.__setattr__(new_obj, field_name, value)

        return new_obj
