import logging
import os

import einops
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm

from neurolens.config import NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.img_text_model import ImageTextModel, load_img_embds
from neurolens.target_model import TargetModel, load_activations
from neurolens.utils.path_utils import PathConfigs

from .neuron_data import BatchedNeuronData, NeuronData, NeuronDataSampleType

logger = logging.getLogger(__name__)


@torch.inference_mode()
def _get_topk_positive_neuron_data(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_dataset: ImageDataset,
    device: str | torch.device | None,
    ignore_input_indices: Int[Tensor, " n_indices"] | None = None,
    input_neurons: list[int] | None = None,
    eps: float = 1e-6,
) -> dict[int, NeuronData]:

    if device is None:
        device = target_model.device

    topk_neurons = config.neuron_data.topk_neurons
    required_sample_count = config.neuron_data.required_sample_count
    mean_top_count = config.neuron_data.mean_top_count
    ignore_overly_active_neurons = config.neuron_data.ignore_overly_active_neurons
    overly_active_neuron_rate = config.neuron_data.overly_active_neuron_rate

    start_idx = 0
    batch_size = 4069

    if mean_top_count > len(img_dataset):
        raise ValueError(f"mean_top_count ({mean_top_count}) must be <= len(img_dataset) ({len(img_dataset)})")

    ignore_sample_indices_mask = torch.zeros(len(img_dataset), device=device, dtype=torch.bool)
    if ignore_input_indices is not None:
        ignore_sample_indices_mask[ignore_input_indices.to(device)] = True

    top_activations = torch.zeros(
        (mean_top_count, target_model.get_total_neuron_count()),
        device=device,
        dtype=torch.float,
    )
    overall_means = torch.zeros(target_model.get_total_neuron_count(), device=device, dtype=torch.float)
    firing_count = torch.zeros(target_model.get_total_neuron_count(), device=device, dtype=torch.int)

    pbar = tqdm(range(0, len(img_dataset), batch_size))
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, len(img_dataset))

        pbar.set_description(f"Processing {start_idx}-{end_idx} image samples of {len(img_dataset)}")

        activations: Float[Tensor, "n_imgs n_neurons"] = load_activations(
            config=config,
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
            device=device,
            sample_indices=(start_idx, end_idx),
        )

        activations[ignore_sample_indices_mask[start_idx:end_idx], :] = 0

        batch_top = torch.topk(activations, mean_top_count, dim=0, sorted=False).values
        top_activations = torch.topk(
            torch.cat(
                [top_activations, batch_top],
                dim=0,
            ),
            mean_top_count,
            sorted=False,
            dim=0,
        ).values

        overall_means += activations.sum(dim=0)
        firing_count += (activations > eps).int().sum(dim=0)

    valid_sample_count = len(img_dataset) - int(ignore_sample_indices_mask.sum().item())
    if required_sample_count > valid_sample_count:
        raise ValueError(
            f"required_sample_count ({required_sample_count}) > valid_sample_count ({valid_sample_count})."
            " This means that there are not enough images in the dataset"
            " to satisfy the amount of required positive samples."
        )
    top_means = top_activations.mean(dim=0)
    overall_means /= valid_sample_count
    firing_rate = firing_count.float() / valid_sample_count

    excluded_neurons: Bool[Tensor, " n_neurons"] = torch.zeros(
        target_model.get_total_neuron_count(), device=device, dtype=torch.bool
    )

    if input_neurons is not None:
        selected_neurons = torch.zeros_like(excluded_neurons, dtype=torch.bool)
        selected_neurons[input_neurons] = True

        excluded_neurons |= ~selected_neurons

    excluded_neurons |= firing_count < required_sample_count

    if ignore_overly_active_neurons:
        excluded_neurons |= firing_rate >= overly_active_neuron_rate

    live_neuron_count = int((~excluded_neurons).sum().item())
    if topk_neurons > live_neuron_count:
        raise ValueError(f"topk_neurons ({topk_neurons}) > live_neuron_count ({live_neuron_count}).")

    top_means[excluded_neurons] = -float("inf")

    _, top_k_neuron_indices = torch.topk(
        top_means,
        k=topk_neurons,
        largest=True,
        sorted=True,
    )

    if torch.any(excluded_neurons[top_k_neuron_indices]):
        raise ValueError(
            f"Neurons {top_k_neuron_indices[excluded_neurons[top_k_neuron_indices]]}"
            " are dead but listed in top_k_neuron_indices."
        )

    activations: Float[Tensor, "n_imgs n_neurons"] = load_activations(
        config=config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
        device=device,
        neuron_indices=top_k_neuron_indices.tolist(),
    )
    activations[ignore_sample_indices_mask, :] = -float("inf")

    topk_vals, topk_img_indices = torch.topk(activations, k=required_sample_count, dim=0, largest=True, sorted=True)
    topk_vals: Float[Tensor, "required_sample_count topk_neurons"]
    topk_img_indices: Int[Tensor, "required_sample_count topk_neurons"]

    results = dict[int, NeuronData]()
    for j, idx in enumerate(top_k_neuron_indices.tolist()):
        neuron_idx = int(idx)
        vals = topk_vals[:, j]
        idxs = topk_img_indices[:, j]

        results[neuron_idx] = NeuronData(
            neuron_idx=neuron_idx,
            indices=idxs,
            activation_values=vals,
            similarity_values=torch.ones_like(vals),
            probe_dataset_overall_mean=overall_means[neuron_idx].item(),
        )

    logger.info(
        f"[target model {target_model.identifier}, img_dataset {img_dataset.identifier}]"
        f" Found {len(results)} positive neuron data points"
    )

    return results


@torch.inference_mode()
def _get_negative_neuron_data(
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    img_dataset: ImageDataset,
    target_model: TargetModel,
    pos_neuron_data: NeuronData,
    device: str | torch.device | None,
    all_img_embds: Float[Tensor, "n_img d_embd"],
    ignore_input_indices: Int[Tensor, " n_indices"] | None = None,
) -> NeuronData:

    batch_size = config.neuron_data.neg_fetch_batch_size
    allow_reuse = config.neuron_data.allow_neg_reuse

    neuron_idx = pos_neuron_data.neuron_idx

    activations: Float[Tensor, " n_imgs"] = load_activations(
        config=config,
        path_configs=path_configs,
        img_dataset=img_dataset,
        target_model=target_model,
        neuron_indices=neuron_idx,
        device=device,
    ).squeeze(dim=1)

    neg_mask = activations < pos_neuron_data.probe_dataset_overall_mean
    if ignore_input_indices is not None:
        neg_mask[ignore_input_indices] = False
    neg_indices = torch.nonzero(neg_mask, as_tuple=True)[0]
    neg_embds = all_img_embds[neg_indices]

    pos_embds = all_img_embds[pos_neuron_data.indices]

    num_neg = neg_embds.size(0)
    num_pos = pos_embds.size(0)

    if num_neg == 0:
        raise ValueError(f"No negative samples available for neuron {neuron_idx}")

    if not allow_reuse and num_pos > num_neg:
        raise ValueError(
            f"allow_reuse == False but num_pos ({num_pos}) > num_neg ({num_neg})."
            " This setup does not allow for fetching enough negative samples."
        )

    used_neg_mask = torch.zeros(num_neg, dtype=torch.bool, device=device)
    out_indices_local = torch.empty((num_pos,), dtype=torch.long, device=device)
    similarity_values = torch.empty((num_pos,), dtype=neg_embds.dtype, device=device)

    for pos_idx in range(num_pos):
        pos_vec: Float[Tensor, " d_embd"] = pos_embds[pos_idx]

        best_val = torch.tensor(-float("inf"), device=device)
        best_idx_local = torch.tensor(0, dtype=torch.long, device=device)

        for start_idx in range(0, num_neg, batch_size):
            end_idx = min(start_idx + batch_size, num_neg)

            neg_embds_batch: Float[Tensor, "neg_batch d_embd"] = neg_embds[start_idx:end_idx]
            sims = einops.einsum(
                pos_vec,
                neg_embds_batch,
                "d_embd, neg_batch d_embd -> neg_batch",
            )

            if not allow_reuse and used_neg_mask[start_idx:end_idx].any():
                sims = sims.masked_fill(used_neg_mask[start_idx:end_idx], -float("inf"))

            batch_val, batch_idx_local = sims.max(dim=0)

            if batch_val > best_val:
                best_val = batch_val
                best_idx_local = batch_idx_local + start_idx

        used_neg_mask[best_idx_local] = True
        out_indices_local[pos_idx] = best_idx_local
        similarity_values[pos_idx] = best_val

    final_neg_indices = neg_indices[out_indices_local]

    return NeuronData(
        neuron_idx=neuron_idx,
        indices=final_neg_indices,
        activation_values=activations[final_neg_indices],
        similarity_values=similarity_values,
        probe_dataset_overall_mean=pos_neuron_data.probe_dataset_overall_mean,
    )


def is_neuron_data_precomputed(
    *,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
) -> bool:

    for sample_type in NeuronDataSampleType:
        csv_path = path_configs.data_results_neuron_data_file_path(
            target_model=target_model,
            img_dataset=img_dataset,
            img_text_model=img_text_model,
            sample_type=sample_type,
        )

        if not os.path.exists(csv_path):
            return False

    return True


def precompute_neuron_data(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
    device: str | torch.device | None = None,
    ignore_input_indices: Int[Tensor, " n_indices"] | None = None,
    input_neurons: list[int] | None = None,
):

    pos_neuron_data: dict[int, NeuronData] = _get_topk_positive_neuron_data(
        config=config,
        path_configs=path_configs,
        target_model=target_model,
        img_dataset=img_dataset,
        ignore_input_indices=ignore_input_indices,
        input_neurons=input_neurons,
        device=device,
    )

    all_img_embds = load_img_embds(
        config=config,
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        device=device,
    )

    neg_neuron_data = dict[int, NeuronData]()
    pbar = tqdm(pos_neuron_data.keys(), desc="Fetching negative samples")
    for neuron_idx in pbar:
        neg_neuron_data[neuron_idx] = _get_negative_neuron_data(
            config=config,
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
            all_img_embds=all_img_embds,
            pos_neuron_data=pos_neuron_data[neuron_idx],
            ignore_input_indices=ignore_input_indices,
            device=device,
        )

    for neuron_data, sample_type in zip(
        [pos_neuron_data, neg_neuron_data],
        [NeuronDataSampleType.POSITIVE, NeuronDataSampleType.NEGATIVE],
    ):
        save_neuron_data_to_csv(
            path_configs=path_configs,
            target_model=target_model,
            img_dataset=img_dataset,
            img_text_model=img_text_model,
            neuron_data=neuron_data,
            sample_type=sample_type,
        )


def save_neuron_data_to_csv(
    *,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
    neuron_data: dict[int, NeuronData],
    sample_type: NeuronDataSampleType,
) -> None:

    values: dict[str, list] = {
        "sample_type": [],
        "neuron_idx": [],
        "sample_rank": [],
        "img_idx": [],
        "activation_value": [],
        "similarity_value": [],
        "probe_dataset_overall_mean": [],
    }

    for neuron_idx, data in neuron_data.items():
        indices = data.indices.detach().cpu().tolist()
        activation_values = data.activation_values.detach().cpu().tolist()
        similarity_values = data.similarity_values.detach().cpu().tolist()

        for sample_rank, (img_idx, activation_value, similarity_value) in enumerate(
            zip(indices, activation_values, similarity_values)
        ):
            values["sample_type"].append(sample_type)
            values["neuron_idx"].append(neuron_idx)
            values["sample_rank"].append(sample_rank)
            values["img_idx"].append(int(img_idx))
            values["activation_value"].append(float(activation_value))
            values["similarity_value"].append(float(similarity_value))
            values["probe_dataset_overall_mean"].append(float(data.probe_dataset_overall_mean))

    csv_path = path_configs.data_results_neuron_data_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
        img_text_model=img_text_model,
        sample_type=sample_type,
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(values)
    df.to_csv(csv_path, index=False)


def load_neuron_data_from_csv(
    *,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
    sample_type: NeuronDataSampleType,
    neuron_indices: int | list[int] | None = None,
    device: str | torch.device | None = None,
) -> tuple[dict[int, NeuronData], list[int]]:

    csv_path = path_configs.data_results_neuron_data_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
        img_text_model=img_text_model,
        sample_type=sample_type,
    )

    df_main = pd.read_csv(csv_path)

    if isinstance(neuron_indices, int):
        neuron_indices = [neuron_indices]

    neuron_data = dict[int, NeuronData]()

    if neuron_indices is None:
        neuron_indices = df_main["neuron_idx"].unique().tolist()

    for neuron_idx in neuron_indices:
        df = df_main[(df_main["neuron_idx"] == neuron_idx) & (df_main["sample_type"] == sample_type)].sort_values(
            "sample_rank"
        )

        if df.empty:
            raise ValueError(f"No {sample_type!r} neuron data found for neuron_idx={neuron_idx}")

        neuron_data[neuron_idx] = NeuronData(
            neuron_idx=neuron_idx,
            indices=torch.tensor(df["img_idx"].to_numpy(), dtype=torch.long, device=device),
            activation_values=torch.tensor(
                df["activation_value"].to_numpy(),
                dtype=torch.float32,
                device=device,
            ),
            similarity_values=torch.tensor(
                df["similarity_value"].to_numpy(),
                dtype=torch.float32,
                device=device,
            ),
            probe_dataset_overall_mean=float(df["probe_dataset_overall_mean"].iloc[0]),
        )

    return neuron_data, neuron_indices


def load_batched_neuron_data(
    *,
    config: NeuroLensConfig,
    path_configs: PathConfigs,
    target_model: TargetModel,
    img_text_model: ImageTextModel,
    img_dataset: ImageDataset,
    neuron_indices: list[int] | None = None,
    device: str | torch.device | None = None,
    precompute_if_missing: bool = True,
) -> BatchedNeuronData:

    if (
        not is_neuron_data_precomputed(
            path_configs=path_configs,
            target_model=target_model,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
        )
        and precompute_if_missing
    ):
        precompute_neuron_data(
            config=config,
            path_configs=path_configs,
            target_model=target_model,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            device=device,
        )

    # 0: positive neuron data, 1: negative
    neuron_data = []

    for sample_type in (NeuronDataSampleType.POSITIVE, NeuronDataSampleType.NEGATIVE):
        neuron_data_dict, neuron_indices = load_neuron_data_from_csv(
            path_configs=path_configs,
            target_model=target_model,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            neuron_indices=neuron_indices,
            sample_type=sample_type,
            device=device,
        )

        neuron_data.append([neuron_data_dict[idx] for idx in neuron_indices])

    return BatchedNeuronData(
        config=config,
        pos_neuron_data_list=neuron_data[0],
        neg_neuron_data_list=neuron_data[1],
        path_configs=path_configs,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        device=device,
    )
