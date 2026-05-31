from enum import StrEnum
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from neurolens.config import NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.dataset.text import TextDataset
from neurolens.evaluation import Evaluation
from neurolens.img_text_model import ImageTextModel
from neurolens.score_function import ScoreFunction
from neurolens.target_model import TargetModel, load_activations
from neurolens.target_model.neuron_data import BatchedNeuronData
from neurolens.utils.path_utils import PathConfigs

from .cosy_config import CoSyConfig
from .stable_diffusion_img_generator import StableDiffusionImageGenerator


class CoSyEvaluationColumnNames(StrEnum):
    PROMPT = "prompt"
    DMA_SCORE = "dma"
    MAX_SCORE = "max"
    AUC_SCORE = "auc"


class CoSyEvaluation(Evaluation):
    def __init__(
        self,
        *,
        config: NeuroLensConfig,
        cosy_config: CoSyConfig,
        path_configs: PathConfigs,
        target_model: TargetModel,
        img_text_model: ImageTextModel,
        img_dataset: ImageDataset,
        text_dataset: TextDataset,
        control_img_dataset: ImageDataset,
        device: str | torch.device | None = None,
    ) -> None:

        if device is None:
            device = target_model.device

        super().__init__(
            config=config,
            path_configs=path_configs,
            target_model=target_model,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            text_dataset=text_dataset,
            device=device,
        )

        self.cosy_config = cosy_config
        self.control_img_dataset = control_img_dataset

        self.img_generator = StableDiffusionImageGenerator(
            config=self.config,
            cosy_config=self.cosy_config,
            path_configs=self.path_configs,
            device=self.device,
        )

    def get_control_activations(self, neuron_indices: list[int]) -> Float[Tensor, "n_img n_neuron"]:

        return load_activations(
            config=self.config,
            path_configs=self.path_configs,
            target_model=self.target_model,
            img_dataset=self.control_img_dataset,
            neuron_indices=neuron_indices,
            device=self.device,
        )

    def get_score_function_save_filepath(self, score_function: ScoreFunction) -> Path:
        return (
            self.get_save_basepath()
            / self.cosy_config.results_dir_name.format(
                img_generator=self.cosy_config.stable_diffusion_model_identifier,
                sample_count=self.config.evaluation.sample_count,
                normalize_activations=self.cosy_config.normalize_activations,
            )
            / self.cosy_config.results_filename.format(score_function=score_function.get_label())
        )

    def evaluate_from_computed_text_scores(
        self,
        score_function: ScoreFunction,
        neuron_data: BatchedNeuronData,
        text_scores: Float[Tensor, "n_neuron topk_text"],
        text_indices: Int[Tensor, "n_neuron topk_text"],
    ) -> None:

        self.all_control_activations = self.get_control_activations(neuron_data.neuron_indices)

        neuron_data = neuron_data.to(self.device)

        if self.cosy_config.normalize_activations:
            max_positive_values = neuron_data.activation_values.max(dim=1, keepdim=True).values
        else:
            max_positive_values = torch.ones((len(neuron_data), 1), device=self.device)

        results: dict[str, list] = {
            self.NEURON_INDEX_COLUMN: [],
            CoSyEvaluationColumnNames.PROMPT: [],
            CoSyEvaluationColumnNames.DMA_SCORE: [],
            CoSyEvaluationColumnNames.MAX_SCORE: [],
            CoSyEvaluationColumnNames.AUC_SCORE: [],
        }

        pbar = tqdm(range(len(neuron_data)))
        for i in pbar:
            neuron_idx = neuron_data.neuron_indices[i]
            pbar.set_description(f"CoSy evaluation for neuron {neuron_idx!r}")

            prompt_idx = int(text_indices[i][0].item())
            prompt = self.text_dataset[prompt_idx]

            if self.cosy_config.enable_img_generation:
                self.img_generator.generate_images(prompt=prompt)

            prompt_activations = self.get_prompt_activations(
                prompt=prompt,
                neuron_idx=neuron_idx,
            )

            control_activations = self.all_control_activations[:, i]

            prompt_mean = prompt_activations.mean()

            # following the equations under https://arxiv.org/abs/2405.20331
            AUC = (control_activations[:, None] < prompt_activations[None, :]).float().mean().item()

            DMA = (prompt_mean / max_positive_values[i]).item()
            MAX_SCORE = (prompt_activations.max().item() / max_positive_values[i]).item()

            results[self.NEURON_INDEX_COLUMN].append(neuron_idx)
            results[CoSyEvaluationColumnNames.PROMPT].append(prompt)
            results[CoSyEvaluationColumnNames.DMA_SCORE].append(DMA)
            results[CoSyEvaluationColumnNames.MAX_SCORE].append(MAX_SCORE)
            results[CoSyEvaluationColumnNames.AUC_SCORE].append(AUC)

        self.save_to_csv(
            score_function=score_function,
            results=results,
        )

    def get_prompt_activations(
        self,
        prompt: str,
        neuron_idx: int,
    ) -> Float[Tensor, " generated_img_count"]:

        images = self.img_generator.load_images(prompt)
        if len(images) == 0:
            return torch.zeros(
                (self.cosy_config.generated_img_count),
                device=self.target_model.device,
            )

        return self.target_model.get_activations(images)[:, neuron_idx]
