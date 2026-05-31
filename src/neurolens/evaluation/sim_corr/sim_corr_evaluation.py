import logging
import os
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
from neurolens.img_text_model import ImageTextModel, load_similarity_matrix
from neurolens.score_function import ScoreFunction
from neurolens.target_model import TargetModel, load_activations
from neurolens.target_model.neuron_data import BatchedNeuronData
from neurolens.utils.path_utils import PathConfigs

from .sim_corr_config import SimCorrConfig

logger = logging.getLogger(__name__)


class SimCorrEvaluationColumnNames(StrEnum):
    TEXT_TEMPLATE = "text-{index}"

    @staticmethod
    def get_text_column_name(text_index: int) -> str:
        return SimCorrEvaluationColumnNames.TEXT_TEMPLATE.format(index=text_index)

    WEIGHT_TEMPLATE = "weight-{index}"

    @staticmethod
    def get_weight_column_name(text_index: int) -> str:
        return SimCorrEvaluationColumnNames.WEIGHT_TEMPLATE.format(index=text_index)

    SIM_CORR_SCORE = "corr"


class SimCorrEvaluation(Evaluation):
    def __init__(
        self,
        *,
        config: NeuroLensConfig,
        sim_corr_config: SimCorrConfig,
        path_configs: PathConfigs,
        target_model: TargetModel,
        img_text_model: ImageTextModel,
        img_dataset: ImageDataset,
        text_dataset: TextDataset,
        simulator: ImageTextModel,
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

        self.sim_corr_config = sim_corr_config
        self.simulator = simulator

    def get_score_function_save_filepath(self, score_function: ScoreFunction) -> Path:
        return (
            self.get_save_basepath()
            / self.sim_corr_config.results_dir_name.format(
                simulator=self.simulator.identifier,
                sample_count=self.config.evaluation.sample_count,
                weighted=self.sim_corr_config.weighted,
                topk_text=self.sim_corr_config.topk_text,
            )
            / self.sim_corr_config.results_filename.format(score_function=score_function.get_label())
        )

    def evaluate_from_computed_text_scores(
        self,
        *,
        score_function: ScoreFunction,
        neuron_data: BatchedNeuronData,
        text_scores: Float[Tensor, "n_neuron topk_text"],
        text_indices: Int[Tensor, "n_neuron topk_text"],
    ) -> None:

        topk_text = self.sim_corr_config.topk_text

        if topk_text > text_scores.shape[1]:
            raise ValueError(f"topk_text ({topk_text}) must be <= text_scores.shape[1] ({text_scores.shape[1]})")
        if topk_text > text_indices.shape[1]:
            raise ValueError(f"topk_text ({topk_text}) must be <= text_indices.shape[1] ({text_indices.shape[1]})")

        test_split_path = self.path_configs.data_precomp_dataset_splits_file_path(
            img_dataset=self.img_dataset, split="test"
        )
        if os.path.isfile(test_split_path):
            eval_indices = torch.load(test_split_path, map_location=self.device)
        else:
            eval_indices = torch.arange(len(self.img_dataset), device=self.device)
            logger.warning(f"Test split indices not found at {test_split_path}. Using all indices!")

        eval_indices = eval_indices.tolist()

        sim_matrix = load_similarity_matrix(
            config=self.config,
            path_configs=self.path_configs,
            img_text_model=self.simulator,
            img_dataset=self.img_dataset,
            text_dataset=self.text_dataset,
            device=self.device,
            img_indices=eval_indices,
            text_indices=None,
        )

        activations = load_activations(
            config=self.config,
            path_configs=self.path_configs,
            target_model=self.target_model,
            img_dataset=self.img_dataset,
            device=self.device,
            sample_indices=eval_indices,
            neuron_indices=neuron_data.neuron_indices,
        )

        results: dict[str, list] = {
            self.NEURON_INDEX_COLUMN: [],
            SimCorrEvaluationColumnNames.SIM_CORR_SCORE: [],
        }

        for j in range(topk_text):
            results[SimCorrEvaluationColumnNames.get_text_column_name(text_index=j)] = []
            results[SimCorrEvaluationColumnNames.get_weight_column_name(text_index=j)] = []

        pbar = tqdm(range(len(neuron_data)))
        for i in pbar:
            neuron_idx = int(neuron_data.neuron_indices[i])
            pbar.set_description(f"SimCorr evaluation for neuron {neuron_idx!r}")

            concept_indices = text_indices[i][:topk_text]

            if self.sim_corr_config.weighted:
                concept_weights = text_scores[i][:topk_text]
            else:
                concept_weights = torch.ones(topk_text, device=self.device)

            pred_activations = sim_matrix[:, concept_indices] @ concept_weights
            target_activations = activations[:, i]

            corr_value = torch.corrcoef(torch.stack([target_activations, pred_activations], dim=0))[0, 1]

            if corr_value.isnan():
                raise ValueError(f"Correlation value for neuron {neuron_idx} is NaN!")

            results[self.NEURON_INDEX_COLUMN].append(neuron_idx)
            results[SimCorrEvaluationColumnNames.SIM_CORR_SCORE].append(corr_value.item())

            for j in range(topk_text):
                results[SimCorrEvaluationColumnNames.get_text_column_name(text_index=j)].append(
                    self.text_dataset[int(concept_indices[j].item())]
                )
                results[SimCorrEvaluationColumnNames.get_weight_column_name(text_index=j)].append(
                    concept_weights[j].item()
                )

        self.save_to_csv(
            score_function=score_function,
            results=results,
        )
