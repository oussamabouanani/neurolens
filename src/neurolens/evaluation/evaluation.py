import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from neurolens.config import NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.dataset.text import TextDataset
from neurolens.img_text_model import ImageTextModel
from neurolens.score_function import ScoreFunction
from neurolens.target_model import TargetModel
from neurolens.target_model.neuron_data import BatchedNeuronData
from neurolens.utils.path_utils import PathConfigs


class Evaluation(ABC):
    NEURON_INDEX_COLUMN = "neuron_idx"

    def __init__(
        self,
        *,
        config: NeuroLensConfig,
        path_configs: PathConfigs,
        target_model: TargetModel,
        img_text_model: ImageTextModel,
        img_dataset: ImageDataset,
        text_dataset: TextDataset,
        device: str | torch.device,
    ) -> None:

        self.config = config
        self.path_configs = path_configs
        self.target_model = target_model
        self.img_text_model = img_text_model
        self.img_dataset = img_dataset
        self.text_dataset = text_dataset
        self.device = torch.device(device)

    def get_save_basepath(self) -> Path:
        return self.path_configs.join(
            self.config.io.results_data_dir_name,
            self.img_text_model.identifier,
            self.target_model.identifier,
            f"{self.img_dataset.identifier}-{self.text_dataset.identifier}",
        )

    @abstractmethod
    def get_score_function_save_filepath(self, score_function: ScoreFunction) -> Path:
        raise NotImplementedError

    def evaluate(
        self,
        *,
        score_function: ScoreFunction,
        neuron_data: BatchedNeuronData,
    ) -> None:

        text_scores, text_indices = score_function.compute_text_scores(neuron_data)

        self.evaluate_from_computed_text_scores(
            score_function=score_function,
            neuron_data=neuron_data,
            text_scores=text_scores,
            text_indices=text_indices,
        )

    @abstractmethod
    def evaluate_from_computed_text_scores(
        self,
        *,
        score_function: ScoreFunction,
        neuron_data: BatchedNeuronData,
        text_scores: Float[Tensor, "n_neuron topk_text"],
        text_indices: Int[Tensor, "n_neuron topk_text"],
    ) -> None:
        raise NotImplementedError

    def save_to_csv(
        self,
        *,
        score_function: ScoreFunction,
        results: dict[str, list],
    ) -> None:

        csv_path = self.get_score_function_save_filepath(score_function=score_function)

        df = pd.DataFrame(results)
        if self.NEURON_INDEX_COLUMN not in df.columns:
            raise ValueError(
                f"Column {self.NEURON_INDEX_COLUMN} not found"
                f" in the CSV file {csv_path} for score function {score_function.get_label()}"
            )

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    def load_from_csv(self, score_function: ScoreFunction, neuron_indices: list[int] | None = None) -> pd.DataFrame:
        csv_path = self.get_score_function_save_filepath(score_function=score_function)

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path} for score function {score_function.get_label()}")

        df = pd.read_csv(csv_path)
        if neuron_indices is not None:
            df_slice = df[df[self.NEURON_INDEX_COLUMN].isin(neuron_indices)]
            if len(df_slice) != len(neuron_indices):
                raise ValueError(
                    f"Not all neuron indices {neuron_indices} are found"
                    f" in the CSV file {csv_path} for score function {score_function.get_label()}"
                )
            return df_slice

        return df
