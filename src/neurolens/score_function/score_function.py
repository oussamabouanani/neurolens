from abc import ABC, abstractmethod

import torch
from jaxtyping import Float, Int
from torch import Tensor

from neurolens.config import NeuroLensConfig
from neurolens.dataset.text import TextDataset
from neurolens.img_text_model import ImageTextModel, load_text_embds_avg_templates
from neurolens.target_model.neuron_data import BatchedNeuronData
from neurolens.utils.path_utils import PathConfigs


class ScoreFunction(ABC):
    def __init__(
        self,
        *,
        config: NeuroLensConfig,
        path_configs: PathConfigs,
        img_text_model: ImageTextModel,
        text_dataset: TextDataset,
    ) -> None:

        self.config = config
        self.topk_text = config.evaluation.topk_text

        self.path_configs = path_configs
        self.img_text_model = img_text_model
        self.text_dataset = text_dataset

    @abstractmethod
    def get_label(self) -> str:
        raise NotImplementedError

    def get_text_embds(
        self,
        device: str | torch.device,
        templates: str | list[str] | None = None,
    ) -> Float[Tensor, "n_text d_embd"]:

        return load_text_embds_avg_templates(
            config=self.config,
            path_configs=self.path_configs,
            img_text_model=self.img_text_model,
            text_dataset=self.text_dataset,
            device=device,
            templates=templates,
        )

    @abstractmethod
    def compute_text_scores(
        self, neuron_data: BatchedNeuronData
    ) -> tuple[Float[Tensor, "n_neuron topk_text"], Int[Tensor, "n_neuron topk_text"]]:
        raise NotImplementedError
