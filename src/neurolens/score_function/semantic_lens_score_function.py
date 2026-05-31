import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from neurolens.config import NeuroLensConfig
from neurolens.dataset.text import TextDataset
from neurolens.img_text_model import ImageTextModel
from neurolens.target_model.neuron_data import BatchedNeuronData
from neurolens.utils.path_utils import PathConfigs

from .score_function import ScoreFunction


class SemanticLensScoreFunction(ScoreFunction):
    def __init__(
        self,
        config: NeuroLensConfig,
        path_configs: PathConfigs,
        img_text_model: ImageTextModel,
        text_dataset: TextDataset,
    ) -> None:
        super().__init__(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            text_dataset=text_dataset,
        )

    def get_label(self) -> str:
        return "semantic_lens"

    def compute_text_scores(
        self, neuron_data: BatchedNeuronData
    ) -> tuple[Float[Tensor, "n_neuron topk_text"], Int[Tensor, "n_neuron topk_text"]]:

        text_embds = self.get_text_embds(device=neuron_data.device)

        pos_means = neuron_data.pos_embds.mean(dim=1)
        pos_means = F.normalize(pos_means, p=2, dim=-1)

        sims = pos_means @ text_embds.T

        return torch.topk(sims, k=self.topk_text, dim=1, largest=True, sorted=True)
