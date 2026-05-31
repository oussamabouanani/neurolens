import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor

from neurolens.config import NeuroLensConfig
from neurolens.dataset.text import TextDataset
from neurolens.img_text_model import ImageTextModel
from neurolens.target_model.neuron_data import BatchedNeuronData
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.torch_utils import get_weighted_average

from .score_function import ScoreFunction


class ContrastiveProjectionScoreFunction(ScoreFunction):
    def __init__(
        self,
        config: NeuroLensConfig,
        path_configs: PathConfigs,
        img_text_model: ImageTextModel,
        text_dataset: TextDataset,
        gamma: float,
    ) -> None:
        super().__init__(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            text_dataset=text_dataset,
        )

        if gamma < 0 or gamma > 1:
            raise ValueError("gamma must be between 0 and 1")

        self.gamma = gamma

    def get_label(self) -> str:
        return f"contrastive_projection-gamma={self.gamma}"

    def compute_text_scores(
        self, neuron_data: BatchedNeuronData
    ) -> tuple[Float[Tensor, "n_neuron topk_text"], Int[Tensor, "n_neuron topk_text"]]:

        text_embds = self.get_text_embds(device=neuron_data.device)

        pos_embds, neg_embds, similarities, weights = (
            neuron_data.pos_embds,
            neuron_data.neg_embds,
            neuron_data.similarity_values,
            neuron_data.activation_values,
        )

        projections = einops.einsum(
            similarities,
            neg_embds,
            "n_neuron n_sample, n_neuron n_sample d_embd -> n_neuron n_sample d_embd",
        )

        residuals = pos_embds - self.gamma * projections

        concept_vectors = get_weighted_average(residuals, weights, weighting="raw", l2_normalise=True)

        text_sims = concept_vectors @ text_embds.T

        return torch.topk(text_sims, k=self.topk_text, dim=1, largest=True, sorted=True)
