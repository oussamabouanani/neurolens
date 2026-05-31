import logging
from math import ceil

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from neurolens.config import VANILLA_TEMPLATE, NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.dataset.text import TextDataset
from neurolens.img_text_model import (
    ImageTextModel,
    load_img_embds,
    load_text_embds_avg_templates,
)
from neurolens.target_model.neuron_data import BatchedNeuronData
from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.torch_utils import transform_weights

from .score_function import ScoreFunction

logger = logging.getLogger(__name__)


class CLIPDissectScoreFunction(ScoreFunction):
    def __init__(
        self,
        config: NeuroLensConfig,
        path_configs: PathConfigs,
        img_text_model: ImageTextModel,
        text_dataset: TextDataset,
        img_dataset: ImageDataset,
        temp: float = 10,
        lambda_weight: float = 1,
        prior_batch_size: int = 4096,
    ) -> None:
        super().__init__(
            config=config,
            path_configs=path_configs,
            img_text_model=img_text_model,
            text_dataset=text_dataset,
        )

        self.temp = temp
        self.lambda_weight = lambda_weight
        self.prior_probs: Float[Tensor, " n_text"] | None = None

        self.img_dataset = img_dataset
        self.prior_batch_size = prior_batch_size

    def get_label(self) -> str:
        return f"clip_dissect-temp={self.temp}-lambda={self.lambda_weight}"

    def _compute_prior_probs(
        self,
        device: str | torch.device,
    ) -> Float[Tensor, " n_text"]:

        text_embds = load_text_embds_avg_templates(
            config=self.config,
            path_configs=self.path_configs,
            img_text_model=self.img_text_model,
            text_dataset=self.text_dataset,
            device=device,
            templates=self.config.dataset.vanilla_template,
        )

        prior_probs = torch.zeros(text_embds.size(0), dtype=torch.float32, device=device)
        n_imgs = len(self.img_dataset)
        batch_count = ceil(n_imgs / self.prior_batch_size)

        pbar = tqdm(range(batch_count), desc="Computing prior probabilities (CLIP Dissect)")

        for batch_idx in pbar:
            start_img_idx = batch_idx * self.prior_batch_size
            end_img_idx = min(start_img_idx + self.prior_batch_size, n_imgs)

            img_indices = list(range(start_img_idx, end_img_idx))

            img_embds = load_img_embds(
                config=self.config,
                path_configs=self.path_configs,
                img_text_model=self.img_text_model,
                img_dataset=self.img_dataset,
                device=device,
                img_indices=img_indices,
            )

            prior_probs += F.softmax((img_embds @ text_embds.T) / self.temp, dim=1).sum(dim=0)

        return prior_probs / n_imgs

    def compute_text_scores(
        self, neuron_data: BatchedNeuronData
    ) -> tuple[Float[Tensor, "n_neuron topk_text"], Int[Tensor, "n_neuron topk_text"]]:

        if self.prior_probs is None:
            logger.info("Computing prior probabilities for CLIP Dissect...")
            self.prior_probs = self._compute_prior_probs(neuron_data.device)

        text_embds = self.get_text_embds(device=neuron_data.device, templates=VANILLA_TEMPLATE)

        if len(text_embds) != len(self.prior_probs):
            raise ValueError(
                f"Number of text embeddings ({len(text_embds)}) does not match number"
                " of prior probabilities ({len(self.prior_probs)}). Did you change the text dataset?"
            )

        self.prior_probs = self.prior_probs.to(neuron_data.activation_values.device)

        probs = F.softmax(neuron_data.pos_embds @ text_embds.T / self.temp, dim=-1)
        weights = transform_weights(neuron_data.activation_values, "raw")

        soft_prod = einops.einsum(
            weights,
            probs,
            "n_neuron n_sample, n_neuron n_sample n_text -> n_neuron n_text",
        ) / weights.sum(dim=1, keepdim=True)

        softwpmi = torch.log(soft_prod + 1e-6) - self.lambda_weight * torch.log(self.prior_probs[None, :])

        return torch.topk(softwpmi, k=self.topk_text, dim=1, largest=True, sorted=True)
