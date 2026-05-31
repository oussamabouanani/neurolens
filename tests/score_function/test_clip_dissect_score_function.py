from types import SimpleNamespace

import pytest
import torch

from neurolens.config import DatasetConfig, EvaluationConfig, NeuroLensConfig
from neurolens.score_function import CLIPDissectScoreFunction


class SizedImageDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def make_config(topk_text=2):
    return NeuroLensConfig(evaluation=EvaluationConfig(topk_text=topk_text))


def make_score_function(
    topk_text=2,
    temp=10,
    lambda_weight=1,
    img_dataset=None,
    config=None,
    prior_batch_size=4096,
):
    return CLIPDissectScoreFunction(
        config=config or make_config(topk_text=topk_text),
        path_configs=SimpleNamespace(),
        img_text_model=SimpleNamespace(),
        text_dataset=SimpleNamespace(),
        img_dataset=img_dataset or SizedImageDataset(1),
        temp=temp,
        lambda_weight=lambda_weight,
        prior_batch_size=prior_batch_size,
    )


def test_clip_dissect_score_function_label():
    score_function = make_score_function(temp=0.5, lambda_weight=0.25)

    assert score_function.get_label() == "clip_dissect-temp=0.5-lambda=0.25"


def test_clip_dissect_score_function_computes_prior_probs_when_missing(monkeypatch):
    score_function = make_score_function()
    neuron_data = SimpleNamespace(
        device=torch.device("cpu"),
        pos_embds=torch.tensor([[[1.0, 0.0]]]),
        activation_values=torch.ones((1, 1)),
    )
    text_embds = torch.eye(2)

    monkeypatch.setattr(
        score_function,
        "_compute_prior_probs",
        lambda device: torch.tensor([0.5, 0.5], device=device),
    )
    score_function.get_text_embds = lambda device, templates=None: text_embds.to(device)

    scores, indices = score_function.compute_text_scores(neuron_data)

    assert score_function.prior_probs is not None
    assert scores.shape == (1, 2)
    assert indices.shape == (1, 2)


def test_clip_dissect_score_function_validates_prior_prob_count():
    score_function = make_score_function()
    score_function.prior_probs = torch.ones(3)
    neuron_data = SimpleNamespace(
        device=torch.device("cpu"),
        pos_embds=torch.ones((1, 1, 2)),
        activation_values=torch.ones((1, 1)),
    )
    score_function.get_text_embds = lambda device, templates=None: torch.ones((2, 2), device=device)

    with pytest.raises(ValueError, match="does not match number of prior probabilities"):
        score_function.compute_text_scores(neuron_data)


def test_clip_dissect_score_function_computes_topk_scores():
    score_function = make_score_function(topk_text=2, temp=1.0, lambda_weight=1.0)
    score_function.prior_probs = torch.tensor([0.5, 0.25, 0.25])
    neuron_data = SimpleNamespace(
        device=torch.device("cpu"),
        pos_embds=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, 0.0]],
            ]
        ),
        activation_values=torch.tensor(
            [
                [3.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )
    text_embds = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    score_function.get_text_embds = lambda device, templates=None: text_embds.to(device)

    scores, indices = score_function.compute_text_scores(neuron_data)

    probs = torch.softmax(neuron_data.pos_embds @ text_embds.T, dim=-1)
    weights = neuron_data.activation_values
    soft_prod = (weights[:, :, None] * probs).sum(dim=1) / weights.sum(dim=1, keepdim=True)
    expected = torch.log(soft_prod + 1e-6) - torch.log(score_function.prior_probs[None, :])
    expected_scores, expected_indices = torch.topk(expected, k=2, dim=1, largest=True, sorted=True)

    assert torch.equal(indices, expected_indices)
    assert torch.allclose(scores, expected_scores)


def test_clip_dissect_score_function_respects_topk_text_config():
    score_function = make_score_function(topk_text=1, temp=1.0)
    score_function.prior_probs = torch.tensor([0.5, 0.25, 0.25])
    neuron_data = SimpleNamespace(
        device=torch.device("cpu"),
        pos_embds=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        activation_values=torch.tensor([[1.0, 1.0]]),
    )
    text_embds = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    score_function.get_text_embds = lambda device, templates=None: text_embds.to(device)

    scores, indices = score_function.compute_text_scores(neuron_data)

    assert scores.shape == (1, 1)
    assert indices.shape == (1, 1)


def test_compute_prior_probs_averages_probabilities_over_image_dataset(monkeypatch):
    config = NeuroLensConfig(
        dataset=DatasetConfig(vanilla_template="a {}"),
        evaluation=EvaluationConfig(topk_text=2),
    )
    score_function = make_score_function(
        config=config,
        temp=1.0,
        img_dataset=SizedImageDataset(3),
        prior_batch_size=2,
    )
    img_embds = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    text_embds = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    loaded_img_indices = []
    loaded_templates = []

    def fake_load_text_embds_avg_templates(**kwargs):
        loaded_templates.append(kwargs["templates"])
        return text_embds.to(kwargs["device"])

    def fake_load_img_embds(**kwargs):
        loaded_img_indices.append(kwargs["img_indices"])
        return img_embds[kwargs["img_indices"]].to(kwargs["device"])

    monkeypatch.setattr(
        "neurolens.score_function.clip_dissect_score_function.load_text_embds_avg_templates",
        fake_load_text_embds_avg_templates,
    )
    monkeypatch.setattr(
        "neurolens.score_function.clip_dissect_score_function.load_img_embds",
        fake_load_img_embds,
    )

    prior_probs = score_function._compute_prior_probs(device="cpu")

    expected = torch.cat(
        [
            torch.softmax(img_embds[[0, 1]] @ text_embds.T, dim=1),
            torch.softmax(img_embds[[2]] @ text_embds.T, dim=1),
        ],
        dim=0,
    ).mean(dim=0)
    assert loaded_templates == ["a {}"]
    assert loaded_img_indices == [[0, 1], [2]]
    assert torch.allclose(prior_probs, expected)
