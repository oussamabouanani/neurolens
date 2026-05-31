from types import SimpleNamespace

import pytest
import torch

from neurolens.config import EvaluationConfig, NeuroLensConfig
from neurolens.score_function import ContrastiveProjectionScoreFunction


def make_config(topk_text=2):
    return NeuroLensConfig(evaluation=EvaluationConfig(topk_text=topk_text))


def make_score_function(topk_text=2, gamma=0.5):
    return ContrastiveProjectionScoreFunction(
        config=make_config(topk_text=topk_text),
        path_configs=SimpleNamespace(),
        img_text_model=SimpleNamespace(),
        text_dataset=SimpleNamespace(),
        gamma=gamma,
    )


def test_contrastive_projection_score_function_validates_gamma():
    with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
        make_score_function(gamma=-0.1)

    with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
        make_score_function(gamma=1.1)


def test_contrastive_projection_score_function_accepts_boundary_gammas():
    assert make_score_function(gamma=0).gamma == 0
    assert make_score_function(gamma=1).gamma == 1


def test_contrastive_projection_score_function_label():
    score_function = make_score_function(gamma=0.125)

    assert score_function.get_label() == "contrastive_projection-gamma=0.125"


def test_contrastive_projection_score_function_computes_topk_scores():
    score_function = make_score_function(topk_text=2, gamma=0.5)
    neuron_data = SimpleNamespace(
        device=torch.device("cpu"),
        pos_embds=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        ),
        neg_embds=torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
        similarity_values=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        activation_values=torch.tensor(
            [
                [3.0, 1.0],
                [1.0, 3.0],
            ]
        ),
    )
    text_embds = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        dim=-1,
    )
    score_function.get_text_embds = lambda device, templates=None: text_embds.to(device)

    scores, indices = score_function.compute_text_scores(neuron_data)

    assert torch.equal(indices, torch.tensor([[0, 2], [0, 2]]))
    assert torch.allclose(
        scores,
        torch.tensor([[0.9486833, 0.8944272], [0.9284767, 0.9191450]]),
    )


def test_contrastive_projection_score_function_gamma_changes_projection():
    neuron_data = SimpleNamespace(
        device=torch.device("cpu"),
        pos_embds=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        neg_embds=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        similarity_values=torch.tensor([[1.0, 1.0]]),
        activation_values=torch.tensor([[1.0, 1.0]]),
    )
    text_embds = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, -1.0],
            ]
        ),
        dim=-1,
    )

    no_projection_score_function = make_score_function(topk_text=1, gamma=0.0)
    no_projection_score_function.get_text_embds = lambda device, templates=None: text_embds.to(device)
    full_projection_score_function = make_score_function(topk_text=1, gamma=1.0)
    full_projection_score_function.get_text_embds = lambda device, templates=None: text_embds.to(device)

    no_projection_scores, no_projection_indices = no_projection_score_function.compute_text_scores(neuron_data)
    full_projection_scores, full_projection_indices = full_projection_score_function.compute_text_scores(neuron_data)

    assert torch.equal(no_projection_indices, torch.tensor([[0]]))
    assert torch.allclose(no_projection_scores, torch.tensor([[1.0]]))
    assert torch.equal(full_projection_indices, torch.tensor([[1]]))
    assert torch.allclose(full_projection_scores, torch.tensor([[1.0]]))
