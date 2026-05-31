from types import SimpleNamespace

import torch

from neurolens.config import EvaluationConfig, NeuroLensConfig
from neurolens.score_function import SemanticLensScoreFunction


def make_config(topk_text=2):
    return NeuroLensConfig(evaluation=EvaluationConfig(topk_text=topk_text))


def make_score_function(topk_text=2):
    return SemanticLensScoreFunction(
        config=make_config(topk_text=topk_text),
        path_configs=SimpleNamespace(),
        img_text_model=SimpleNamespace(),
        text_dataset=SimpleNamespace(),
    )


def test_semantic_lens_score_function_label():
    assert make_score_function().get_label() == "semantic_lens"


def test_semantic_lens_score_function_computes_topk_scores():
    score_function = make_score_function(topk_text=2)
    neuron_data = SimpleNamespace(
        device=torch.device("cpu"),
        pos_embds=torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
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

    assert torch.equal(indices, torch.tensor([[0, 2], [2, 1]]))
    assert torch.allclose(
        scores,
        torch.tensor([[1.0, 2**-0.5], [0.9486833, 0.8944272]]),
    )


def test_semantic_lens_score_function_respects_topk_text_config():
    score_function = make_score_function(topk_text=1)
    neuron_data = SimpleNamespace(
        device=torch.device("cpu"),
        pos_embds=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
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

    assert scores.shape == (1, 1)
    assert indices.shape == (1, 1)
    assert torch.equal(indices, torch.tensor([[2]]))
    assert torch.allclose(scores, torch.tensor([[1.0]]))
