from types import SimpleNamespace

import pytest

from neurolens.config import EvaluationConfig, NeuroLensConfig
from neurolens.score_function import ScoreFunction


def make_config(topk_text=2):
    return NeuroLensConfig(evaluation=EvaluationConfig(topk_text=topk_text))


def test_score_function_base_class_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        ScoreFunction(
            config=make_config(),
            path_configs=SimpleNamespace(),
            img_text_model=SimpleNamespace(),
            text_dataset=SimpleNamespace(),
        )
