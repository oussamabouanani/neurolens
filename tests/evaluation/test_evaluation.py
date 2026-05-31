from pathlib import Path

import pandas as pd
import pytest

from neurolens.config import IOConfig, NeuroLensConfig
from neurolens.evaluation import Evaluation
from neurolens.score_function import ScoreFunction
from neurolens.utils.path_utils import PathConfigs


class DummyScoreFunction(ScoreFunction):
    def __init__(self, config):
        self.config = config
        self.topk_text = config.evaluation.topk_text

    def get_label(self) -> str:
        return "dummy-score"

    def compute_text_scores(self, neuron_data):
        raise NotImplementedError


class DummyEvaluation(Evaluation):
    def get_score_function_save_filepath(self, score_function: ScoreFunction) -> Path:
        return self.path_configs.join("results", f"{score_function.get_label()}.csv")

    def evaluate_from_computed_text_scores(self, *, score_function, neuron_data, text_scores, text_indices) -> None:
        raise NotImplementedError


class DummyTargetModel:
    identifier = "target"
    device = "cpu"


class DummyIdentifiedObject:
    def __init__(self, identifier):
        self.identifier = identifier


@pytest.fixture
def evaluation(tmp_path):
    config = NeuroLensConfig(io=IOConfig(root_data_dir_path=tmp_path))
    return DummyEvaluation(
        config=config,
        path_configs=PathConfigs(config),
        target_model=DummyTargetModel(),
        img_text_model=DummyIdentifiedObject("img-text"),
        img_dataset=DummyIdentifiedObject("images"),
        text_dataset=DummyIdentifiedObject("texts"),
        device="cpu",
    )


def test_get_save_basepath_includes_model_and_dataset_identifiers(evaluation):
    assert evaluation.get_save_basepath() == (
        evaluation.config.io.root_data_dir_path / "data_results/img-text/target/images-texts"
    )


def test_save_to_csv_requires_neuron_index_column(evaluation):
    score_function = DummyScoreFunction(evaluation.config)

    with pytest.raises(ValueError, match="Column neuron_idx not found"):
        evaluation.save_to_csv(
            score_function=score_function,
            results={"score": [0.5]},
        )


def test_save_and_load_csv_round_trips_results(evaluation):
    score_function = DummyScoreFunction(evaluation.config)
    results = {
        Evaluation.NEURON_INDEX_COLUMN: [7, 3],
        "prompt": ["cat", "dog"],
        "score": [0.5, 1.25],
    }

    evaluation.save_to_csv(score_function=score_function, results=results)
    loaded = evaluation.load_from_csv(score_function)

    pd.testing.assert_frame_equal(loaded, pd.DataFrame(results))


def test_load_from_csv_filters_requested_neuron_indices(evaluation):
    score_function = DummyScoreFunction(evaluation.config)
    evaluation.save_to_csv(
        score_function=score_function,
        results={
            Evaluation.NEURON_INDEX_COLUMN: [7, 3, 11],
            "score": [0.5, 1.25, -2.0],
        },
    )

    loaded = evaluation.load_from_csv(score_function, neuron_indices=[3, 11])

    assert loaded[Evaluation.NEURON_INDEX_COLUMN].tolist() == [3, 11]
    assert loaded["score"].tolist() == [1.25, -2.0]


def test_load_from_csv_rejects_missing_requested_neuron_indices(evaluation):
    score_function = DummyScoreFunction(evaluation.config)
    evaluation.save_to_csv(
        score_function=score_function,
        results={Evaluation.NEURON_INDEX_COLUMN: [7], "score": [0.5]},
    )

    with pytest.raises(ValueError, match="Not all neuron indices"):
        evaluation.load_from_csv(score_function, neuron_indices=[7, 99])


def test_load_from_csv_requires_existing_file(evaluation):
    score_function = DummyScoreFunction(evaluation.config)

    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        evaluation.load_from_csv(score_function)
