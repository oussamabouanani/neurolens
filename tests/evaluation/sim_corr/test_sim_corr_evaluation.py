import pandas as pd
import pytest
import torch
from PIL import Image

from neurolens.config import EvaluationConfig, IOConfig, NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.dataset.text import TextDataset
from neurolens.evaluation import Evaluation
from neurolens.evaluation.sim_corr import (
    SimCorrConfig,
    SimCorrEvaluation,
    SimCorrEvaluationColumnNames,
)
from neurolens.img_text_model import ImageTextModel
from neurolens.score_function import ScoreFunction
from neurolens.target_model import TargetModel
from neurolens.utils.path_utils import PathConfigs


class DummyImageDataset(ImageDataset):
    def __init__(self):
        super().__init__("images")

    def __len__(self):
        return 4

    def get_mean(self):
        return (0.0, 0.0, 0.0)

    def get_std(self):
        return (1.0, 1.0, 1.0)

    def __getitem__(self, idx):
        return Image.new("RGB", (2, 2))


class DummyImageTextModel(ImageTextModel):
    def __init__(self, identifier="img-text"):
        super().__init__(identifier, "cpu")

    def get_embd_dim(self) -> int:
        return 2

    def get_img_embds(self, images):
        raise NotImplementedError

    def get_text_embds(self, text):
        raise NotImplementedError


class DummyTargetModel(TargetModel):
    def __init__(self):
        super().__init__("target", "cpu")

    def get_img_processor(self):
        return lambda image: torch.zeros((3, image.height, image.width))

    def get_total_neuron_count(self) -> int:
        return 10

    def _get_activations(self, img_features):
        raise NotImplementedError


class DummyScoreFunction(ScoreFunction):
    def __init__(self, config):
        self.config = config
        self.topk_text = config.evaluation.topk_text

    def get_label(self) -> str:
        return "score-a"

    def compute_text_scores(self, neuron_data):
        return (
            torch.tensor([[0.9, 0.2], [0.8, 0.1]]),
            torch.tensor([[1, 0], [0, 1]]),
        )


class SimpleNeuronData:
    def __init__(self):
        self.neuron_indices = [5, 9]

    def __len__(self):
        return len(self.neuron_indices)


@pytest.fixture
def config(tmp_path):
    return NeuroLensConfig(
        io=IOConfig(root_data_dir_path=tmp_path),
        evaluation=EvaluationConfig(sample_count=3, topk_text=2),
    )


def make_evaluation(config, sim_corr_config=None):
    if sim_corr_config is None:
        sim_corr_config = SimCorrConfig(topk_text=2)

    return SimCorrEvaluation(
        config=config,
        sim_corr_config=sim_corr_config,
        path_configs=PathConfigs(config),
        target_model=DummyTargetModel(),
        img_text_model=DummyImageTextModel(),
        img_dataset=DummyImageDataset(),
        text_dataset=TextDataset("texts", ["alpha", "beta", "gamma"]),
        simulator=DummyImageTextModel("simulator"),
        device="cpu",
    )


def patch_precomputed_inputs(monkeypatch, sim_matrix, activations, captured=None):
    def fake_load_similarity_matrix(**kwargs):
        if captured is not None:
            captured["similarity_kwargs"] = kwargs
        return sim_matrix

    def fake_load_activations(**kwargs):
        if captured is not None:
            captured["activation_kwargs"] = kwargs
        return activations

    monkeypatch.setattr(
        "neurolens.evaluation.sim_corr.sim_corr_evaluation.load_similarity_matrix",
        fake_load_similarity_matrix,
    )
    monkeypatch.setattr(
        "neurolens.evaluation.sim_corr.sim_corr_evaluation.load_activations",
        fake_load_activations,
    )


def test_get_score_function_save_filepath_includes_sim_corr_parameters(config):
    evaluation = make_evaluation(config)
    score_function = DummyScoreFunction(config)

    assert evaluation.get_score_function_save_filepath(score_function) == (
        config.io.root_data_dir_path / "data_results/img-text/target/images-texts/"
        "sim_corr-simulator=simulator-sample_count=3-weighted=False-topk_text=2/score-a.csv"
    )


def test_evaluate_delegates_computed_text_scores_to_sim_corr_metrics(config):
    evaluation = make_evaluation(config)
    score_function = DummyScoreFunction(config)
    neuron_data = SimpleNeuronData()
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)

    evaluation.evaluate_from_computed_text_scores = capture

    evaluation.evaluate(
        score_function=score_function,
        neuron_data=neuron_data,
    )

    assert captured["score_function"] is score_function
    assert captured["neuron_data"] is neuron_data
    assert torch.allclose(captured["text_scores"], torch.tensor([[0.9, 0.2], [0.8, 0.1]]))
    assert captured["text_indices"].tolist() == [[1, 0], [0, 1]]


def test_evaluate_from_computed_text_scores_writes_unweighted_metrics(config, monkeypatch):
    evaluation = make_evaluation(config, SimCorrConfig(topk_text=2, weighted=False))
    score_function = DummyScoreFunction(config)
    neuron_data = SimpleNeuronData()
    sim_matrix = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 2.0],
        ]
    )
    activations = torch.tensor(
        [
            [1.0, 10.0],
            [3.0, 8.0],
            [5.0, 6.0],
            [7.0, 4.0],
        ]
    )
    captured = {}
    patch_precomputed_inputs(
        monkeypatch,
        sim_matrix=sim_matrix,
        activations=activations,
        captured=captured,
    )

    evaluation.evaluate_from_computed_text_scores(
        score_function=score_function,
        neuron_data=neuron_data,
        text_scores=torch.tensor([[0.1, 0.9], [0.6, 0.4]]),
        text_indices=torch.tensor([[0, 1], [1, 2]]),
    )

    df = pd.read_csv(evaluation.get_score_function_save_filepath(score_function))

    expected_corrs = [
        torch.corrcoef(torch.stack([activations[:, 0], sim_matrix[:, [0, 1]].sum(1)]))[0, 1].item(),
        torch.corrcoef(torch.stack([activations[:, 1], sim_matrix[:, [1, 2]].sum(1)]))[0, 1].item(),
    ]

    assert captured["similarity_kwargs"]["img_indices"] == [0, 1, 2, 3]
    assert captured["activation_kwargs"]["neuron_indices"] == [5, 9]
    assert df[Evaluation.NEURON_INDEX_COLUMN].tolist() == [5, 9]
    assert df[SimCorrEvaluationColumnNames.SIM_CORR_SCORE].tolist() == pytest.approx(expected_corrs)
    assert df[SimCorrEvaluationColumnNames.get_text_column_name(0)].tolist() == [
        "alpha",
        "beta",
    ]
    assert df[SimCorrEvaluationColumnNames.get_text_column_name(1)].tolist() == [
        "beta",
        "gamma",
    ]
    assert df[SimCorrEvaluationColumnNames.get_weight_column_name(0)].tolist() == [
        1.0,
        1.0,
    ]
    assert df[SimCorrEvaluationColumnNames.get_weight_column_name(1)].tolist() == [
        1.0,
        1.0,
    ]


def test_evaluate_from_computed_text_scores_writes_weighted_metrics(config, monkeypatch):
    evaluation = make_evaluation(config, SimCorrConfig(topk_text=2, weighted=True))
    score_function = DummyScoreFunction(config)
    neuron_data = SimpleNeuronData()
    sim_matrix = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 2.0],
        ]
    )
    activations = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]
    )
    text_scores = torch.tensor([[2.0, -1.0], [0.5, 1.5]])
    text_indices = torch.tensor([[0, 1], [1, 2]])
    patch_precomputed_inputs(
        monkeypatch,
        sim_matrix=sim_matrix,
        activations=activations,
    )

    evaluation.evaluate_from_computed_text_scores(
        score_function=score_function,
        neuron_data=neuron_data,
        text_scores=text_scores,
        text_indices=text_indices,
    )

    df = pd.read_csv(evaluation.get_score_function_save_filepath(score_function))

    expected_corrs = []
    for i in range(len(neuron_data)):
        pred_activations = sim_matrix[:, text_indices[i]] @ text_scores[i]
        expected_corrs.append(torch.corrcoef(torch.stack([activations[:, i], pred_activations]))[0, 1].item())

    assert df[SimCorrEvaluationColumnNames.SIM_CORR_SCORE].tolist() == pytest.approx(expected_corrs)
    assert df[SimCorrEvaluationColumnNames.get_weight_column_name(0)].tolist() == [
        2.0,
        0.5,
    ]
    assert df[SimCorrEvaluationColumnNames.get_weight_column_name(1)].tolist() == [
        -1.0,
        1.5,
    ]


def test_evaluate_from_computed_text_scores_uses_test_split_indices(config, monkeypatch):
    evaluation = make_evaluation(config)
    score_function = DummyScoreFunction(config)
    split_path = evaluation.path_configs.data_precomp_dataset_splits_file_path(
        img_dataset=evaluation.img_dataset,
        split="test",
    )
    split_path.parent.mkdir(parents=True)
    torch.save(torch.tensor([1, 3]), split_path)
    captured = {}
    patch_precomputed_inputs(
        monkeypatch,
        sim_matrix=torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
        activations=torch.tensor([[1.0, 4.0], [2.0, 3.0]]),
        captured=captured,
    )

    neuron_data = SimpleNeuronData()
    evaluation.evaluate_from_computed_text_scores(
        score_function=score_function,
        neuron_data=neuron_data,
        text_scores=torch.ones((len(neuron_data), 2)),
        text_indices=torch.tensor([[0, 1], [1, 2]]),
    )

    assert captured["similarity_kwargs"]["img_indices"] == [1, 3]
    assert captured["activation_kwargs"]["sample_indices"] == [1, 3]


def test_evaluate_from_computed_text_scores_rejects_oversized_topk_text(config, monkeypatch):
    evaluation = make_evaluation(config, SimCorrConfig(topk_text=3))
    patch_precomputed_inputs(
        monkeypatch,
        sim_matrix=torch.ones((4, 3)),
        activations=torch.ones((4, 2)),
    )

    with pytest.raises(ValueError, match="topk_text"):
        evaluation.evaluate_from_computed_text_scores(
            score_function=DummyScoreFunction(config),
            neuron_data=SimpleNeuronData(),
            text_scores=torch.ones((2, 2)),
            text_indices=torch.ones((2, 2), dtype=torch.long),
        )


def test_evaluate_from_computed_text_scores_rejects_short_text_indices(config, monkeypatch):
    evaluation = make_evaluation(config, SimCorrConfig(topk_text=2))
    patch_precomputed_inputs(
        monkeypatch,
        sim_matrix=torch.ones((4, 3)),
        activations=torch.ones((4, 2)),
    )

    with pytest.raises(ValueError, match="topk_text"):
        evaluation.evaluate_from_computed_text_scores(
            score_function=DummyScoreFunction(config),
            neuron_data=SimpleNeuronData(),
            text_scores=torch.ones((2, 2)),
            text_indices=torch.ones((2, 1), dtype=torch.long),
        )


def test_evaluate_from_computed_text_scores_uses_text_dataset_getitem(config, monkeypatch):
    evaluation = make_evaluation(config, SimCorrConfig(topk_text=1))
    evaluation.text_dataset.set_template("a photo of {}")
    patch_precomputed_inputs(
        monkeypatch,
        sim_matrix=torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        activations=torch.tensor([[1.0, 4.0], [2.0, 3.0]]),
    )

    evaluation.evaluate_from_computed_text_scores(
        score_function=DummyScoreFunction(config),
        neuron_data=SimpleNeuronData(),
        text_scores=torch.ones((2, 1)),
        text_indices=torch.tensor([[0], [1]]),
    )

    df = pd.read_csv(evaluation.get_score_function_save_filepath(DummyScoreFunction(config)))

    assert df[SimCorrEvaluationColumnNames.get_text_column_name(0)].tolist() == [
        "a photo of alpha",
        "a photo of beta",
    ]


def test_evaluate_from_computed_text_scores_rejects_nan_correlation(config, monkeypatch):
    evaluation = make_evaluation(config)
    patch_precomputed_inputs(
        monkeypatch,
        sim_matrix=torch.ones((4, 3)),
        activations=torch.ones((4, 2)),
    )

    with pytest.raises(ValueError, match="Correlation value"):
        evaluation.evaluate_from_computed_text_scores(
            score_function=DummyScoreFunction(config),
            neuron_data=SimpleNeuronData(),
            text_scores=torch.ones((2, 2)),
            text_indices=torch.tensor([[0, 1], [1, 2]]),
        )
