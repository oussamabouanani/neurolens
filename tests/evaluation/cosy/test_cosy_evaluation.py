import pandas as pd
import pytest
import torch
from PIL import Image

from neurolens.config import EvaluationConfig, IOConfig, NeuroLensConfig
from neurolens.dataset.image import ImageDataset
from neurolens.dataset.text import TextDataset
from neurolens.evaluation import Evaluation
from neurolens.evaluation.cosy import (
    CoSyConfig,
    CoSyEvaluation,
    CoSyEvaluationColumnNames,
)
from neurolens.img_text_model import ImageTextModel
from neurolens.score_function import ScoreFunction
from neurolens.target_model import TargetModel
from neurolens.utils.path_utils import PathConfigs


class DummyImageDataset(ImageDataset):
    def __init__(self, identifier="images"):
        super().__init__(identifier)

    def __len__(self):
        return 1

    def get_mean(self):
        return (0.0, 0.0, 0.0)

    def get_std(self):
        return (1.0, 1.0, 1.0)

    def __getitem__(self, idx):
        return Image.new("RGB", (2, 2))


class DummyImageTextModel(ImageTextModel):
    def __init__(self):
        super().__init__("img-text", "cpu")

    def get_embd_dim(self) -> int:
        return 2

    def get_img_embds(self, images):
        raise NotImplementedError

    def get_text_embds(self, text):
        raise NotImplementedError


class DummyTargetModel(TargetModel):
    def __init__(self):
        super().__init__("target", "cpu")
        self.activation_batches = []

    def get_img_processor(self):
        return lambda image: torch.zeros((3, image.height, image.width))

    def get_total_neuron_count(self) -> int:
        return 2

    def _get_activations(self, img_features):
        batch = self.activation_batches.pop(0)
        return batch.to(self.device)


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


@pytest.fixture
def config(tmp_path):
    return NeuroLensConfig(
        io=IOConfig(root_data_dir_path=tmp_path),
        evaluation=EvaluationConfig(sample_count=3, topk_text=2),
    )


@pytest.fixture
def cosy_config():
    return CoSyConfig(
        stable_diffusion_model_identifier="sd-test",
        generated_img_count=2,
    )


def make_evaluation(config, cosy_config):
    return CoSyEvaluation(
        config=config,
        cosy_config=cosy_config,
        path_configs=PathConfigs(config),
        target_model=DummyTargetModel(),
        img_text_model=DummyImageTextModel(),
        img_dataset=DummyImageDataset(),
        text_dataset=TextDataset("texts", ["alpha", "beta"]),
        control_img_dataset=DummyImageDataset("control"),
        device="cpu",
    )


class SimpleNeuronData:
    def __init__(self):
        self.neuron_indices = [5, 9]
        self.activation_values = torch.tensor(
            [
                [2.0, 4.0, 1.0],
                [10.0, 5.0, 1.0],
            ]
        )

    def __len__(self):
        return len(self.neuron_indices)

    def to(self, device):
        self.activation_values = self.activation_values.to(device)
        return self


def test_get_score_function_save_filepath_includes_cosy_parameters(config, cosy_config):
    evaluation = make_evaluation(config, cosy_config)
    score_function = DummyScoreFunction(config)

    assert evaluation.get_score_function_save_filepath(score_function) == (
        config.io.root_data_dir_path / "data_results/img-text/target/images-texts/"
        "cosy_evals-img_gen=sd-test-sample_count=3-normalize=True/score-a.csv"
    )


def test_evaluate_delegates_computed_text_scores_to_cosy_metrics(config, cosy_config):
    evaluation = make_evaluation(config, cosy_config)
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


def test_evaluate_from_computed_text_scores_writes_cosy_metrics(config, cosy_config):
    evaluation = make_evaluation(config, cosy_config)
    score_function = DummyScoreFunction(config)
    neuron_data = SimpleNeuronData()

    evaluation.get_control_activations = lambda neuron_indices: torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 4.0],
            [5.0, 8.0],
        ]
    )
    evaluation.get_prompt_activations = lambda prompt, neuron_idx: {
        ("beta", 5): torch.tensor([4.0, 6.0]),
        ("alpha", 9): torch.tensor([2.0, 10.0]),
    }[(prompt, neuron_idx)]

    evaluation.evaluate_from_computed_text_scores(
        score_function=score_function,
        neuron_data=neuron_data,
        text_scores=torch.zeros((2, 2)),
        text_indices=torch.tensor([[1, 0], [0, 1]]),
    )

    df = pd.read_csv(evaluation.get_score_function_save_filepath(score_function))

    assert df[Evaluation.NEURON_INDEX_COLUMN].tolist() == [5, 9]
    assert df[CoSyEvaluationColumnNames.PROMPT].tolist() == ["beta", "alpha"]
    assert df[CoSyEvaluationColumnNames.DMA_SCORE].tolist() == pytest.approx([1.25, 0.6])
    assert df[CoSyEvaluationColumnNames.MAX_SCORE].tolist() == pytest.approx([1.5, 1.0])
    assert df[CoSyEvaluationColumnNames.AUC_SCORE].tolist() == [
        pytest.approx(5 / 6),
        pytest.approx(4 / 6),
    ]


def test_evaluate_from_computed_text_scores_can_skip_activation_normalization(
    config,
):
    cosy_config = CoSyConfig(normalize_activations=False, generated_img_count=2)
    evaluation = make_evaluation(config, cosy_config)
    score_function = DummyScoreFunction(config)
    neuron_data = SimpleNeuronData()

    evaluation.get_control_activations = lambda neuron_indices: torch.tensor([[0.0, 0.0]])
    evaluation.get_prompt_activations = lambda prompt, neuron_idx: torch.tensor([2.0, 4.0])

    evaluation.evaluate_from_computed_text_scores(
        score_function=score_function,
        neuron_data=neuron_data,
        text_scores=torch.zeros((2, 2)),
        text_indices=torch.tensor([[1, 0], [0, 1]]),
    )

    df = pd.read_csv(evaluation.get_score_function_save_filepath(score_function))

    assert df[CoSyEvaluationColumnNames.DMA_SCORE].tolist() == [3.0, 3.0]
    assert df[CoSyEvaluationColumnNames.MAX_SCORE].tolist() == [4.0, 4.0]


def test_get_prompt_activations_returns_zeros_when_all_images_are_filtered(config, cosy_config):
    evaluation = make_evaluation(config, cosy_config)
    evaluation.img_generator.load_images = lambda prompt: []

    activations = evaluation.get_prompt_activations(prompt="alpha", neuron_idx=5)

    assert torch.equal(activations, torch.zeros(2))


def test_get_prompt_activations_loads_images_through_target_model(config, cosy_config):
    evaluation = make_evaluation(config, cosy_config)
    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]
    evaluation.img_generator.load_images = lambda prompt: images
    evaluation.target_model.activation_batches.append(
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )

    activations = evaluation.get_prompt_activations(prompt="alpha", neuron_idx=1)

    assert torch.equal(activations, torch.tensor([2.0, 5.0]))
