import pytest
import torch

from neurolens.config import (
    EvaluationConfig,
    IOConfig,
    NeuroLensConfig,
    NeuronDataConfig,
)
from neurolens.target_model.neuron_data import (
    NeuronData,
    NeuronDataSampleType,
    is_neuron_data_precomputed,
    load_batched_neuron_data,
    load_neuron_data_from_csv,
    precompute_neuron_data,
)
from neurolens.target_model.neuron_data.utils import (
    _get_negative_neuron_data,
    _get_topk_positive_neuron_data,
    save_neuron_data_to_csv,
)
from neurolens.utils.path_utils import PathConfigs


class Identified:
    def __init__(self, identifier):
        self.identifier = identifier


class ImageDataset(Identified):
    def __init__(self, identifier="images", length=4):
        super().__init__(identifier)
        self.length = length

    def __len__(self):
        return self.length


class TargetModel(Identified):
    device = torch.device("cpu")

    def __init__(self, identifier, total_neuron_count=3):
        super().__init__(identifier)
        self.total_neuron_count = total_neuron_count

    def get_total_neuron_count(self):
        return self.total_neuron_count


class ImageTextModel(Identified):
    pass


@pytest.fixture
def path_configs(tmp_path):
    return PathConfigs(NeuroLensConfig(io=IOConfig(root_data_dir_path=tmp_path)))


def make_neuron_data(neuron_idx, indices, activations, similarities, mean):
    return NeuronData(
        neuron_idx=neuron_idx,
        indices=torch.tensor(indices, dtype=torch.long),
        activation_values=torch.tensor(activations, dtype=torch.float32),
        similarity_values=torch.tensor(similarities, dtype=torch.float32),
        probe_dataset_overall_mean=mean,
    )


def make_load_activations_mock(activations: torch.Tensor):
    def load_activations_mock(**kwargs):
        rows = kwargs.get("sample_indices")
        columns = kwargs.get("neuron_indices")

        if rows is None:
            row_index = slice(None)
        elif isinstance(rows, int):
            row_index = [rows]
        elif isinstance(rows, tuple):
            row_index = slice(rows[0], rows[1])
        else:
            row_index = rows

        if columns is None:
            column_index = slice(None)
        elif isinstance(columns, int):
            column_index = [columns]
        elif isinstance(columns, tuple):
            column_index = slice(columns[0], columns[1])
        else:
            column_index = columns

        return activations[row_index, column_index]

    return load_activations_mock


def test_get_topk_positive_neuron_data_selects_live_neurons_and_samples(monkeypatch, path_configs):
    activations = torch.tensor(
        [
            [0.0, 4.0, 1.0],
            [2.0, 3.0, 0.0],
            [5.0, 2.0, 2.0],
            [1.0, 1.0, 9.0],
        ]
    )

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    result = _get_topk_positive_neuron_data(
        config=NeuroLensConfig(neuron_data=NeuronDataConfig(topk_neurons=2, required_sample_count=2, mean_top_count=2)),
        path_configs=path_configs,
        target_model=TargetModel("target"),
        img_dataset=ImageDataset(length=4),
        device="cpu",
    )

    assert list(result) == [2, 0]
    assert torch.equal(result[2].indices, torch.tensor([3, 2]))
    assert torch.equal(result[2].activation_values, torch.tensor([9.0, 2.0]))
    assert result[2].probe_dataset_overall_mean == pytest.approx(3.0)
    assert torch.equal(result[0].indices, torch.tensor([2, 1]))
    assert torch.equal(result[0].activation_values, torch.tensor([5.0, 2.0]))


def test_get_topk_positive_neuron_data_respects_ignored_indices(monkeypatch, path_configs):
    activations = torch.tensor(
        [
            [1.0],
            [5.0],
            [3.0],
        ]
    )

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    result = _get_topk_positive_neuron_data(
        config=NeuroLensConfig(
            neuron_data=NeuronDataConfig(
                topk_neurons=1,
                required_sample_count=2,
                ignore_overly_active_neurons=False,
                mean_top_count=2,
            )
        ),
        path_configs=path_configs,
        target_model=TargetModel("target", total_neuron_count=1),
        img_dataset=ImageDataset(length=3),
        device="cpu",
        ignore_input_indices=torch.tensor([1]),
    )

    assert torch.equal(result[0].indices, torch.tensor([2, 0]))
    assert torch.equal(result[0].activation_values, torch.tensor([3.0, 1.0]))


def test_get_topk_positive_neuron_data_respects_input_neurons(monkeypatch, path_configs):
    activations = torch.tensor(
        [
            [10.0, 1.0, 4.0],
            [9.0, 2.0, 3.0],
            [8.0, 3.0, 2.0],
        ]
    )

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    result = _get_topk_positive_neuron_data(
        config=NeuroLensConfig(
            neuron_data=NeuronDataConfig(
                topk_neurons=1,
                required_sample_count=2,
                ignore_overly_active_neurons=False,
                mean_top_count=2,
            )
        ),
        path_configs=path_configs,
        target_model=TargetModel("target"),
        img_dataset=ImageDataset(length=3),
        device="cpu",
        input_neurons=[1, 2],
    )

    assert list(result) == [2]


def test_get_topk_positive_neuron_data_can_exclude_overly_active_neurons(monkeypatch, path_configs):
    activations = torch.tensor(
        [
            [10.0, 1.0, 0.0],
            [9.0, 0.0, 0.0],
            [8.0, 2.0, 0.0],
            [7.0, 0.0, 0.0],
        ]
    )

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    result = _get_topk_positive_neuron_data(
        config=NeuroLensConfig(
            neuron_data=NeuronDataConfig(
                topk_neurons=1,
                required_sample_count=2,
                ignore_overly_active_neurons=True,
                mean_top_count=2,
                overly_active_neuron_rate=0.8,
            )
        ),
        path_configs=path_configs,
        target_model=TargetModel("target"),
        img_dataset=ImageDataset(length=4),
        device="cpu",
    )

    assert list(result) == [1]


def test_get_topk_positive_neuron_data_ranks_by_mean_top_count(monkeypatch, path_configs):
    activations = torch.tensor(
        [
            [10.0, 6.0],
            [0.0, 6.0],
            [0.0, 6.0],
        ]
    )

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    result = _get_topk_positive_neuron_data(
        config=NeuroLensConfig(
            neuron_data=NeuronDataConfig(
                topk_neurons=1,
                required_sample_count=1,
                mean_top_count=2,
                ignore_overly_active_neurons=False,
            )
        ),
        path_configs=path_configs,
        target_model=TargetModel("target", total_neuron_count=2),
        img_dataset=ImageDataset(length=3),
        device="cpu",
    )

    assert list(result) == [1]


def test_get_topk_positive_neuron_data_rejects_oversized_mean_top_count(
    path_configs,
):
    with pytest.raises(ValueError, match="mean_top_count"):
        _get_topk_positive_neuron_data(
            config=NeuroLensConfig(
                neuron_data=NeuronDataConfig(
                    topk_neurons=1,
                    required_sample_count=1,
                    mean_top_count=4,
                )
            ),
            path_configs=path_configs,
            target_model=TargetModel("target", total_neuron_count=1),
            img_dataset=ImageDataset(length=3),
            device="cpu",
        )


def test_get_topk_positive_neuron_data_rejects_impossible_sample_count(monkeypatch, path_configs):
    activations = torch.ones((3, 1))

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    with pytest.raises(ValueError, match="required_sample_count"):
        _get_topk_positive_neuron_data(
            config=NeuroLensConfig(
                neuron_data=NeuronDataConfig(
                    topk_neurons=1,
                    required_sample_count=3,
                    ignore_overly_active_neurons=False,
                    mean_top_count=3,
                )
            ),
            path_configs=path_configs,
            target_model=TargetModel("target", total_neuron_count=1),
            img_dataset=ImageDataset(length=3),
            device="cpu",
            ignore_input_indices=torch.tensor([1]),
        )


def test_get_topk_positive_neuron_data_rejects_too_many_requested_neurons(monkeypatch, path_configs):
    activations = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    with pytest.raises(ValueError, match="topk_neurons"):
        _get_topk_positive_neuron_data(
            config=NeuroLensConfig(
                neuron_data=NeuronDataConfig(topk_neurons=2, required_sample_count=2, mean_top_count=2)
            ),
            path_configs=path_configs,
            target_model=TargetModel("target"),
            img_dataset=ImageDataset(length=3),
            device="cpu",
        )


def test_get_negative_neuron_data_selects_most_similar_available_negatives(monkeypatch, path_configs):
    activations = torch.tensor([[10.0], [9.0], [1.0], [2.0], [3.0]])

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    result = _get_negative_neuron_data(
        config=NeuroLensConfig(neuron_data=NeuronDataConfig(neg_fetch_batch_size=2, allow_neg_reuse=False)),
        path_configs=path_configs,
        img_dataset=ImageDataset(length=5),
        target_model=TargetModel("target"),
        pos_neuron_data=make_neuron_data(
            neuron_idx=0,
            indices=[0, 1],
            activations=[10.0, 9.0],
            similarities=[1.0, 1.0],
            mean=5.0,
        ),
        device="cpu",
        all_img_embds=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.0],
                [0.0, 0.8],
                [0.5, 0.5],
            ]
        ),
    )

    assert result.neuron_idx == 0
    assert torch.equal(result.indices, torch.tensor([2, 3]))
    assert torch.equal(result.activation_values, torch.tensor([1.0, 2.0]))
    assert torch.equal(result.similarity_values, torch.tensor([0.9, 0.8]))


def test_get_negative_neuron_data_rejects_reuse_when_too_few_negatives(monkeypatch, path_configs):
    activations = torch.tensor([[10.0], [9.0], [1.0]])

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    with pytest.raises(ValueError, match="allow_reuse == False"):
        _get_negative_neuron_data(
            config=NeuroLensConfig(
                neuron_data=NeuronDataConfig(
                    neg_fetch_batch_size=2,
                    allow_neg_reuse=False,
                )
            ),
            path_configs=path_configs,
            img_dataset=ImageDataset(length=3),
            target_model=TargetModel("target"),
            pos_neuron_data=make_neuron_data(
                neuron_idx=0,
                indices=[0, 1],
                activations=[10.0, 9.0],
                similarities=[1.0, 1.0],
                mean=5.0,
            ),
            device="cpu",
            all_img_embds=torch.eye(3),
        )


def test_get_negative_neuron_data_rejects_when_no_negatives(monkeypatch, path_configs):
    activations = torch.tensor([[10.0], [9.0], [8.0]])

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    with pytest.raises(ValueError, match="No negative samples"):
        _get_negative_neuron_data(
            config=NeuroLensConfig(),
            path_configs=path_configs,
            img_dataset=ImageDataset(length=3),
            target_model=TargetModel("target"),
            pos_neuron_data=make_neuron_data(
                neuron_idx=0,
                indices=[0],
                activations=[10.0],
                similarities=[1.0],
                mean=5.0,
            ),
            device="cpu",
            all_img_embds=torch.eye(3),
        )


def test_get_negative_neuron_data_allows_reuse_when_configured(monkeypatch, path_configs):
    activations = torch.tensor([[10.0], [9.0], [1.0]])

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    result = _get_negative_neuron_data(
        config=NeuroLensConfig(neuron_data=NeuronDataConfig(neg_fetch_batch_size=2, allow_neg_reuse=True)),
        path_configs=path_configs,
        img_dataset=ImageDataset(length=3),
        target_model=TargetModel("target"),
        pos_neuron_data=make_neuron_data(
            neuron_idx=0,
            indices=[0, 1],
            activations=[10.0, 9.0],
            similarities=[1.0, 1.0],
            mean=5.0,
        ),
        device="cpu",
        all_img_embds=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
    )

    assert torch.equal(result.indices, torch.tensor([2, 2]))


def test_get_negative_neuron_data_respects_ignored_indices(monkeypatch, path_configs):
    activations = torch.tensor([[10.0], [9.0], [1.0], [2.0]])

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_activations",
        make_load_activations_mock(activations),
    )

    result = _get_negative_neuron_data(
        config=NeuroLensConfig(neuron_data=NeuronDataConfig(neg_fetch_batch_size=2, allow_neg_reuse=False)),
        path_configs=path_configs,
        img_dataset=ImageDataset(length=4),
        target_model=TargetModel("target"),
        pos_neuron_data=make_neuron_data(
            neuron_idx=0,
            indices=[0],
            activations=[10.0],
            similarities=[1.0],
            mean=5.0,
        ),
        device="cpu",
        all_img_embds=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.0], [1.0, 0.0]]),
        ignore_input_indices=torch.tensor([3]),
    )

    assert torch.equal(result.indices, torch.tensor([2]))


def test_save_and_load_neuron_data_round_trips_csv(path_configs):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco", length=4)

    save_neuron_data_to_csv(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        neuron_data={
            3: make_neuron_data(3, [2, 0], [0.7, 0.4], [0.9, 0.6], 0.25),
            5: make_neuron_data(5, [1], [0.8], [0.5], 0.5),
        },
        sample_type=NeuronDataSampleType.POSITIVE,
    )

    result, neuron_indices = load_neuron_data_from_csv(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        sample_type=NeuronDataSampleType.POSITIVE,
        neuron_indices=[3, 5],
        device="cpu",
    )

    assert neuron_indices == [3, 5]
    assert set(result) == {3, 5}
    assert torch.equal(result[3].indices, torch.tensor([2, 0]))
    assert torch.allclose(result[3].activation_values, torch.tensor([0.7, 0.4]))
    assert torch.allclose(result[3].similarity_values, torch.tensor([0.9, 0.6]))
    assert result[3].probe_dataset_overall_mean == pytest.approx(0.25)
    assert torch.equal(result[5].indices, torch.tensor([1]))


def test_load_neuron_data_from_csv_infers_neuron_indices(path_configs):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco", length=4)

    save_neuron_data_to_csv(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        neuron_data={
            3: make_neuron_data(3, [2], [0.7], [0.9], 0.25),
            5: make_neuron_data(5, [1], [0.8], [0.5], 0.5),
        },
        sample_type=NeuronDataSampleType.POSITIVE,
    )

    result, neuron_indices = load_neuron_data_from_csv(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        sample_type=NeuronDataSampleType.POSITIVE,
        device="cpu",
    )

    assert neuron_indices == [3, 5]
    assert set(result) == {3, 5}


def test_load_neuron_data_from_csv_raises_for_missing_neuron(path_configs):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco", length=4)

    save_neuron_data_to_csv(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        neuron_data={3: make_neuron_data(3, [2], [0.7], [0.9], 0.25)},
        sample_type=NeuronDataSampleType.POSITIVE,
    )

    with pytest.raises(ValueError, match="No .* neuron data found"):
        load_neuron_data_from_csv(
            path_configs=path_configs,
            target_model=target_model,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            sample_type=NeuronDataSampleType.POSITIVE,
            neuron_indices=9,
        )


def test_load_batched_neuron_data_precomputes_missing_cache(monkeypatch, path_configs):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco", length=4)
    config = NeuroLensConfig(evaluation=EvaluationConfig(sample_count=1))
    pos_neuron_data = {3: make_neuron_data(3, [0, 1], [0.9, 0.8], [1.0, 1.0], 0.5)}
    neg_neuron_data = make_neuron_data(3, [2, 3], [0.2, 0.1], [0.7, 0.6], 0.5)

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils._get_topk_positive_neuron_data",
        lambda **kwargs: pos_neuron_data,
    )
    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_img_embds",
        lambda **kwargs: torch.eye(4),
    )
    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils._get_negative_neuron_data",
        lambda **kwargs: neg_neuron_data,
    )
    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.neuron_data.load_img_embds",
        lambda **kwargs: torch.eye(4),
    )

    result = load_batched_neuron_data(
        config=config,
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        neuron_indices=[3],
        device="cpu",
    )

    assert is_neuron_data_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
    assert result.neuron_indices == [3]
    assert torch.equal(result.pos_sample_indices, torch.tensor([[0]]))
    assert torch.equal(result.neg_sample_indices, torch.tensor([[2]]))


def test_load_batched_neuron_data_can_disable_missing_cache_precompute(path_configs):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco", length=4)

    with pytest.raises(FileNotFoundError):
        load_batched_neuron_data(
            config=NeuroLensConfig(),
            path_configs=path_configs,
            target_model=target_model,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            neuron_indices=[3],
            device="cpu",
            precompute_if_missing=False,
        )

    assert not is_neuron_data_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )


def test_load_batched_neuron_data_uses_positive_and_negative_csvs(monkeypatch, path_configs):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco", length=5)

    for sample_type, indices, activations, similarities in [
        (NeuronDataSampleType.POSITIVE, [0, 1, 2], [0.9, 0.8, 0.7], [1.0, 1.0, 1.0]),
        (NeuronDataSampleType.NEGATIVE, [3, 4, 2], [0.2, 0.1, 0.0], [0.6, 0.5, 0.4]),
    ]:
        save_neuron_data_to_csv(
            path_configs=path_configs,
            target_model=target_model,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            neuron_data={3: make_neuron_data(3, indices, activations, similarities, 0.25)},
            sample_type=sample_type,
        )

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.neuron_data.load_img_embds",
        lambda **kwargs: torch.eye(5),
    )

    result = load_batched_neuron_data(
        config=NeuroLensConfig(evaluation=EvaluationConfig(sample_count=2)),
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        neuron_indices=[3],
        device="cpu",
    )

    assert result.neuron_indices == [3]
    assert torch.equal(result.pos_sample_indices, torch.tensor([[0, 1]]))
    assert torch.equal(result.neg_sample_indices, torch.tensor([[3, 4]]))
    assert torch.equal(
        result.pos_embds,
        torch.eye(5)[torch.tensor([[0, 1]])],
    )
    assert torch.equal(
        result.neg_embds,
        torch.eye(5)[torch.tensor([[3, 4]])],
    )


def test_load_batched_neuron_data_infers_neuron_indices(monkeypatch, path_configs):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco", length=6)

    for sample_type, neuron_data in [
        (
            NeuronDataSampleType.POSITIVE,
            {
                3: make_neuron_data(3, [0, 1], [0.9, 0.8], [1.0, 1.0], 0.25),
                5: make_neuron_data(5, [2, 3], [0.7, 0.6], [1.0, 1.0], 0.5),
            },
        ),
        (
            NeuronDataSampleType.NEGATIVE,
            {
                3: make_neuron_data(3, [4, 5], [0.2, 0.1], [0.6, 0.5], 0.25),
                5: make_neuron_data(5, [0, 1], [0.3, 0.2], [0.4, 0.3], 0.5),
            },
        ),
    ]:
        save_neuron_data_to_csv(
            path_configs=path_configs,
            target_model=target_model,
            img_text_model=img_text_model,
            img_dataset=img_dataset,
            neuron_data=neuron_data,
            sample_type=sample_type,
        )

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.neuron_data.load_img_embds",
        lambda **kwargs: torch.eye(6),
    )

    result = load_batched_neuron_data(
        config=NeuroLensConfig(evaluation=EvaluationConfig(sample_count=1)),
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        device="cpu",
    )

    assert result.neuron_indices == [3, 5]
    assert torch.equal(result.pos_sample_indices, torch.tensor([[0], [2]]))
    assert torch.equal(result.neg_sample_indices, torch.tensor([[4], [0]]))


def test_precompute_neuron_data_writes_positive_and_negative_csvs(monkeypatch, path_configs):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco", length=4)
    pos_neuron_data = {3: make_neuron_data(3, [0, 1], [0.9, 0.8], [1.0, 1.0], 0.5)}
    neg_neuron_data = make_neuron_data(3, [2, 3], [0.2, 0.1], [0.7, 0.6], 0.5)

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils._get_topk_positive_neuron_data",
        lambda **kwargs: pos_neuron_data,
    )
    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils.load_img_embds",
        lambda **kwargs: torch.eye(4),
    )
    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.utils._get_negative_neuron_data",
        lambda **kwargs: neg_neuron_data,
    )

    precompute_neuron_data(
        config=NeuroLensConfig(),
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        device="cpu",
    )

    pos_loaded, _ = load_neuron_data_from_csv(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        sample_type=NeuronDataSampleType.POSITIVE,
        neuron_indices=3,
    )
    neg_loaded, _ = load_neuron_data_from_csv(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        sample_type=NeuronDataSampleType.NEGATIVE,
        neuron_indices=3,
    )

    assert torch.equal(pos_loaded[3].indices, torch.tensor([0, 1]))
    assert torch.equal(neg_loaded[3].indices, torch.tensor([2, 3]))


def test_is_neuron_data_precomputed_returns_true_when_all_sample_files_exist(
    path_configs,
):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco")

    for sample_type in NeuronDataSampleType:
        csv_path = path_configs.data_results_neuron_data_file_path(
            target_model=target_model,
            img_dataset=img_dataset,
            img_text_model=img_text_model,
            sample_type=sample_type,
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.touch()

    assert is_neuron_data_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )


def test_is_neuron_data_precomputed_returns_false_when_any_sample_file_is_missing(
    path_configs,
):
    target_model = TargetModel("resnet")
    img_text_model = ImageTextModel("clip")
    img_dataset = ImageDataset("coco")
    csv_path = path_configs.data_results_neuron_data_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
        img_text_model=img_text_model,
        sample_type=NeuronDataSampleType.POSITIVE,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.touch()

    assert not is_neuron_data_precomputed(
        path_configs=path_configs,
        target_model=target_model,
        img_text_model=img_text_model,
        img_dataset=img_dataset,
    )
