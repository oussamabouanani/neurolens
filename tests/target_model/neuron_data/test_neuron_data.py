import pytest
import torch

from neurolens.config import EvaluationConfig, NeuroLensConfig
from neurolens.target_model.neuron_data import BatchedNeuronData, NeuronData


class Identified:
    identifier = "identified"


def make_neuron_data(neuron_idx, indices, activations, similarities, mean):
    return NeuronData(
        neuron_idx=neuron_idx,
        indices=torch.tensor(indices, dtype=torch.long),
        activation_values=torch.tensor(activations, dtype=torch.float32),
        similarity_values=torch.tensor(similarities, dtype=torch.float32),
        probe_dataset_overall_mean=mean,
    )


def test_neuron_data_to_returns_moved_copy():
    neuron_data = make_neuron_data(4, [0], [0.9], [1.0], 0.25)

    moved = neuron_data.to(torch.device("meta"))

    assert moved is not neuron_data
    assert neuron_data.indices.device.type == "cpu"
    assert moved.indices.device.type == "meta"
    assert moved.activation_values.device.type == "meta"
    assert moved.similarity_values.device.type == "meta"
    assert moved.neuron_idx == neuron_data.neuron_idx
    assert moved.probe_dataset_overall_mean == neuron_data.probe_dataset_overall_mean


def test_batched_neuron_data_stacks_values_and_fetches_embeddings(monkeypatch):
    all_embds = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 10.0, 0.0],
                [2.0, 11.0, 0.0],
                [3.0, 12.0, 0.0],
                [4.0, 13.0, 0.0],
                [5.0, 14.0, 0.0],
                [6.0, 15.0, 0.0],
            ]
        ),
        dim=-1,
    )

    def fake_load_img_embds(**kwargs):
        return all_embds

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.neuron_data.load_img_embds",
        fake_load_img_embds,
    )

    batched = BatchedNeuronData(
        config=NeuroLensConfig(evaluation=EvaluationConfig(sample_count=2)),
        pos_neuron_data_list=[
            make_neuron_data(4, [2, 0, 5], [0.9, 0.8, 0.7], [1.0, 1.0, 1.0], 0.25),
            make_neuron_data(7, [3, 1, 4], [0.6, 0.5, 0.4], [1.0, 1.0, 1.0], 0.5),
        ],
        neg_neuron_data_list=[
            make_neuron_data(4, [1, 3, 4], [0.2, 0.1, 0.0], [0.95, 0.85, 0.75], 0.25),
            make_neuron_data(7, [0, 2, 5], [0.3, 0.2, 0.1], [0.65, 0.55, 0.45], 0.5),
        ],
        path_configs=Identified(),
        img_text_model=Identified(),
        img_dataset=Identified(),
    )

    assert len(batched) == 2
    assert batched.neuron_indices == [4, 7]
    assert torch.equal(batched.pos_sample_indices, torch.tensor([[2, 0], [3, 1]]))
    assert torch.equal(batched.neg_sample_indices, torch.tensor([[1, 3], [0, 2]]))
    assert torch.equal(
        batched.activation_values,
        torch.tensor([[0.9, 0.8], [0.6, 0.5]]),
    )
    assert torch.equal(
        batched.neg_activation_values,
        torch.tensor([[0.2, 0.1], [0.3, 0.2]]),
    )
    assert torch.equal(
        batched.similarity_values,
        torch.tensor([[0.95, 0.85], [0.65, 0.55]]),
    )
    assert batched.probe_dataset_overall_mean == (0.25, 0.5)
    assert torch.equal(
        batched.pos_embds,
        torch.stack(
            [
                all_embds[torch.tensor([2, 0])],
                all_embds[torch.tensor([3, 1])],
            ]
        ),
    )
    assert torch.equal(
        batched.neg_embds,
        torch.stack(
            [
                all_embds[torch.tensor([1, 3])],
                all_embds[torch.tensor([0, 2])],
            ]
        ),
    )


def test_batched_neuron_data_to_returns_moved_copy(monkeypatch):
    all_embds = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        dim=-1,
    )

    def fake_load_img_embds(**kwargs):
        return all_embds

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.neuron_data.load_img_embds",
        fake_load_img_embds,
    )

    batched = BatchedNeuronData(
        config=NeuroLensConfig(evaluation=EvaluationConfig(sample_count=1)),
        pos_neuron_data_list=[
            make_neuron_data(4, [0], [0.9], [1.0], 0.25),
        ],
        neg_neuron_data_list=[
            make_neuron_data(4, [1], [0.2], [0.95], 0.25),
        ],
        path_configs=Identified(),
        img_text_model=Identified(),
        img_dataset=Identified(),
    )

    moved = batched.to(torch.device("meta"))

    assert moved is not batched
    assert batched.pos_sample_indices.device.type == "cpu"
    assert moved.pos_sample_indices.device.type == "meta"
    assert moved.neg_sample_indices.device.type == "meta"
    assert moved.activation_values.device.type == "meta"
    assert moved.neg_activation_values.device.type == "meta"
    assert moved.similarity_values.device.type == "meta"
    assert moved.pos_embds.device.type == "meta"
    assert moved.neg_embds.device.type == "meta"


def test_batched_neuron_data_rejects_non_normalized_embeddings(monkeypatch):
    all_embds = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 11.0],
        ]
    )

    def fake_load_img_embds(**kwargs):
        return all_embds

    monkeypatch.setattr(
        "neurolens.target_model.neuron_data.neuron_data.load_img_embds",
        fake_load_img_embds,
    )

    with pytest.raises(ValueError, match="pos_embds in batch are not l2 normalized"):
        BatchedNeuronData(
            config=NeuroLensConfig(evaluation=EvaluationConfig(sample_count=1)),
            pos_neuron_data_list=[
                make_neuron_data(4, [0], [0.9], [1.0], 0.25),
            ],
            neg_neuron_data_list=[
                make_neuron_data(4, [1], [0.2], [0.95], 0.25),
            ],
            path_configs=Identified(),
            img_text_model=Identified(),
            img_dataset=Identified(),
        )


def test_batched_neuron_data_requires_matching_positive_and_negative_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        BatchedNeuronData(
            config=NeuroLensConfig(evaluation=EvaluationConfig(sample_count=1)),
            pos_neuron_data_list=[
                make_neuron_data(1, [0], [1.0], [1.0], 0.5),
            ],
            neg_neuron_data_list=[],
            path_configs=Identified(),
            img_text_model=Identified(),
            img_dataset=Identified(),
        )
