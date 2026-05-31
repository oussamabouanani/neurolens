from pathlib import Path

import pytest
from pydantic import ValidationError

from neurolens import DEFAULT_CONFIG, NeuroLensConfig, load_config
from neurolens.config import DatasetConfig, EvaluationConfig, IOConfig, NeuronDataConfig


def test_default_config_values():
    config = NeuroLensConfig()

    assert config.io.root_data_dir_path == Path(".")
    assert config.io.raw_data_dir_name == "data_raw"
    assert config.io.precomputed_data_dir_name == "data_precomp"
    assert config.io.results_data_dir_name == "data_results"
    assert config.io.target_model_parent_dir_name == "target_models"
    assert config.io.img_text_model_parent_dir_name == "img_text_models"
    assert config.io.neuron_data_filename == "{img_text_model}_{sample_type}_neuron_data.csv"
    assert config.io.activations_dir_name == "activations"
    assert config.io.activations_filename == "{img_dataset}_activations.zarr"
    assert config.io.img_embds_dir_name == "img_embds"
    assert config.io.img_embds_filename == "{img_dataset}_img_embds.zarr"
    assert config.io.text_embds_dir_name == "text_embds"
    assert config.io.text_embds_filename == "{text_dataset}_{template}_text_embds.zarr"
    assert config.io.sim_mat_dir_name == "sim_mat"
    assert config.io.sim_mat_filename == "{img_dataset}_{text_dataset}_sim_mat.zarr"
    assert config.io.dataset_split_dir_name == "dataset_splits-{img_dataset}"
    assert config.io.zarr_batch_save_count == 128
    assert config.dataset.vanilla_template == "{}"
    assert config.dataset.templates == [
        "{}",
        "{}-like",
        "a {}",
        "an image of a {}",
        "a closeup of a {}",
        "an image of a closeup of {}",
    ]
    assert config.dataset.similarity_matrix_use_vanilla_template is True
    assert config.dataset.precomp_batch_size == 128
    assert config.neuron_data.topk_neurons == 200
    assert config.neuron_data.required_sample_count == 100
    assert config.neuron_data.mean_top_count == 100
    assert config.neuron_data.ignore_overly_active_neurons is True
    assert config.neuron_data.overly_active_neuron_rate == 0.8
    assert config.neuron_data.neg_fetch_batch_size == 4096
    assert config.neuron_data.allow_neg_reuse is True
    assert config.evaluation.topk_text == 5
    assert config.evaluation.sample_count == 30


def test_default_config_export_matches_model_defaults():
    assert DEFAULT_CONFIG == NeuroLensConfig()


def test_config_accepts_nested_config_instances():
    config = NeuroLensConfig(
        io=IOConfig(root_data_dir_path="/tmp/neurolens-data"),
        dataset=DatasetConfig(
            vanilla_template="a {} template",
            templates=["{}", "a photo of {}"],
            similarity_matrix_use_vanilla_template=False,
            precomp_batch_size=16,
        ),
        neuron_data=NeuronDataConfig(topk_neurons=50, required_sample_count=25, mean_top_count=10),
        evaluation=EvaluationConfig(topk_text=10, sample_count=12),
    )

    assert config.io.root_data_dir_path == Path("/tmp/neurolens-data")
    assert config.dataset.vanilla_template == "a {} template"
    assert config.dataset.templates == ["{}", "a photo of {}"]
    assert config.dataset.similarity_matrix_use_vanilla_template is False
    assert config.dataset.precomp_batch_size == 16
    assert config.neuron_data.topk_neurons == 50
    assert config.neuron_data.required_sample_count == 25
    assert config.neuron_data.mean_top_count == 10
    assert config.evaluation.topk_text == 10
    assert config.evaluation.sample_count == 12


def test_config_rejects_unknown_top_level_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NeuroLensConfig(unknown_field=True)


def test_config_rejects_unknown_nested_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NeuroLensConfig(io={"root_data_dir_path": "/data", "unknown_field": True})


def test_config_is_immutable():
    config = NeuroLensConfig()

    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.io = IOConfig(root_data_dir_path="/data")

    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.io.root_data_dir_path = Path("/data")


def test_config_requires_positive_counts():
    with pytest.raises(ValidationError, match="greater than 0"):
        IOConfig(zarr_batch_save_count=0)

    with pytest.raises(ValidationError, match="greater than 0"):
        DatasetConfig(precomp_batch_size=0)

    with pytest.raises(ValidationError, match="greater than 0"):
        NeuronDataConfig(topk_neurons=0)

    with pytest.raises(ValidationError, match="greater than 0"):
        NeuronDataConfig(required_sample_count=0)

    with pytest.raises(ValidationError, match="greater than 0"):
        NeuronDataConfig(mean_top_count=0)

    with pytest.raises(ValidationError, match="greater than 0"):
        EvaluationConfig(topk_text=0)

    with pytest.raises(ValidationError, match="greater than 0"):
        EvaluationConfig(sample_count=0)


def test_dataset_config_validates_vanilla_template():
    with pytest.raises(ValidationError, match="exactly the following fields once"):
        DatasetConfig(vanilla_template="no placeholder")


def test_dataset_config_validates_templates():
    assert DatasetConfig(templates=["{}", "a photo of {}"]).templates == [
        "{}",
        "a photo of {}",
    ]

    with pytest.raises(ValidationError, match="at least one template is required"):
        DatasetConfig(templates=[])

    with pytest.raises(ValidationError, match="exactly the following fields once"):
        DatasetConfig(templates=["no placeholder"])


def test_io_config_validates_neuron_data_filename_template():
    assert (
        IOConfig(neuron_data_filename="{img_text_model}_{sample_type}.csv").neuron_data_filename
        == "{img_text_model}_{sample_type}.csv"
    )

    with pytest.raises(ValidationError, match="exactly the following fields once in this order"):
        IOConfig(neuron_data_filename="{sample_type}_{img_text_model}.csv")

    with pytest.raises(ValidationError, match="exactly the following fields once"):
        IOConfig(neuron_data_filename="{img_text_model}.csv")

    with pytest.raises(ValidationError, match="exactly the following fields once"):
        IOConfig(neuron_data_filename="{img_text_model}_{sample_type}_{sample_type}.csv")

    with pytest.raises(ValidationError, match="exactly the following fields once"):
        IOConfig(neuron_data_filename="{img_text_model!r}_{sample_type}.csv")


@pytest.mark.parametrize(
    "field_name, valid_filename",
    [
        ("activations_filename", "{img_dataset}_acts.zarr"),
        ("img_embds_filename", "{img_dataset}_embds.zarr"),
        ("text_embds_filename", "{text_dataset}_{template}_embds.zarr"),
        ("sim_mat_filename", "{img_dataset}_{text_dataset}_sims.zarr"),
        ("dataset_split_dir_name", "splits-{img_dataset}"),
    ],
)
def test_io_config_accepts_valid_filename_templates(field_name, valid_filename):
    assert getattr(IOConfig(**{field_name: valid_filename}), field_name) == valid_filename


@pytest.mark.parametrize(
    "field_name, invalid_filename",
    [
        ("activations_filename", "activations.zarr"),
        ("img_embds_filename", "img_embds.zarr"),
        ("text_embds_filename", "{template}_{text_dataset}_embds.zarr"),
        ("sim_mat_filename", "{text_dataset}_{img_dataset}_sims.zarr"),
        ("dataset_split_dir_name", "splits-{text_dataset}"),
    ],
)
def test_io_config_rejects_invalid_filename_templates(field_name, invalid_filename):
    with pytest.raises(ValidationError, match="exactly the following fields once"):
        IOConfig(**{field_name: invalid_filename})


@pytest.mark.parametrize(
    "field_name",
    [
        "raw_data_dir_name",
        "precomputed_data_dir_name",
        "results_data_dir_name",
        "target_model_parent_dir_name",
        "img_text_model_parent_dir_name",
        "activations_dir_name",
        "img_embds_dir_name",
        "text_embds_dir_name",
        "sim_mat_dir_name",
    ],
)
def test_io_config_rejects_path_values_for_directory_names(field_name):
    with pytest.raises(ValidationError, match="plain name"):
        IOConfig(**{field_name: "nested/path"})


@pytest.mark.parametrize(
    "field_name, invalid_filename",
    [
        ("neuron_data_filename", "nested/{img_text_model}_{sample_type}.csv"),
        ("activations_filename", "../{img_dataset}_activations.zarr"),
        ("img_embds_filename", "{img_dataset}/img_embds.zarr"),
        ("text_embds_filename", "{text_dataset}_{template}/text_embds.zarr"),
        ("sim_mat_filename", "{img_dataset}_{text_dataset}/sim_mat.zarr"),
        ("dataset_split_dir_name", "splits/{img_dataset}"),
    ],
)
def test_io_config_rejects_path_values_for_filename_templates(field_name: str, invalid_filename: str):
    with pytest.raises(ValidationError, match="plain name"):
        IOConfig(**{field_name: invalid_filename})


@pytest.mark.parametrize(
    "field_name, invalid_filename",
    [
        ("neuron_data_filename", "{img_text_model}_{sample_type}_neuron_data.zarr"),
        ("neuron_data_filename", "{img_text_model}_{sample_type}_neuron_data"),
        ("activations_filename", "{img_dataset}_activations.csv"),
        ("activations_filename", "{img_dataset}_activations"),
        ("img_embds_filename", "{img_dataset}_img_embds.csv"),
        ("img_embds_filename", "{img_dataset}_img_embds"),
        ("text_embds_filename", "{text_dataset}_{template}_text_embds.csv"),
        ("text_embds_filename", "{text_dataset}_{template}_text_embds"),
        ("sim_mat_filename", "{img_dataset}_{text_dataset}_sim_mat.csv"),
        ("sim_mat_filename", "{img_dataset}_{text_dataset}_sim_mat"),
    ],
)
def test_io_config_rejects_invalid_filename_extensions(field_name: str, invalid_filename: str):
    with pytest.raises(ValidationError, match="filename must end with"):
        IOConfig(**{field_name: invalid_filename})


def test_load_config_returns_defaults_when_no_input_is_provided():
    assert load_config() == DEFAULT_CONFIG


def test_load_config_accepts_config_instance():
    base_config = NeuroLensConfig(
        io=IOConfig(
            root_data_dir_path=Path("/data"),
            precomputed_data_dir_name="precomp",
        )
    )

    config = load_config(base_config)

    assert config.io.root_data_dir_path == Path("/data")
    assert config.io.precomputed_data_dir_name == "precomp"
    assert config.io.raw_data_dir_name == DEFAULT_CONFIG.io.raw_data_dir_name


def test_load_config_accepts_nested_config_dict():
    config = load_config(
        {
            "io": {
                "root_data_dir_path": "/data",
                "activations_filename": "{img_dataset}_target_acts.zarr",
                "zarr_batch_save_count": 8,
            },
            "dataset": {
                "templates": ["{}", "a photo of {}"],
                "similarity_matrix_use_vanilla_template": False,
                "precomp_batch_size": 32,
            },
            "neuron_data": {
                "topk_neurons": 50,
            },
            "evaluation": {
                "topk_text": 7,
                "sample_count": 11,
            },
        }
    )

    assert config.io.root_data_dir_path == Path("/data")
    assert config.io.activations_filename == "{img_dataset}_target_acts.zarr"
    assert config.io.zarr_batch_save_count == 8
    assert config.dataset.templates == ["{}", "a photo of {}"]
    assert config.dataset.similarity_matrix_use_vanilla_template is False
    assert config.dataset.precomp_batch_size == 32
    assert config.neuron_data.topk_neurons == 50
    assert config.neuron_data.required_sample_count == 100
    assert config.evaluation.topk_text == 7
    assert config.evaluation.sample_count == 11


def test_load_config_validates_data():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config({"io": {"raw_data_dir_nam": "typo"}})
