from dataclasses import dataclass
from pathlib import Path

import pytest

from neurolens.config import IOConfig, NeuroLensConfig
from neurolens.utils.path_utils import PathConfigs


@dataclass
class Identified:
    identifier: str


def test_join_uses_configured_root_data_dir():
    path_configs = PathConfigs(NeuroLensConfig(io=IOConfig(root_data_dir_path=Path("/data/root"))))

    assert path_configs.join("a", "b") == Path("/data/root/a/b")


def test_target_activation_file_path_uses_configured_filename():
    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(
                root_data_dir_path=Path("/data/root"),
                activations_filename="{img_dataset}_acts.zarr",
            )
        )
    )
    target_model = Identified("resnet")
    img_dataset = Identified("coco")

    assert path_configs.data_precomp_target_activations_file_path(
        target_model=target_model, img_dataset=img_dataset
    ) == Path("/data/root/data_precomp/target_models/resnet/activations/coco_acts.zarr")


def test_imgtext_embedding_file_path_uses_configured_filename():
    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(
                root_data_dir_path=Path("/data/root"),
                img_embds_filename="{img_dataset}_img_embeddings.zarr",
            )
        )
    )
    img_text_model = Identified("clip")
    img_dataset = Identified("coco")

    assert path_configs.data_precomp_imgtext_img_embds_file_path(
        img_text_model=img_text_model, img_dataset=img_dataset
    ) == Path("/data/root/data_precomp/img_text_models/clip/img_embds/coco_img_embeddings.zarr")


def test_imgtext_text_embedding_file_path_uses_configured_filename():
    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(
                root_data_dir_path=Path("/data/root"),
                text_embds_filename="{text_dataset}_{template}_text_embeddings.zarr",
            )
        )
    )
    img_text_model = Identified("clip")
    text_dataset = Identified("labels")

    assert path_configs.data_precomp_imgtext_text_embds_file_path(
        img_text_model=img_text_model, text_dataset=text_dataset, template="{}"
    ) == Path("/data/root/data_precomp/img_text_models/clip/text_embds/labels_{}_text_embeddings.zarr")


def test_imgtext_similarity_matrix_file_path_uses_joint_dataset_namespace():
    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(
                root_data_dir_path=Path("/data/root"),
                sim_mat_filename="{img_dataset}_{text_dataset}_similarities.zarr",
            )
        )
    )
    img_text_model = Identified("clip")
    img_dataset = Identified("coco")
    text_dataset = Identified("labels")

    assert path_configs.data_precomp_imgtext_sim_matrix_file_path(
        img_text_model=img_text_model,
        img_dataset=img_dataset,
        text_dataset=text_dataset,
    ) == Path("/data/root/data_precomp/img_text_models/clip/sim_mat/coco_labels_similarities.zarr")


def test_dataset_splits_file_path_uses_image_dataset_namespace_and_split_filename():
    path_configs = PathConfigs(NeuroLensConfig(io=IOConfig(root_data_dir_path=Path("/data/root"))))
    img_dataset = Identified("coco")
    assert path_configs.data_precomp_dataset_splits_file_path(
        img_dataset=img_dataset,
        split="train",
    ) == Path("/data/root/data_raw/dataset_splits-coco/train_ids.pt")


def test_dataset_splits_file_path_uses_configured_directory_name():
    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(
                root_data_dir_path=Path("/data/root"),
                dataset_split_dir_name="splits-{img_dataset}",
            )
        )
    )
    img_dataset = Identified("coco")
    assert path_configs.data_precomp_dataset_splits_file_path(
        img_dataset=img_dataset,
        split="val",
    ) == Path("/data/root/data_raw/splits-coco/val_ids.pt")


def test_dataset_splits_file_path_rejects_unknown_split():
    path_configs = PathConfigs(NeuroLensConfig(io=IOConfig(root_data_dir_path=Path("/data/root"))))

    with pytest.raises(ValueError, match="Invalid split"):
        path_configs.data_precomp_dataset_splits_file_path(
            img_dataset=Identified("coco"),
            split="dev",
        )


def test_results_neuron_data_file_path_uses_target_img_dataset_and_imgtext_model():
    path_configs = PathConfigs(NeuroLensConfig(io=IOConfig(root_data_dir_path=Path("/data/root"))))
    target_model = Identified("resnet")
    img_dataset = Identified("coco")
    img_text_model = Identified("clip")

    assert path_configs.data_results_neuron_data_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
        img_text_model=img_text_model,
        sample_type="positive",
    ) == Path("/data/root/data_results/resnet/coco/clip_positive_neuron_data.csv")


def test_results_neuron_data_file_path_uses_sample_type_in_filename():
    path_configs = PathConfigs(NeuroLensConfig(io=IOConfig(root_data_dir_path=Path("/data/root"))))
    target_model = Identified("resnet")
    img_dataset = Identified("coco")
    img_text_model = Identified("clip")

    assert path_configs.data_results_neuron_data_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
        img_text_model=img_text_model,
        sample_type="negative",
    ) == Path("/data/root/data_results/resnet/coco/clip_negative_neuron_data.csv")


def test_results_neuron_data_file_path_uses_configured_filename_template():

    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(
                root_data_dir_path=Path("/data/root"),
                neuron_data_filename="{img_text_model}-{sample_type}.csv",
            )
        )
    )
    target_model = Identified("resnet")
    img_dataset = Identified("coco")
    img_text_model = Identified("clip")

    assert path_configs.data_results_neuron_data_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
        img_text_model=img_text_model,
        sample_type="positive",
    ) == Path("/data/root/data_results/resnet/coco/clip-positive.csv")


def test_custom_parent_and_precomputed_dir_names_are_used():
    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(
                root_data_dir_path=Path("/data/root"),
                precomputed_data_dir_name="precomputed",
                target_model_parent_dir_name="targets",
                img_text_model_parent_dir_name="image_text",
                activations_dir_name="acts",
                img_embds_dir_name="images",
            )
        )
    )
    target_model = Identified("resnet")
    img_text_model = Identified("clip")
    img_dataset = Identified("coco")

    assert path_configs.data_precomp_target_activations_file_path(
        target_model=target_model, img_dataset=img_dataset
    ) == Path("/data/root/precomputed/targets/resnet/acts/coco_activations.zarr")
    assert path_configs.data_precomp_imgtext_img_embds_file_path(
        img_text_model=img_text_model, img_dataset=img_dataset
    ) == Path("/data/root/precomputed/image_text/clip/images/coco_img_embds.zarr")


def test_custom_results_dir_name_is_used():
    path_configs = PathConfigs(
        NeuroLensConfig(
            io=IOConfig(
                root_data_dir_path=Path("/data/root"),
                results_data_dir_name="results",
            )
        )
    )
    target_model = Identified("resnet")
    img_dataset = Identified("coco")
    img_text_model = Identified("clip")

    assert path_configs.data_results_neuron_data_file_path(
        target_model=target_model,
        img_dataset=img_dataset,
        img_text_model=img_text_model,
        sample_type="positive",
    ) == Path("/data/root/results/resnet/coco/clip_positive_neuron_data.csv")
