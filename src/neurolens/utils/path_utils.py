from pathlib import Path
from typing import Protocol

from neurolens.config import NeuroLensConfig


class HasIdentifier(Protocol):
    identifier: str


class PathConfigs:
    def __init__(self, config: NeuroLensConfig):

        self.config = config

    def join(self, *paths: str | Path) -> Path:
        return self.config.io.root_data_dir_path.joinpath(*paths)

    def data_precomp_target_activations_file_path(
        self, *, target_model: HasIdentifier, img_dataset: HasIdentifier
    ) -> Path:

        return self.join(
            self.config.io.precomputed_data_dir_name,
            self.config.io.target_model_parent_dir_name,
            target_model.identifier,
            self.config.io.activations_dir_name,
            self.config.io.activations_filename.format(img_dataset=img_dataset.identifier),
        )

    def data_precomp_imgtext_img_embds_file_path(
        self, *, img_text_model: HasIdentifier, img_dataset: HasIdentifier
    ) -> Path:

        return self.join(
            self.config.io.precomputed_data_dir_name,
            self.config.io.img_text_model_parent_dir_name,
            img_text_model.identifier,
            self.config.io.img_embds_dir_name,
            self.config.io.img_embds_filename.format(img_dataset=img_dataset.identifier),
        )

    def data_precomp_imgtext_text_embds_file_path(
        self,
        *,
        img_text_model: HasIdentifier,
        text_dataset: HasIdentifier,
        template: str,
    ) -> Path:

        return self.join(
            self.config.io.precomputed_data_dir_name,
            self.config.io.img_text_model_parent_dir_name,
            img_text_model.identifier,
            self.config.io.text_embds_dir_name,
            self.config.io.text_embds_filename.format(
                text_dataset=text_dataset.identifier,
                template=template,
            ),
        )

    def data_precomp_imgtext_sim_matrix_file_path(
        self,
        *,
        img_text_model: HasIdentifier,
        img_dataset: HasIdentifier,
        text_dataset: HasIdentifier,
    ) -> Path:

        return self.join(
            self.config.io.precomputed_data_dir_name,
            self.config.io.img_text_model_parent_dir_name,
            img_text_model.identifier,
            self.config.io.sim_mat_dir_name,
            self.config.io.sim_mat_filename.format(
                img_dataset=img_dataset.identifier,
                text_dataset=text_dataset.identifier,
            ),
        )

    def data_precomp_dataset_splits_file_path(
        self,
        *,
        img_dataset: HasIdentifier,
        split: str,
    ) -> Path:

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}, must be one of ['train', 'val', 'test']")

        return self.join(
            self.config.io.raw_data_dir_name,
            self.config.io.dataset_split_dir_name.format(
                img_dataset=img_dataset.identifier,
            ),
            f"{split}_ids.pt",
        )

    def data_results_neuron_data_file_path(
        self,
        *,
        target_model: HasIdentifier,
        img_dataset: HasIdentifier,
        img_text_model: HasIdentifier,
        sample_type: str,
    ) -> Path:

        return self.join(
            self.config.io.results_data_dir_name,
            target_model.identifier,
            img_dataset.identifier,
            self.config.io.neuron_data_filename.format(
                img_text_model=img_text_model.identifier,
                sample_type=sample_type,
            ),
        )
