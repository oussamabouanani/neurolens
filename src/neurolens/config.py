from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from neurolens.utils.str_utils import (
    validate_plain_path_component,
    validate_simple_unique_ordered_fields,
    validate_suffix,
    validate_text_template,
)


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, frozen=True)


class IOConfig(FrozenConfigModel):
    root_data_dir_path: Path = Path(".")
    raw_data_dir_name: str = "data_raw"
    precomputed_data_dir_name: str = "data_precomp"
    results_data_dir_name: str = "data_results"

    target_model_parent_dir_name: str = "target_models"
    img_text_model_parent_dir_name: str = "img_text_models"

    neuron_data_filename: str = "{img_text_model}_{sample_type}_neuron_data.csv"

    activations_dir_name: str = "activations"
    activations_filename: str = "{img_dataset}_activations.zarr"

    img_embds_dir_name: str = "img_embds"
    img_embds_filename: str = "{img_dataset}_img_embds.zarr"

    text_embds_dir_name: str = "text_embds"
    text_embds_filename: str = "{text_dataset}_{template}_text_embds.zarr"

    sim_mat_dir_name: str = "sim_mat"
    sim_mat_filename: str = "{img_dataset}_{text_dataset}_sim_mat.zarr"

    dataset_split_dir_name: str = "dataset_splits-{img_dataset}"

    # used for zarr stores
    zarr_batch_save_count: int = Field(default=128, gt=0)

    @field_validator("neuron_data_filename")
    @classmethod
    def validate_neuron_data_filename(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "img_text_model", "sample_type")
        return value

    @field_validator("activations_filename")
    @classmethod
    def validate_activations_filename(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "img_dataset")
        return value

    @field_validator("img_embds_filename")
    @classmethod
    def validate_img_embds_filename(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "img_dataset")
        return value

    @field_validator("text_embds_filename")
    @classmethod
    def validate_text_embds_filename(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "text_dataset", "template")
        return value

    @field_validator("sim_mat_filename")
    @classmethod
    def validate_sim_mat_filename(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "img_dataset", "text_dataset")
        return value

    @field_validator("dataset_split_dir_name")
    @classmethod
    def validate_dataset_split_dir_name(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "img_dataset")
        return value

    @field_validator(
        "raw_data_dir_name",
        "precomputed_data_dir_name",
        "results_data_dir_name",
        "target_model_parent_dir_name",
        "img_text_model_parent_dir_name",
        "neuron_data_filename",
        "activations_filename",
        "img_embds_filename",
        "text_embds_filename",
        "sim_mat_filename",
        "activations_dir_name",
        "img_embds_dir_name",
        "text_embds_dir_name",
        "sim_mat_dir_name",
        "neuron_data_filename",
        "activations_filename",
        "img_embds_filename",
        "text_embds_filename",
        "sim_mat_filename",
        "dataset_split_dir_name",
    )
    @classmethod
    def validate_plain_path(cls, value: str) -> str:
        validate_plain_path_component(value)
        return value

    @field_validator("neuron_data_filename")
    @classmethod
    def validate_csv_filename_extension(cls, value: str) -> str:
        validate_suffix(".csv", value=value)
        return value

    @field_validator(
        "activations_filename",
        "img_embds_filename",
        "text_embds_filename",
        "sim_mat_filename",
    )
    @classmethod
    def validate_zarr_filename_extension(cls, value: str) -> str:
        validate_suffix(".zarr", value=value)
        return value


VANILLA_TEMPLATE = "{}"


class DatasetConfig(FrozenConfigModel):
    vanilla_template: str = VANILLA_TEMPLATE
    templates: list[str] = Field(
        default=[
            "{}",
            "{}-like",
            "a {}",
            "an image of a {}",
            "a closeup of a {}",
            "an image of a closeup of {}",
        ]
    )

    similarity_matrix_use_vanilla_template: bool = True

    precomp_batch_size: int = Field(default=128, gt=0)

    @field_validator("vanilla_template")
    @classmethod
    def validate_vanilla_template(cls, value: str) -> str:
        validate_text_template(value)
        return value

    @field_validator("templates")
    @classmethod
    def validate_templates(cls, value: list[str]) -> list[str]:

        if len(value) == 0:
            raise ValueError("at least one template is required")

        validate_text_template(value)
        return value


class NeuronDataConfig(FrozenConfigModel):
    topk_neurons: int = Field(default=200, gt=0)
    required_sample_count: int = Field(default=100, gt=0)
    mean_top_count: int = Field(default=100, gt=0)

    ignore_overly_active_neurons: bool = True
    overly_active_neuron_rate: float = Field(default=0.8, gt=0, lt=1)

    neg_fetch_batch_size: int = Field(default=4096, gt=0)
    allow_neg_reuse: bool = True


class EvaluationConfig(FrozenConfigModel):
    topk_text: int = Field(default=5, gt=0)
    sample_count: int = Field(default=30, gt=0)


class VLMConfig(FrozenConfigModel):
    img_count: int = Field(default=9, gt=0)
    grid_row_count: int = Field(default=3, gt=0)
    grid_size: int = Field(default=128, gt=0)

    max_text_length: int = Field(default=40, gt=0)

    @field_validator("grid_row_count")
    @classmethod
    def validate_grid_row_count(cls, value: int, info: ValidationInfo) -> int:

        if "img_count" not in info.data:
            return value

        img_count = info.data["img_count"]

        if value > img_count:
            raise ValueError(f"grid_row_count ({value}) must be <= img_count ({img_count})")

        if img_count % value != 0:
            raise ValueError(f"img_count ({img_count}) must be divisible by grid_row_count ({value})")

        return value

    @field_validator("grid_size")
    @classmethod
    def validate_grid_size(cls, value: int, info: ValidationInfo) -> int:

        if "img_count" not in info.data or "grid_row_count" not in info.data:
            return value

        img_count = info.data["img_count"]
        grid_row_count = info.data["grid_row_count"]

        if value < img_count // grid_row_count:
            raise ValueError(
                f"grid_size ({value}) must be >= img_count // grid_row_count ({img_count // grid_row_count})"
            )

        if value < grid_row_count:
            raise ValueError(f"grid_size ({value}) must be >= grid_row_count ({grid_row_count})")

        return value

    pos_only_prompt: str = (
        "Find three visual concepts that is shared by ALL images.\n"
        "- The concept should describe a visual object class, property, style, or action\n"
        "- Use at most 4 words per concept\n"
        "- Output EXACTLY three features, separated by commas\n"
        "Answer with the three concepts only."
        "Examples:\n"
        "German Shepard, red background, black color, dark spots,"
        " many objects, long fur, small white bubbles, short arm\n"
        "\n"
        "The three concepts that are present in all images are: "
    )

    # <image> is the image token for some VLMs (e.g. InternVL3)
    pos_neg_prompt: str = (
        "Image-1: <image>\n"
        "Image-2: <image>\n\n"
        "You are shown two sets of images:\n"
        "- Image-1: images of interest (positives)\n"
        "- Image-2: contrastive images (negatives)\n\n"
        "Find three shared visual concepts that is present in all Image-1 images and absent from Image-2 images. "
        "The concepts should only fit to Image-1 images, not Image-2 images."
        "\n"
        "- The concepts should describe a visual object class, property, style, or action\n"
        "- Use at most 4 words per concept\n"
        "- Output EXACTLY three features, separated by commas\n"
        "Answer with the three concepts only."
        "Examples:\n"
        "German Shepard, red background, black color, dark spots,"
        " many objects, long fur, small white bubbles, short arm\n"
        "\n"
        "The three concepts that are present in all Image-1 images but missing in Image-2 images are: "
    )


class NeuroLensConfig(FrozenConfigModel):
    io: IOConfig = Field(default_factory=IOConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    neuron_data: NeuronDataConfig = Field(default_factory=NeuronDataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)


DEFAULT_CONFIG: Final = NeuroLensConfig()


def load_config(
    config: NeuroLensConfig | dict[str, Any] | None = None,
) -> NeuroLensConfig:
    if isinstance(config, NeuroLensConfig):
        return config

    return NeuroLensConfig.model_validate({} if config is None else config)
