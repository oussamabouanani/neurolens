from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from neurolens.utils.str_utils import (
    validate_plain_path_component,
    validate_simple_unique_ordered_fields,
    validate_suffix,
)


class CoSyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, frozen=True)

    normalize_activations: bool = True

    enable_img_generation: bool = False
    stable_diffusion_model_identifier: str = "stable-diffusion-v1-5"
    stable_diffusion_model_repo: str = "runwayml/stable-diffusion-v1-5"
    generated_imgs_dir_name: str = "generated_imgs-{img_generator}"
    generated_img_filename: str = "{prompt}_{index}.jpg"
    generated_img_count: int = Field(default=20, gt=0)

    results_dir_name: str = (
        "cosy_evals-img_gen={img_generator}-sample_count={sample_count}-normalize={normalize_activations}"
    )
    results_filename: str = "{score_function}.csv"

    @field_validator("generated_imgs_dir_name")
    @classmethod
    def validate_generated_imgs_dir_name(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "img_generator")
        return value

    @field_validator("generated_img_filename")
    @classmethod
    def validate_generated_img_filename(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "prompt", "index")
        validate_suffix(".jpg", value=value)
        return value

    @field_validator("results_dir_name")
    @classmethod
    def validate_results_dir_name(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "img_generator", "sample_count", "normalize_activations")
        return value

    @field_validator("results_filename")
    @classmethod
    def validate_results_filename(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "score_function")
        validate_suffix(".csv", value=value)
        return value

    @field_validator(
        "stable_diffusion_model_identifier",
        "generated_imgs_dir_name",
        "generated_img_filename",
        "results_dir_name",
        "results_filename",
    )
    @classmethod
    def validate_plain_path(cls, value: str) -> str:
        validate_plain_path_component(value)
        return value


DEFAULT_COSY_CONFIG: Final = CoSyConfig()


def load_config(
    config: CoSyConfig | dict[str, Any] | None = None,
) -> CoSyConfig:
    if isinstance(config, CoSyConfig):
        return config

    return CoSyConfig.model_validate({} if config is None else config)
