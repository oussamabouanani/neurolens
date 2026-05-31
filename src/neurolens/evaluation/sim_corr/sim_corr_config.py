from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from neurolens.utils.str_utils import (
    validate_plain_path_component,
    validate_simple_unique_ordered_fields,
    validate_suffix,
)


class SimCorrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, frozen=True)

    weighted: bool = False
    topk_text: int = Field(default=1, gt=0)

    results_dir_name: str = (
        "sim_corr-simulator={simulator}-sample_count={sample_count}-weighted={weighted}-topk_text={topk_text}"
    )
    results_filename: str = "{score_function}.csv"

    @field_validator("results_dir_name")
    @classmethod
    def validate_results_dir_name(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "simulator", "sample_count", "weighted", "topk_text")
        return value

    @field_validator("results_filename")
    @classmethod
    def validate_results_filename(cls, value: str) -> str:
        validate_simple_unique_ordered_fields(value, "score_function")
        validate_suffix(".csv", value=value)
        return value

    @field_validator(
        "results_dir_name",
        "results_filename",
    )
    @classmethod
    def validate_plain_path(cls, value: str) -> str:
        validate_plain_path_component(value)
        return value


DEFAULT_SIM_CORR_CONFIG: Final = SimCorrConfig()


def load_config(
    config: SimCorrConfig | dict[str, Any] | None = None,
) -> SimCorrConfig:
    if isinstance(config, SimCorrConfig):
        return config

    return SimCorrConfig.model_validate({} if config is None else config)
