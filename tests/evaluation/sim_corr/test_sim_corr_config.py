import pytest
from pydantic import ValidationError

from neurolens.evaluation.sim_corr import (
    DEFAULT_SIM_CORR_CONFIG,
    SimCorrConfig,
    load_sim_corr_config,
)


def test_default_sim_corr_config_values():
    config = SimCorrConfig()

    assert config.weighted is False
    assert config.topk_text == 1
    assert (
        config.results_dir_name
        == "sim_corr-simulator={simulator}-sample_count={sample_count}-weighted={weighted}-topk_text={topk_text}"
    )
    assert config.results_filename == "{score_function}.csv"


def test_default_sim_corr_config_export_matches_model_defaults():
    assert DEFAULT_SIM_CORR_CONFIG == SimCorrConfig()


def test_load_sim_corr_config_accepts_none_dicts_and_instances():
    config = SimCorrConfig(topk_text=2)

    assert load_sim_corr_config() == SimCorrConfig()
    assert load_sim_corr_config({"topk_text": 3}).topk_text == 3
    assert load_sim_corr_config(config) is config


def test_sim_corr_config_rejects_unknown_fields_and_is_immutable():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SimCorrConfig(unknown_field=True)

    config = SimCorrConfig()
    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.topk_text = 4


def test_sim_corr_config_requires_positive_topk_text():
    with pytest.raises(ValidationError, match="greater than 0"):
        SimCorrConfig(topk_text=0)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("results_dir_name", "../{simulator}_{sample_count}_{weighted}_{topk_text}"),
        ("results_filename", "../{score_function}.csv"),
    ],
)
def test_sim_corr_config_rejects_path_values(field_name, value):
    with pytest.raises(ValidationError, match="plain name"):
        SimCorrConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("results_dir_name", "{sample_count}_{simulator}_{weighted}_{topk_text}"),
        ("results_filename", "results.csv"),
    ],
)
def test_sim_corr_config_rejects_invalid_template_fields(field_name, value):
    with pytest.raises(ValidationError, match="exactly the following fields once"):
        SimCorrConfig(**{field_name: value})


def test_sim_corr_config_rejects_invalid_suffixes():
    with pytest.raises(ValidationError, match="must end with"):
        SimCorrConfig(results_filename="{score_function}.txt")
