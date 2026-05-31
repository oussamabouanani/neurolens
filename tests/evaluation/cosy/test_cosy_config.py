import pytest
from pydantic import ValidationError

from neurolens.evaluation.cosy import DEFAULT_COSY_CONFIG, CoSyConfig, load_cosy_config


def test_default_cosy_config_values():
    config = CoSyConfig()

    assert config.normalize_activations is True
    assert config.enable_img_generation is False
    assert config.stable_diffusion_model_identifier == "stable-diffusion-v1-5"
    assert config.stable_diffusion_model_repo == "runwayml/stable-diffusion-v1-5"
    assert config.generated_imgs_dir_name == "generated_imgs-{img_generator}"
    assert config.generated_img_filename == "{prompt}_{index}.jpg"
    assert config.generated_img_count == 20
    assert (
        config.results_dir_name
        == "cosy_evals-img_gen={img_generator}-sample_count={sample_count}-normalize={normalize_activations}"
    )
    assert config.results_filename == "{score_function}.csv"


def test_default_cosy_config_export_matches_model_defaults():
    assert DEFAULT_COSY_CONFIG == CoSyConfig()


def test_load_cosy_config_accepts_none_dicts_and_instances():
    config = CoSyConfig(generated_img_count=2)

    assert load_cosy_config() == CoSyConfig()
    assert load_cosy_config({"generated_img_count": 3}).generated_img_count == 3
    assert load_cosy_config(config) is config


def test_cosy_config_rejects_unknown_fields_and_is_immutable():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CoSyConfig(unknown_field=True)

    config = CoSyConfig()
    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.generated_img_count = 4


def test_cosy_config_requires_positive_generated_image_count():
    with pytest.raises(ValidationError, match="greater than 0"):
        CoSyConfig(generated_img_count=0)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("stable_diffusion_model_identifier", "nested/model"),
        ("generated_imgs_dir_name", "../generated_imgs-{img_generator}"),
        ("generated_img_filename", "{prompt}/{index}.jpg"),
        (
            "results_dir_name",
            "nested/{img_generator}_{sample_count}_{normalize_activations}",
        ),
        ("results_filename", "../{score_function}.csv"),
    ],
)
def test_cosy_config_rejects_path_values(field_name, value):
    with pytest.raises(ValidationError, match="plain name"):
        CoSyConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("generated_imgs_dir_name", "generated_imgs"),
        ("generated_img_filename", "{index}_{prompt}.jpg"),
        ("results_dir_name", "{sample_count}_{img_generator}_{normalize_activations}"),
        ("results_filename", "results.csv"),
    ],
)
def test_cosy_config_rejects_invalid_template_fields(field_name, value):
    with pytest.raises(ValidationError, match="exactly the following fields once"):
        CoSyConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("generated_img_filename", "{prompt}_{index}.png"),
        ("results_filename", "{score_function}.txt"),
    ],
)
def test_cosy_config_rejects_invalid_suffixes(field_name, value):
    with pytest.raises(ValidationError, match="must end with"):
        CoSyConfig(**{field_name: value})
