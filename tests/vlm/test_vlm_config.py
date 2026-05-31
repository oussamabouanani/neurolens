import pytest
from pydantic import ValidationError

from neurolens.config import VLMConfig


def test_vlm_config_default_values():
    config = VLMConfig()

    assert config.img_count == 9
    assert config.grid_row_count == 3
    assert config.grid_size == 128
    assert config.max_text_length == 40


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"img_count": 0}, "greater than 0"),
        ({"grid_row_count": 0}, "greater than 0"),
        ({"grid_size": 0}, "greater than 0"),
        ({"max_text_length": 0}, "greater than 0"),
        (
            {"img_count": 2, "grid_row_count": 3},
            "grid_row_count .* must be <= img_count",
        ),
        (
            {"img_count": 10, "grid_row_count": 3},
            "img_count .* must be divisible by grid_row_count",
        ),
        (
            {"img_count": 9, "grid_row_count": 3, "grid_size": 2},
            "grid_size .* must be >= img_count // grid_row_count",
        ),
        (
            {"img_count": 3, "grid_row_count": 3, "grid_size": 2},
            "grid_size .* must be >= grid_row_count",
        ),
    ],
)
def test_vlm_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        VLMConfig(**kwargs)


def test_vlm_config_accepts_smallest_valid_grid_size():
    config = VLMConfig(img_count=9, grid_row_count=3, grid_size=3)

    assert config.grid_size == 3
