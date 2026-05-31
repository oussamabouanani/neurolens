import pytest

from neurolens.utils.str_utils import (
    validate_plain_path_component,
    validate_simple_unique_ordered_fields,
    validate_suffix,
)


def test_validate_simple_unique_ordered_fields_accepts_expected_fields():
    assert (
        validate_simple_unique_ordered_fields(
            "{img_dataset}_{text_dataset}.zarr",
            "img_dataset",
            "text_dataset",
        )
        is None
    )


def test_validate_simple_unique_ordered_fields_accepts_plain_positional_field():
    assert validate_simple_unique_ordered_fields("a photo of {}", "") is None


@pytest.mark.parametrize(
    "value",
    [
        "{text_dataset}_{img_dataset}.zarr",
        "{img_dataset}.zarr",
        "{img_dataset}_{text_dataset}_{text_dataset}.zarr",
        "{img_dataset!r}_{text_dataset}.zarr",
        "{img_dataset:>10}_{text_dataset}.zarr",
    ],
)
def test_validate_simple_unique_ordered_fields_rejects_invalid_fields(value):
    with pytest.raises(ValueError, match="exactly the following fields once"):
        validate_simple_unique_ordered_fields(value, "img_dataset", "text_dataset")


def test_validate_simple_unique_ordered_fields_rejects_malformed_format_string():
    with pytest.raises(ValueError, match="valid parsable string"):
        validate_simple_unique_ordered_fields("{", "img_dataset")


@pytest.mark.parametrize(
    "value",
    [
        "data_raw",
        "{img_dataset}_activations.zarr",
        "a prompt with spaces",
    ],
)
def test_validate_plain_path_component_accepts_plain_names(value):
    assert validate_plain_path_component(value) is None


@pytest.mark.parametrize(
    "value",
    [
        ".",
        "..",
        "nested/path",
        "../outside",
    ],
)
def test_validate_plain_path_component_rejects_paths(value):
    with pytest.raises(ValueError, match="plain name"):
        validate_plain_path_component(value)


def test_validate_plain_path_component_rejects_empty_value():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_plain_path_component("")


def test_validate_suffix_accepts_any_expected_suffix():
    assert validate_suffix(".csv", ".zarr", value="results.csv") is None
    assert validate_suffix("csv", "zarr", value="activations.zarr") is None


def test_validate_suffix_requires_suffixes():
    with pytest.raises(ValueError, match="at least one suffix"):
        validate_suffix(value="results.csv")


def test_validate_suffix_rejects_unexpected_suffix():
    with pytest.raises(ValueError, match="filename must end with"):
        validate_suffix(".csv", value="results.zarr")
