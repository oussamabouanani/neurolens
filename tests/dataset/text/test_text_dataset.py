import pytest

from neurolens.dataset.text import TextDataset
from neurolens.utils.str_utils import (
    validate_plain_path_component,
    validate_text_template,
)


def test_sorts_texts_and_applies_default_template():
    dataset = TextDataset("labels", ["zebra", "apple", "banana"])

    assert len(dataset) == 3
    assert dataset.texts == ["apple", "banana", "zebra"]
    assert dataset[0] == "apple"


def test_copies_input_text_order_by_sorting_without_mutating_original_list():
    texts = ["zebra", "apple", "banana"]

    dataset = TextDataset("labels", texts)

    assert texts == ["zebra", "apple", "banana"]
    assert dataset.texts == ["apple", "banana", "zebra"]


def test_set_template_updates_getitem_output():
    dataset = TextDataset("labels", ["cat"])

    dataset.set_template("a photo of {}")

    assert dataset[0] == "a photo of cat"


def test_reset_template_restores_default_getitem_output():
    dataset = TextDataset("labels", ["cat"])

    dataset.set_template("a photo of {}")
    dataset.reset_template()

    assert dataset[0] == "cat"


def test_getitem_raises_for_invalid_index():
    dataset = TextDataset("labels", ["cat"])

    with pytest.raises(IndexError, match="Index 1 out of range"):
        dataset[1]


def test_rejects_text_containing_slash():
    with pytest.raises(ValueError, match="plain name"):
        TextDataset("labels", ["cat/dog"])


def test_validate_plain_path_component_accepts_text_without_slash():
    assert validate_plain_path_component("cat dog") is None


def test_validate_text_template_accepts_one_plain_positional_placeholder():
    assert validate_text_template("a photo of {}") == ["a photo of {}"]


def test_validate_text_template_accepts_list_of_templates():
    assert validate_text_template(["{}", "a photo of {}"]) == ["{}", "a photo of {}"]


def test_validate_text_template_rejects_missing_placeholder():
    with pytest.raises(ValueError, match="exactly the following fields once"):
        validate_text_template("a photo")


def test_validate_text_template_rejects_multiple_placeholders():
    with pytest.raises(ValueError, match="exactly the following fields once"):
        validate_text_template("{} {}")


def test_validate_text_template_rejects_named_placeholder():
    with pytest.raises(ValueError, match="exactly the following fields once"):
        validate_text_template("a photo of {text}")


def test_validate_text_template_rejects_format_spec():
    with pytest.raises(ValueError, match="exactly the following fields once"):
        validate_text_template("{:>10}")


def test_validate_text_template_rejects_escaped_literal_braces():
    with pytest.raises(ValueError, match="exactly the following fields once"):
        validate_text_template("{{}}")


def test_validate_text_template_rejects_malformed_braces():
    with pytest.raises(ValueError):
        validate_text_template("{")
