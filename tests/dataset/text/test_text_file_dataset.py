from pathlib import Path

import pytest

from neurolens.config import IOConfig, NeuroLensConfig
from neurolens.dataset.text import (
    TextDataset,
    TextFileDataset,
    save_augmented_text_dataset,
)
from neurolens.utils.path_utils import PathConfigs


def test_loads_non_empty_lines_from_file(tmp_path: Path):
    file_path = tmp_path / "labels.txt"
    file_path.write_text("zebra\n\n apple \nbanana\n", encoding="utf-8")

    dataset = TextFileDataset("labels", file_path)

    assert len(dataset) == 3
    assert dataset.texts == ["apple", "banana", "zebra"]
    assert dataset[0] == "apple"


def test_accepts_string_file_path(tmp_path: Path):
    file_path = tmp_path / "labels.txt"
    file_path.write_text("cat\n", encoding="utf-8")

    dataset = TextFileDataset("labels", str(file_path))

    assert dataset.texts == ["cat"]


def test_missing_file_raises(tmp_path: Path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        TextFileDataset("labels", file_path)


def test_file_text_validation_is_applied(tmp_path: Path):
    file_path = tmp_path / "labels.txt"
    file_path.write_text("cat/dog\n", encoding="utf-8")

    with pytest.raises(ValueError, match="plain name"):
        TextFileDataset("labels", file_path)


def test_save_augmented_text_dataset_writes_unique_texts_and_returns_dataset(
    tmp_path: Path,
):
    config = NeuroLensConfig(io=IOConfig(root_data_dir_path=tmp_path))
    path_configs = PathConfigs(config)
    text_dataset = TextDataset("labels", ["zebra", "apple"])

    dataset = save_augmented_text_dataset(
        identifier_suffix="generated",
        generated_text=["banana", "apple", "zebra"],
        text_dataset=text_dataset,
        path_configs=path_configs,
        parent_dir_name="augmented",
    )

    save_path = tmp_path / "data_raw" / "augmented" / "labels-generated.txt"
    assert save_path.read_text(encoding="utf-8") == "apple\nbanana\nzebra\n"
    assert dataset.identifier == "labels-generated"
    assert dataset.texts == ["apple", "banana", "zebra"]


def test_save_augmented_text_dataset_rejects_path_parent_dir_name(tmp_path: Path):
    config = NeuroLensConfig(io=IOConfig(root_data_dir_path=tmp_path))

    with pytest.raises(ValueError, match="plain name"):
        save_augmented_text_dataset(
            identifier_suffix="generated",
            generated_text=["banana"],
            text_dataset=TextDataset("labels", ["apple"]),
            path_configs=PathConfigs(config),
            parent_dir_name="nested/augmented",
        )
