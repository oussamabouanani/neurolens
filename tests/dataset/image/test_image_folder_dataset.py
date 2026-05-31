import os
from pathlib import Path

import pytest
from PIL import Image

from neurolens.dataset.image.image_folder_dataset import ImageFolderDataset


def save_image(path: Path, mode: str = "RGB") -> None:
    image = Image.new(mode, (3, 2), color=(10, 20, 30))
    image.save(path)


def test_discovers_images_from_relative_directory_and_loads_rgb(tmp_path: Path):
    original_cwd = Path.cwd()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "nested").mkdir()

    save_image(image_dir / "a.JPEG")
    save_image(image_dir / "b.png")
    (image_dir / "notes.txt").write_text("not an image")
    save_image(image_dir / "nested" / "ignored.jpg")

    try:
        os.chdir(tmp_path)
        dataset = ImageFolderDataset(
            "relative",
            "images",
            mean=(0.1, 0.2, 0.3),
            std=(0.4, 0.5, 0.6),
        )
    finally:
        os.chdir(original_cwd)

    assert len(dataset) == 2
    assert dataset.image_files == [Path("images") / "a.JPEG", Path("images") / "b.png"]

    try:
        os.chdir(tmp_path)
        image = dataset[0]
    finally:
        os.chdir(original_cwd)

    assert image.mode == "RGB"
    assert image.size == (3, 2)


def test_string_extensions_are_normalized(tmp_path: Path):
    save_image(tmp_path / "a.JPG")
    save_image(tmp_path / "b.png")
    save_image(tmp_path / "ignored.jpeg")

    dataset = ImageFolderDataset(
        "extensions",
        tmp_path,
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
        img_file_extensions=" jpg,PNG ",
    )

    assert len(dataset) == 2
    assert [path.name for path in dataset.image_files] == ["a.JPG", "b.png"]


def test_list_extensions_are_not_mutated(tmp_path: Path):
    extensions = ["jpg", "PNG"]
    save_image(tmp_path / "a.jpg")

    ImageFolderDataset(
        "immutable",
        tmp_path,
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
        img_file_extensions=extensions,
    )

    assert extensions == ["jpg", "PNG"]


def test_getitem_raises_for_invalid_index(tmp_path: Path):
    save_image(tmp_path / "a.png")

    dataset = ImageFolderDataset(
        "index",
        tmp_path,
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
    )

    with pytest.raises(IndexError, match="Index 1 out of range"):
        dataset[1]


def test_getitem_adds_context_to_image_open_errors(tmp_path: Path):
    corrupt_image = tmp_path / "corrupt.png"
    corrupt_image.write_text("not a real image")

    dataset = ImageFolderDataset(
        "corrupt",
        tmp_path,
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
    )

    with pytest.raises(
        OSError,
        match=r"\[Image Dataset 'corrupt'\] Could not open image file",
    ) as context:
        dataset[0]

    assert context.value.__cause__ is not None


def test_empty_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No image files found"):
        ImageFolderDataset(
            "empty",
            tmp_path,
            mean=(0.1, 0.2, 0.3),
            std=(0.4, 0.5, 0.6),
        )
