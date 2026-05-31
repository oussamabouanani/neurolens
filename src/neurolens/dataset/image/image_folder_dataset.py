import logging
from pathlib import Path

from PIL import Image

from .image_dataset import ImageDataset

logger = logging.getLogger(__name__)


class ImageFolderDataset(ImageDataset):
    """
    A dataset of images stored in a folder.
    Unlike torchvision.datasets.ImageFolder, this dataset does not support subfolders.
    """

    def __init__(
        self,
        identifier: str,
        root_dir_path: str | Path,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        img_file_extensions: str | tuple[str, ...] | list[str] = (
            ".JPEG",
            ".jpg",
            ".png",
        ),
    ):
        super().__init__(identifier)

        root_dir_path = Path(root_dir_path)
        if not root_dir_path.is_dir():
            raise FileNotFoundError(
                f"[Image Dataset {identifier!r}] Directory '{root_dir_path}' does not exist or is not a directory."
            )

        if isinstance(img_file_extensions, str):
            normalized_extensions = img_file_extensions.split(",")
        else:
            normalized_extensions = list(img_file_extensions)

        for i, extension in enumerate(normalized_extensions):
            extension = extension.strip().lower()
            if not extension.startswith("."):
                extension = f".{extension}"
            normalized_extensions[i] = extension

        self.image_files = sorted(
            [f for f in root_dir_path.iterdir() if f.is_file() and f.suffix.lower() in normalized_extensions]
        )

        if len(self.image_files) == 0:
            raise FileNotFoundError(
                f"[Image Dataset {self.identifier!r}] No image files found in directory {root_dir_path!r}."
            )

        self.mean = mean
        self.std = std

        logger.info(
            f"[Image Dataset {self.identifier!r}]: {len(self.image_files)!r}"
            " image files found in directory {root_dir_path!r}."
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def get_mean(self) -> tuple[float, float, float]:
        return self.mean

    def get_std(self) -> tuple[float, float, float]:
        return self.std

    def __getitem__(self, idx: int) -> Image.Image:

        if not 0 <= idx < len(self.image_files):
            raise IndexError(
                f"[Image Dataset {self.identifier!r}] Index {idx!r} out of range"
                f" for ImageDataset of size {len(self.image_files)!r}"
            )

        img_path: Path = self.image_files[idx]

        try:
            with Image.open(img_path) as image:
                return image.convert("RGB")
        except OSError as e:
            raise OSError(f"[Image Dataset {self.identifier!r}] Could not open image file at {img_path!r}: {e}") from e
