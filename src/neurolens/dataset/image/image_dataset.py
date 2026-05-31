import logging
from abc import ABC, abstractmethod

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ImageDataset(Dataset, ABC):
    """Base class for image datasets"""

    def __init__(
        self,
        identifier: str,
    ) -> None:

        self.identifier = identifier

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_mean(self) -> tuple[float, float, float]:
        """Returns the mean of the dataset"""
        raise NotImplementedError

    @abstractmethod
    def get_std(self) -> tuple[float, float, float]:
        """Returns the standard deviation of the dataset"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Image.Image:
        raise NotImplementedError
