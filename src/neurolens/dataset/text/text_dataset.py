import logging

from torch.utils.data import Dataset

from neurolens.config import VANILLA_TEMPLATE
from neurolens.utils.str_utils import (
    validate_plain_path_component,
    validate_text_template,
)

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, identifier: str, texts: list[str]) -> None:

        self.identifier = identifier
        self.current_template: str = VANILLA_TEMPLATE

        for i in range(len(texts)):
            validate_plain_path_component(texts[i])

        self.texts = sorted(texts)

        logger.info(f"[Text Dataset {self.identifier!r}] Initialized with {len(self.texts)!r} texts")

    def __len__(self) -> int:
        return len(self.texts)

    def reset_template(self):
        self.set_template(VANILLA_TEMPLATE)

    def set_template(self, template: str):

        validate_text_template(template)

        logger.info(f"[Text Dataset {self.identifier!r}] Setting template to {template.format('text')!r}")

        self.current_template = template

    def __getitem__(self, idx: int) -> str:

        if not 0 <= idx < len(self.texts):
            raise IndexError(
                f"[Text Dataset {self.identifier!r}] Index {idx!r} out of range"
                f" for TextDataset of size {len(self.texts)!r}"
            )
        return self.current_template.format(self.texts[idx])
