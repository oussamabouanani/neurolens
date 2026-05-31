from pathlib import Path

import numpy as np

from neurolens.utils.path_utils import PathConfigs
from neurolens.utils.str_utils import validate_plain_path_component

from .text_dataset import TextDataset


class TextFileDataset(TextDataset):
    """
    A dataset that reads text from a file.
    The file is expected to contain one text/label per line.
    """

    def __init__(self, identifier: str, file_path: str | Path) -> None:

        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(
                f"[Text Dataset {identifier!r}] File {file_path!r} does not exist or is not a file."
            )

        texts = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                texts.append(line)

        super().__init__(identifier, texts)


def save_augmented_text_dataset(
    identifier_suffix: str,
    generated_text: list[str],
    text_dataset: TextDataset,
    path_configs: PathConfigs,
    parent_dir_name: str,
) -> TextFileDataset:

    validate_plain_path_component(parent_dir_name)

    new_texts = np.array(text_dataset.texts + generated_text)
    new_texts = np.unique(new_texts).tolist()

    new_identifier = f"{text_dataset.identifier}-{identifier_suffix}"
    validate_plain_path_component(new_identifier)

    save_path = path_configs.join(
        path_configs.config.io.raw_data_dir_name,
        parent_dir_name,
        f"{new_identifier}.txt",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        save_path,
        "w",
    ) as f:
        for text in new_texts:
            f.write(f"{text}\n")

    return TextFileDataset(identifier=new_identifier, file_path=save_path)
