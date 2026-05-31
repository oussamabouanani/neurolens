import pytest
import torch
from PIL import Image

from neurolens.config import IOConfig, NeuroLensConfig
from neurolens.evaluation.cosy import CoSyConfig
from neurolens.evaluation.cosy.stable_diffusion_img_generator import (
    StableDiffusionImageGenerator,
)
from neurolens.utils.path_utils import PathConfigs


@pytest.fixture
def generator(tmp_path):
    config = NeuroLensConfig(io=IOConfig(root_data_dir_path=tmp_path))
    cosy_config = CoSyConfig(
        stable_diffusion_model_identifier="sd-test",
        generated_img_count=3,
    )
    return StableDiffusionImageGenerator(
        config=config,
        cosy_config=cosy_config,
        path_configs=PathConfigs(config),
        device="cpu",
    )


def test_constructor_creates_generated_image_directory(generator, tmp_path):
    assert generator.generated_img_dir_path == tmp_path / "data_raw/generated_imgs-sd-test"
    assert generator.generated_img_dir_path.is_dir()


def test_get_generated_image_path_uses_prompt_subdirectory(generator, tmp_path):
    assert generator.get_generated_image_path("red chair", 2) == (
        tmp_path / "data_raw/generated_imgs-sd-test/red chair/red chair_2.jpg"
    )


def test_count_existing_prompt_imgs_stops_at_first_missing_index(generator):
    for index in [0, 1, 3]:
        path = generator.get_generated_image_path("prompt", index)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path)

    assert generator.count_existing_prompt_imgs("prompt") == 2


def test_generate_images_reuses_existing_images_and_saves_remaining(generator):
    existing_path = generator.get_generated_image_path("prompt", 0)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(existing_path)

    class DummyPipe:
        def __call__(self, prompts):
            self.prompts = prompts
            return type(
                "Result",
                (),
                {
                    "images": [
                        Image.new("RGB", (2, 2), color=(0, 255, 0)),
                        Image.new("RGB", (2, 2), color=(0, 0, 255)),
                    ]
                },
            )()

    pipe = DummyPipe()
    generator.pipe = pipe

    generator.generate_images("prompt")

    assert pipe.prompts == ["prompt", "prompt"]
    assert generator.get_generated_image_path("prompt", 1).is_file()
    assert generator.get_generated_image_path("prompt", 2).is_file()


def test_generate_images_rejects_prompt_path_components(generator):
    with pytest.raises(ValueError, match="plain name"):
        generator.generate_images("../prompt")


def test_generate_images_skips_pipeline_when_enough_images_exist(generator):
    for index in range(generator.cosy_config.generated_img_count):
        path = generator.get_generated_image_path("prompt", index)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path)

    generator.generate_images("prompt")

    assert generator.pipe is None


def test_load_images_returns_non_black_images_and_warns(generator, caplog):
    for index, color in enumerate([(255, 0, 0), (0, 0, 0), (0, 255, 0)]):
        path = generator.get_generated_image_path("prompt", index)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color=color).save(path)

    images = generator.load_images("prompt")

    assert len(images) == 2
    assert "Only 2 out of 3 image files" in caplog.text


def test_load_images_requires_all_expected_files(generator):
    path = generator.get_generated_image_path("prompt", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path)

    with pytest.raises(FileNotFoundError, match="index 1"):
        generator.load_images("prompt")


def test_unload_pipe_releases_pipeline(generator, monkeypatch):
    synchronized = []
    emptied = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronized.append(True))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))

    generator.pipe = object()
    generator._unload_pipe()

    assert generator.pipe is None
    assert synchronized == [True]
    assert emptied == [True]
