from typing import Callable

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from PIL import Image
from torch import Tensor


def is_l2_normalized(x: Tensor, dim: int, atol: float = 1e-6) -> bool:
    norms = torch.linalg.vector_norm(x, ord=2, dim=dim)
    return torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=atol,
    )


ImageProcessor = Callable[
    [Image.Image],
    Float[Tensor, "channel height width"],
]


def get_image_features(
    image_processor: ImageProcessor,
    images: Image.Image | list[Image.Image],
) -> Float[Tensor, "batch channel height width"]:

    if isinstance(images, list):
        if len(images) == 0:
            raise ValueError("image list can not be empty")
        elif not all(isinstance(img, Image.Image) for img in images):
            raise TypeError("One of the elements in the list is not a PIL image list!")
    elif isinstance(images, Image.Image):
        images = [images]
    else:
        raise TypeError("images must be a single PIL image or a list of PIL images")

    img_features = torch.stack([image_processor(img) for img in images], dim=0)

    if not isinstance(img_features, Tensor):
        raise TypeError("image processor must return a tensor")
    elif img_features.ndim != 4:
        raise ValueError("image features must have 4 dimensions")

    return img_features


def transform_weights(
    weights: Tensor,
    weighting: str = "raw",
) -> Tensor:

    if weighting == "none":
        return torch.ones_like(weights)
    if weighting == "raw":
        return weights
    if weighting == "squared":
        return weights.square()
    if weighting == "sqrt":
        return weights.clamp_min(0).sqrt()
    if weighting == "log":
        return torch.log1p(weights)

    raise ValueError(f"Unknown weighting scheme: {weighting}")


def get_weighted_average(
    x: Float[Tensor, "batch n_sample dim"],
    weights: Float[Tensor, "batch n_sample"],
    weighting: str = "raw",
    l2_normalise: bool = True,
) -> Float[Tensor, "batch dim"]:

    weights = transform_weights(weights, weighting)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    weighted_avg = einops.einsum(weights, x, "batch n_sample, batch n_sample dim -> batch dim")

    return F.normalize(weighted_avg, dim=-1, p=2) if l2_normalise else weighted_avg


def generate_splits(
    size: int,
    splits: tuple[float, float, float],
    seed: int | None = None,
):
    if not torch.isclose(torch.tensor(sum(splits)), torch.tensor(1.0)):
        raise ValueError(f"splits must sum to 1.0, got {sum(splits)}")

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    prem_indices = torch.randperm(size, generator=generator)

    n_train = int(size * splits[0])
    n_val = int(size * splits[1])

    train_indices = prem_indices[:n_train]
    val_indices = prem_indices[n_train : n_train + n_val]
    test_indices = prem_indices[n_train + n_val :]

    return train_indices, val_indices, test_indices
