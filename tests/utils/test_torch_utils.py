import pytest
import torch
from PIL import Image

from neurolens.utils.torch_utils import (
    generate_splits,
    get_image_features,
    get_weighted_average,
    is_l2_normalized,
    transform_weights,
)


def image_to_tensor_processor(image):
    return torch.zeros((3, image.height, image.width))


def test_is_l2_normalized_accepts_unit_vectors_along_dim():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [3.0 / 5.0, 4.0 / 5.0],
        ]
    )

    assert is_l2_normalized(x, dim=1)


def test_is_l2_normalized_rejects_non_unit_vectors():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert not is_l2_normalized(x, dim=1)


def test_is_l2_normalized_rejects_zero_vectors():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    assert not is_l2_normalized(x, dim=1)


def test_is_l2_normalized_respects_dim_argument():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
        ]
    )

    assert not is_l2_normalized(x, dim=0)
    assert not is_l2_normalized(x, dim=1)

    y = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    assert is_l2_normalized(y, dim=0)
    assert is_l2_normalized(y, dim=1)


def test_is_l2_normalized_respects_absolute_tolerance():
    x = torch.tensor([[1.0 + 5e-5, 0.0]])

    assert is_l2_normalized(x, dim=1, atol=1e-4)
    assert not is_l2_normalized(x, dim=1, atol=1e-7)


def test_is_l2_normalized_handles_higher_rank_tensors():
    x = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0 / 5.0, 4.0 / 5.0], [5.0 / 13.0, 12.0 / 13.0]],
        ]
    )

    assert is_l2_normalized(x, dim=2)


@pytest.mark.parametrize(
    "weighting, expected",
    [
        ("none", [[1.0, 1.0, 1.0]]),
        ("raw", [[0.0, 4.0, 9.0]]),
        ("squared", [[0.0, 16.0, 81.0]]),
        ("sqrt", [[0.0, 2.0, 3.0]]),
        ("log", [[0.0, 1.6094379, 2.3025851]]),
    ],
)
def test_transform_weights_applies_weighting_scheme(weighting, expected):
    weights = torch.tensor([[0.0, 4.0, 9.0]])

    assert torch.allclose(transform_weights(weights, weighting), torch.tensor(expected))


def test_transform_weights_clamps_negative_values_for_sqrt():
    weights = torch.tensor([[-4.0, 0.0, 9.0]])

    assert torch.equal(
        transform_weights(weights, "sqrt"),
        torch.tensor([[0.0, 0.0, 3.0]]),
    )


def test_transform_weights_rejects_unknown_weighting_scheme():
    with pytest.raises(ValueError, match="Unknown weighting scheme"):
        transform_weights(torch.ones((1, 2)), "mystery")


def test_get_weighted_average_returns_l2_normalized_vectors_by_default():
    x = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 3.0]],
        ]
    )
    weights = torch.tensor([[3.0, 1.0], [1.0, 1.0]])

    result = get_weighted_average(x, weights)
    expected_raw = torch.tensor([[1.5, 0.25], [0.5, 1.5]])
    expected = torch.nn.functional.normalize(expected_raw, dim=-1, p=2)

    assert torch.allclose(result, expected)
    assert is_l2_normalized(result, dim=-1)


def test_get_weighted_average_can_skip_l2_normalization():
    x = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    weights = torch.tensor([[3.0, 1.0]])

    result = get_weighted_average(x, weights, l2_normalise=False)

    assert torch.allclose(result, torch.tensor([[1.5, 0.25]]))
    assert not is_l2_normalized(result, dim=-1)


def test_get_weighted_average_applies_weighting_before_averaging():
    x = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    weights = torch.tensor([[3.0, 1.0]])

    result = get_weighted_average(x, weights, weighting="none", l2_normalise=False)

    assert torch.allclose(result, torch.tensor([[1.0, 0.5]]))


def test_generate_splits_returns_disjoint_exhaustive_partitions():
    train_indices, val_indices, test_indices = generate_splits(
        size=10,
        splits=(0.6, 0.2, 0.2),
        seed=0,
    )

    assert len(train_indices) == 6
    assert len(val_indices) == 2
    assert len(test_indices) == 2

    all_indices = torch.cat([train_indices, val_indices, test_indices])
    assert torch.equal(torch.sort(all_indices).values, torch.arange(10))
    assert len(torch.unique(all_indices)) == 10


def test_generate_splits_assigns_rounding_remainder_to_test_split():
    train_indices, val_indices, test_indices = generate_splits(
        size=11,
        splits=(0.6, 0.2, 0.2),
        seed=0,
    )

    assert len(train_indices) == 6
    assert len(val_indices) == 2
    assert len(test_indices) == 3


def test_generate_splits_is_reproducible_with_seed():
    first = generate_splits(size=20, splits=(0.5, 0.25, 0.25), seed=123)
    second = generate_splits(size=20, splits=(0.5, 0.25, 0.25), seed=123)

    assert all(torch.equal(first_split, second_split) for first_split, second_split in zip(first, second))


def test_generate_splits_rejects_splits_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="splits must sum to 1.0"):
        generate_splits(size=10, splits=(0.7, 0.2, 0.2))


def test_get_image_features_processes_single_pil_image():
    image = Image.new("RGB", (5, 7))

    img_features = get_image_features(image_to_tensor_processor, image)

    assert img_features.shape == (1, 3, 7, 5)
    assert torch.equal(img_features, torch.zeros((1, 3, 7, 5)))


def test_get_image_features_processes_pil_image_lists():
    images = [Image.new("RGB", (5, 7)), Image.new("RGB", (5, 7))]

    img_features = get_image_features(image_to_tensor_processor, images)

    assert img_features.shape == (2, 3, 7, 5)
    assert torch.equal(img_features, torch.zeros((2, 3, 7, 5)))


def test_get_image_features_rejects_empty_image_lists():
    with pytest.raises(ValueError, match="image list can not be empty"):
        get_image_features(image_to_tensor_processor, [])


def test_get_image_features_rejects_non_pil_inputs():
    with pytest.raises(TypeError, match="single PIL image or a list of PIL images"):
        get_image_features(image_to_tensor_processor, torch.zeros((3, 7, 5)))


def test_get_image_features_rejects_non_pil_items_in_image_lists():
    images = [Image.new("RGB", (5, 7)), torch.zeros((3, 7, 5))]

    with pytest.raises(TypeError, match="not a PIL image list"):
        get_image_features(image_to_tensor_processor, images)


def test_get_image_features_rejects_non_tensor_processor_outputs():
    with pytest.raises(TypeError, match="expected Tensor as element 0"):
        get_image_features(lambda image: image, Image.new("RGB", (5, 7)))


def test_get_image_features_rejects_processor_outputs_with_wrong_rank():
    with pytest.raises(ValueError, match="image features must have 4 dimensions"):
        get_image_features(lambda image: torch.zeros((7, 5)), Image.new("RGB", (5, 7)))


def test_get_image_features_rejects_processor_outputs_with_mismatched_shapes():
    images = [Image.new("RGB", (5, 7)), Image.new("RGB", (6, 7))]

    with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
        get_image_features(image_to_tensor_processor, images)
