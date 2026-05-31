import numpy as np
import pytest
import torch
import zarr

from neurolens.utils.zarr_utils import get_zarr_data, zarr_load_to_torch, zarr_save


def create_zarr_array(store, data, chunks=(2, 2)):
    z = zarr.create_array(
        store=store,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
    )
    z[:] = data
    return z


def load_zarr_array(store, data, **kwargs):
    return zarr_load_to_torch(store, validate_shape=data.shape, **kwargs)


def test_get_zarr_data_returns_shape_chunks_and_dtype(tmp_path):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32), chunks=(2, 3))

    shape, chunks, dtype = get_zarr_data(store)

    assert shape == (4, 3)
    assert chunks == (2, 3)
    assert dtype == "float32"


def test_zarr_load_to_torch_loads_full_2d_array(tmp_path):
    store = tmp_path / "data.zarr"
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    create_zarr_array(store, data)

    tensor = load_zarr_array(store, data)

    assert torch.equal(tensor, torch.from_numpy(data))


def test_zarr_load_to_torch_selects_int_list_and_tuple_indices(tmp_path):
    store = tmp_path / "data.zarr"
    data = np.arange(20, dtype=np.float32).reshape(5, 4)
    create_zarr_array(store, data)

    assert torch.equal(
        load_zarr_array(store, data, rows=1),
        torch.from_numpy(data[[1]]),
    )
    assert torch.equal(
        load_zarr_array(store, data, rows=[0, 2], columns=[1, 3]),
        torch.from_numpy(data[[0, 2]][:, [1, 3]]),
    )
    assert torch.equal(
        load_zarr_array(store, data, rows=(1, 4), columns=(0, 2)),
        torch.from_numpy(data[1:4][:, 0:2]),
    )


def test_zarr_load_to_torch_converts_dtype(tmp_path):
    store = tmp_path / "data.zarr"
    data = np.arange(6, dtype=np.int64).reshape(2, 3)
    create_zarr_array(store, data)

    tensor = load_zarr_array(store, data, torch_dtype=torch.float32)

    assert tensor.dtype == torch.float32


def test_zarr_load_to_torch_rejects_non_2d_arrays(tmp_path):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((2, 3, 4), dtype=np.float32), chunks=(1, 3, 4))

    with pytest.raises(ValueError, match="Expected shape"):
        zarr_load_to_torch(store, validate_shape=(2, 3))


@pytest.mark.parametrize("validate_shape", [(4,), (4, 3, 2)])
def test_zarr_load_to_torch_rejects_non_2d_validate_shape(tmp_path, validate_shape):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected 2D array"):
        zarr_load_to_torch(store, validate_shape=validate_shape)


def test_zarr_load_to_torch_rejects_mismatched_validate_shape(tmp_path):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected shape"):
        zarr_load_to_torch(store, validate_shape=(5, 3))


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (1.5, None),
        (None, 1.5),
        ("bad", None),
        (None, "bad"),
    ],
)
def test_zarr_load_to_torch_rejects_unsupported_index_types(tmp_path, rows, columns):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError):
        zarr_load_to_torch(store, validate_shape=(4, 3), rows=rows, columns=columns)


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        ([1.5], None),
        ((0.5, 2), None),
        (None, [1.5]),
        (None, (0.5, 2)),
    ],
)
def test_zarr_load_to_torch_rejects_non_integer_indices(tmp_path, rows, columns):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected integer indices"):
        zarr_load_to_torch(store, validate_shape=(4, 3), rows=rows, columns=columns)


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        ([-1], None),
        ([4], None),
        (None, [-1]),
        (None, [3]),
        ((0, 5), None),
        (None, (0, 4)),
    ],
)
def test_zarr_load_to_torch_rejects_out_of_bounds_indices(tmp_path, rows, columns):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError):
        zarr_load_to_torch(store, validate_shape=(4, 3), rows=rows, columns=columns)


def test_zarr_save_creates_store_without_writing_values(tmp_path):
    store = tmp_path / "created.zarr"

    written = zarr_save(
        store,
        shape=(4, 3),
        dtype=torch.float32,
        chunks=(2, 3),
    )

    assert written == 0
    assert get_zarr_data(store) == ((4, 3), (2, 3), "float32")


def test_zarr_save_writes_tuple_list_and_int_indices(tmp_path):
    store = tmp_path / "data.zarr"

    zarr_save(store, shape=(5, 3), dtype="float32", chunks=(2, 3))
    assert (
        zarr_save(
            store,
            indices=(1, 3),
            values=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        == 2
    )
    assert (
        zarr_save(
            store,
            indices=[0, 4],
            values=torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        )
        == 2
    )
    assert zarr_save(store, indices=3, values=torch.tensor([[13.0, 14.0, 15.0]])) == 1

    expected = torch.tensor(
        [
            [7.0, 8.0, 9.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [13.0, 14.0, 15.0],
            [10.0, 11.0, 12.0],
        ]
    )
    assert torch.equal(zarr_load_to_torch(store, validate_shape=(5, 3)), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": None, "dtype": "float32", "chunks": (2, 3)},
        {"shape": (4, 3), "dtype": None, "chunks": (2, 3)},
        {"shape": (4, 3), "dtype": "float32", "chunks": None},
    ],
)
def test_zarr_save_requires_create_metadata_for_missing_stores(tmp_path, kwargs):
    with pytest.raises(ValueError):
        zarr_save(tmp_path / "missing.zarr", **kwargs)


def test_zarr_save_rejects_non_2d_create_shape(tmp_path):
    with pytest.raises(ValueError, match="Expected 2D array"):
        zarr_save(
            tmp_path / "data.zarr",
            shape=(2, 3, 4),
            dtype="float32",
            chunks=(1, 3, 4),
        )


def test_zarr_save_rejects_non_2d_existing_store(tmp_path):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((2, 3, 4), dtype=np.float32), chunks=(1, 3, 4))

    with pytest.raises(ValueError, match="existing zarr store"):
        zarr_save(store, indices=0, values=torch.zeros((1, 3)))


def test_zarr_save_rejects_existing_store_shape_mismatch(tmp_path):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="does not match expected shape"):
        zarr_save(
            store,
            indices=(0, 2),
            values=torch.zeros((2, 3)),
            shape=(5, 3),
        )


def test_zarr_save_rejects_non_2d_values(tmp_path):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="values to write"):
        zarr_save(store, indices=0, values=torch.zeros(3))


@pytest.mark.parametrize(
    ("indices", "values"),
    [
        (None, torch.zeros((1, 3))),
        ([0], None),
    ],
)
def test_zarr_save_requires_indices_and_values_together(tmp_path, indices, values):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Both indices"):
        zarr_save(store, indices=indices, values=values)


@pytest.mark.parametrize("indices", [[1.5], (0.5, 2)])
def test_zarr_save_rejects_non_integer_indices(tmp_path, indices):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected integer indices"):
        zarr_save(store, indices=indices, values=torch.zeros((1, 3)))


@pytest.mark.parametrize(
    ("indices", "values"),
    [
        ((0, 2), torch.zeros((1, 3))),
        ([0, 1], torch.zeros((1, 3))),
    ],
)
def test_zarr_save_rejects_values_with_wrong_row_count(tmp_path, indices, values):
    store = tmp_path / "data.zarr"
    create_zarr_array(store, np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="rows of values"):
        zarr_save(store, indices=indices, values=values)
