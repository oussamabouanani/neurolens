import logging
import os
from pathlib import Path

import numpy as np
import torch
import zarr

# zarr store/load functions for 2D arrays

ArrayLike = np.ndarray | zarr.core.array.Array

logger = logging.getLogger(__name__)


def get_zarr_data(
    store: str | Path,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:

    try:
        z = zarr.open_array(store=store, mode="r")
        return z.shape, z.chunks, str(z.dtype)
    except Exception as e:
        raise Exception(f"Error opening zarr store: {store!r}") from e


def zarr_to_torch(
    a: ArrayLike,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> torch.Tensor:

    np_arr = a[:] if isinstance(a, zarr.core.array.Array) else a

    tensor = torch.from_numpy(np_arr)

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def _verify_tuple_indices(indices: tuple[int, int], shape: tuple[int, int], dim: int):

    if len(indices) != 2:
        raise ValueError(f"Expected (start, end), got tuple of len={len(indices)}.")
    if dim not in [0, 1]:
        raise ValueError(f"Expected dim in [0, 1], got {dim}.")
    if not all(isinstance(idx, int) for idx in indices):
        raise ValueError(f"Expected integer indices, got {indices}.")

    start, end = indices

    if not (0 <= start < end <= shape[dim]):
        raise ValueError(f"Slice out of bounds: start={start}, end={end}, N={shape[dim]}.")

    return start, end


def _verify_list_indices(indices: list[int], shape: tuple[int, int], dim: int):

    if len(indices) == 0:
        raise ValueError("List of indices must have at least one element, got empty list.")
    if dim not in [0, 1]:
        raise ValueError(f"Expected dim in [0, 1], got {dim}.")
    if not all(isinstance(idx, int) for idx in indices):
        raise ValueError(f"Expected integer indices, got {indices}.")

    idx_min = min(indices)
    idx_max = max(indices)

    if idx_min < 0 or idx_max >= shape[dim]:
        raise ValueError(f"Index out of bounds: min={idx_min}, max={idx_max}, N={shape[dim]}.")


def get_zarr_shape(store: str | Path) -> tuple[int, int]:
    return zarr.open_array(store=store, mode="r").shape


def zarr_load_to_torch(
    store: str | Path,
    validate_shape: tuple[int, int],
    rows: int | list[int] | tuple[int, int] | None = None,
    columns: int | list[int] | tuple[int, int] | None = None,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> torch.Tensor:

    z = zarr.open_array(store=store, mode="r")

    if len(validate_shape) != 2:
        raise ValueError(f"Expected 2D array, got {len(validate_shape)} dimensions.")

    if z.shape != validate_shape:
        raise ValueError(f"Expected shape of zarr store to be {validate_shape}, got {z.shape}.")

    # column input validation: convert any iterable to a list
    if columns is None:
        columns = list(range(z.shape[1]))
    elif isinstance(columns, int):
        columns = [columns]
    elif isinstance(columns, tuple):
        start, end = _verify_tuple_indices(columns, z.shape, dim=1)
        columns = list(range(start, end))

    if not isinstance(columns, list):
        raise ValueError(f"`columns` must be an int, list[int], tuple[int, int], or None, got {type(columns)}.")

    _verify_list_indices(columns, z.shape, dim=1)

    # row validation and selection
    if rows is None:
        target_z = z.oindex[:, columns]
    else:
        if isinstance(rows, int):
            rows = [rows]

        if isinstance(rows, tuple):
            start, end = _verify_tuple_indices(rows, z.shape, dim=0)
            target_z = z.oindex[start:end, columns]

        elif isinstance(rows, list):
            _verify_list_indices(rows, z.shape, dim=0)
            target_z = z.oindex[rows, columns]

        else:
            raise ValueError(f"`indices` must be an int, list[int], tuple[int, int], or None, got {type(rows)!r}.")

    return zarr_to_torch(target_z, dtype=torch_dtype, device=device)


def zarr_save(
    store: str | Path,
    indices: int | list[int] | tuple[int, int] | None = None,
    values: torch.Tensor | None = None,
    shape: tuple[int, int] | None = None,
    dtype: str | torch.dtype | None = None,
    chunks: tuple[int, ...] | None = None,
) -> int:

    if shape is not None and len(shape) != 2:
        raise ValueError(f"Expected 2D array, got {len(shape)} dimensions.")

    if not os.path.exists(store):
        if shape is None:
            raise ValueError("Shape must be provided if store does not exist.")
        if dtype is None:
            raise ValueError("Dtype must be provided if store does not exist.")
        if chunks is None:
            raise ValueError("Chunks must be provided if store does not exist.")

        zarr.create_array(
            store=store,
            shape=shape,
            dtype=(torch.empty((), dtype=dtype).numpy().dtype if isinstance(dtype, torch.dtype) else dtype),
            chunks=chunks,
        )

        logger.info(f"Created zarr store: {store!r}")

    indices_none = indices is None
    values_none = values is None

    if indices_none and values_none:
        logger.warning(f"No indices or values provided, not saving to zarr store: {store!r}")
        return 0

    if indices_none or values_none:
        raise ValueError(
            f"Both indices (None? {indices_none}) and values (None? {values_none}) must be provided to write."
        )

    z = zarr.open_array(store=store, mode="r+")

    if len(z.shape) != 2:
        raise ValueError(f"Expected 2D array, got existing zarr store of {len(z.shape)} dimensions.")

    if shape is not None and z.shape != shape:
        raise ValueError(f"Existing zarr store shape {z.shape} does not match expected shape {shape}.")

    values_np = values.detach().contiguous().cpu().numpy()

    if len(values_np.shape) != 2:
        raise ValueError(f"Expected 2D array, got values to write into zarr of {len(values_np.shape)} dimensions.")
    if values_np.shape[1] != z.shape[1]:
        raise ValueError(f"Expected values with {z.shape[1]} columns, got {values_np.shape[1]}.")

    if isinstance(indices, int):
        indices = [indices]

    if isinstance(indices, tuple):
        start, end = _verify_tuple_indices(indices, z.shape, dim=0)
        if values_np.shape[0] != end - start:
            raise ValueError(f"Expected {end - start} rows of values, got {values_np.shape[0]}.")
        z[start:end] = values_np
        return end - start

    elif isinstance(indices, list):
        _verify_list_indices(indices, z.shape, dim=0)
        if values_np.shape[0] != len(indices):
            raise ValueError(f"Expected {len(indices)} rows of values, got {values_np.shape[0]}.")
        z[indices] = values_np
        return len(indices)

    else:
        raise ValueError(f"`indices` must be an int, list[int], or tuple[int, int], got {type(indices)!r}.")
