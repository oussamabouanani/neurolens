import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neurolens.utils.precompute_utils import precompute_rows
from neurolens.utils.zarr_utils import zarr_load_to_torch


def test_precompute_rows_writes_chunked_rows_to_zarr(tmp_path):
    store = tmp_path / "nested" / "data.zarr"
    dataset = TensorDataset(torch.arange(5, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    saved_chunks = []

    def compute_batch(batch):
        (values,) = batch
        return torch.stack([values, values + 10.0, values + 20.0], dim=1)

    precompute_rows(
        save_path=store,
        dataloader=dataloader,
        total_rows=5,
        compute_batch=compute_batch,
        batch_save_count=2,
        desc="test precompute",
        log_saved_chunk=lambda chunk_idx: saved_chunks.append(chunk_idx) or "",
    )

    assert store.exists()
    assert saved_chunks == [0, 1]
    assert torch.equal(
        zarr_load_to_torch(store, validate_shape=(5, 3)),
        torch.tensor(
            [
                [0.0, 10.0, 20.0],
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
                [3.0, 13.0, 23.0],
                [4.0, 14.0, 24.0],
            ]
        ),
    )


def test_precompute_rows_uses_output_rows_for_indices(tmp_path):
    store = tmp_path / "data.zarr"
    dataset = TensorDataset(torch.arange(3, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    def compute_batch(batch):
        (values,) = batch
        return torch.stack([values, values + 100.0], dim=1)

    precompute_rows(
        save_path=store,
        dataloader=dataloader,
        total_rows=3,
        compute_batch=compute_batch,
        batch_save_count=1,
        desc="test precompute",
    )

    assert torch.equal(
        zarr_load_to_torch(store, validate_shape=(3, 2)),
        torch.tensor(
            [
                [0.0, 100.0],
                [1.0, 101.0],
                [2.0, 102.0],
            ]
        ),
    )


@pytest.mark.parametrize("batch_save_count", [0, -1])
def test_precompute_rows_rejects_non_positive_batch_save_count(tmp_path, batch_save_count):
    dataloader = DataLoader(TensorDataset(torch.arange(1)), batch_size=1)

    with pytest.raises(ValueError, match="batch_save_count must be >= 1"):
        precompute_rows(
            save_path=tmp_path / "data.zarr",
            dataloader=dataloader,
            total_rows=1,
            compute_batch=lambda batch: torch.zeros((1, 1)),
            batch_save_count=batch_save_count,
            desc="test precompute",
        )


def test_precompute_rows_rejects_non_2d_batch_outputs(tmp_path):
    dataloader = DataLoader(TensorDataset(torch.arange(1)), batch_size=1)

    with pytest.raises(ValueError, match="batch_out must be a 2D tensor"):
        precompute_rows(
            save_path=tmp_path / "data.zarr",
            dataloader=dataloader,
            total_rows=1,
            compute_batch=lambda batch: torch.zeros(1),
            batch_save_count=1,
            desc="test precompute",
        )
