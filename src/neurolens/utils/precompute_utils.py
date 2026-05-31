import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .zarr_utils import zarr_save

logger = logging.getLogger(__name__)


@torch.inference_mode()
def precompute_rows(
    *,
    save_path: Path,
    dataloader: DataLoader,
    total_rows: int,
    compute_batch: Callable[[Any], torch.Tensor],
    batch_save_count: int,
    desc: str,
    log_saved_chunk: Callable[[int], str] | None = None,
) -> None:

    if batch_save_count < 1:
        raise ValueError("batch_save_count must be >= 1")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_chunk(chunk_idx: int):

        elem_concat = torch.cat([elem.cpu() for elem in elem_list], dim=0)

        zarr_save(
            store=save_path,
            indices=(global_start_idx, global_end_idx),
            values=elem_concat,
            shape=(total_rows, elem_concat.shape[1]),
            dtype="float32",
            chunks=(elem_concat.shape[0], elem_concat.shape[1]),
        )

        if log_saved_chunk is not None:
            logger.info(log_saved_chunk(chunk_idx))

    pbar = tqdm(dataloader, desc=desc)

    global_start_idx, global_end_idx = -1, -1
    elem_list = []

    chunk_idx = 0
    batch_save_counter = 0
    next_start_idx = 0

    for batch in pbar:
        batch_out = compute_batch(batch)

        if batch_out.ndim != 2:
            raise ValueError("batch_out must be a 2D tensor")

        batch_len = batch_out.shape[0]

        if global_start_idx == -1:
            global_start_idx = next_start_idx
            global_end_idx = global_start_idx

        global_end_idx += batch_len
        elem_list.append(batch_out)

        batch_save_counter += 1
        if batch_save_counter % batch_save_count == 0:
            batch_save_counter = 0

            _save_chunk(chunk_idx)

            next_start_idx = global_end_idx
            global_start_idx, global_end_idx = -1, -1
            elem_list = []
            chunk_idx += 1

    if len(elem_list) > 0:
        _save_chunk(chunk_idx)
