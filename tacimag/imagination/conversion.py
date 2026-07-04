"""Zarr-only replay-buffer loading for simulation datasets.

ManiFeel/TacSL episodes are recorded straight into `replay_buffer.zarr`
(no video encoding), so this is a simplified, video-free version of
diffusion_policy's `real_data_to_replay_buffer`.
"""
import os

import zarr
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer


def sim_data_to_replay_buffer(
        dataset_path: str,
        out_store=None,
        lowdim_keys=(),
        image_keys=()):
    """Copy the episode zarr under `dataset_path` into `out_store`.

    Accepts both layouts: `<dataset_path>/replay_buffer.zarr` or
    `dataset_path` itself as the zarr root (data/ + meta/) — the latter is
    how the ManiFeel HuggingFace zips extract.

    lowdim_keys + 'action' are copied wholesale; each image-like key
    (tactile RGB, force field, cameras — any [N, H, W, C] array) is copied
    frame-chunked for efficient random access during training.
    """
    in_rb_path = os.path.join(dataset_path, "replay_buffer.zarr")
    if not os.path.isdir(in_rb_path):
        in_rb_path = dataset_path
    in_rb = ReplayBuffer.create_from_path(in_rb_path, mode='r')
    n_steps = in_rb.n_steps

    lowdim_keys = list(lowdim_keys)
    chunks_map = {}
    compressor_map = {}
    for key in lowdim_keys + ["action"]:
        chunks_map[key] = in_rb[key].shape
        compressor_map[key] = None

    out_rb = ReplayBuffer.copy_from_store(
        src_store=in_rb.root.store,
        store=out_store,
        keys=lowdim_keys + ["action"],
        chunks=chunks_map,
        compressors=compressor_map
    )

    for key in image_keys:
        arr = zarr.open(os.path.join(in_rb_path, "data", key), "r")
        n, h, w, c = arr.shape
        assert n == n_steps, f"{key}: {n} frames != {n_steps} steps"

        out_rb.data.require_dataset(
            name=key,
            shape=(n_steps, h, w, c),
            chunks=(1, h, w, c),
            dtype=arr.dtype,
            compressor=None
        )
        out_arr = out_rb[key]
        for i in tqdm(range(n_steps), desc=f"Copy {key}"):
            out_arr[i] = arr[i]

    return out_rb
