"""Stage-1 imagination dataset.

Serves (vision + proprio) conditioning obs and the tactile imagination target
from a simulation `replay_buffer.zarr`. The tactile target's normalizer is
selected by `tactile_normalize`:

  'image_range' — 2x-1, assumes values in [0,1]   (tacrgb default)
  'identity'    — no-op; the policy diffuses raw values (tacff default;
                  pair with policy.target_normalize=false)
"""
import copy
import hashlib
import json
import os
import shutil
from typing import Dict

import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)

from tacimag.imagination.conversion import sim_data_to_replay_buffer


def _identity_normalizer():
    scale = np.array([1.0], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    stat = {'min': np.array([-1.0], np.float32), 'max': np.array([1.0], np.float32),
            'mean': np.array([0.0], np.float32), 'std': np.array([1.0], np.float32)}
    return SingleFieldLinearNormalizer.create_manual(scale, offset, stat)


class ImaginationDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            tactile_normalize: str = 'image_range',
        ):
        assert os.path.isdir(dataset_path), f"not a directory: {dataset_path}"
        assert tactile_normalize in ('image_range', 'identity')

        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(
                OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(
                shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(
                dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore())
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        # clean up a half-written cache without masking
                        # the original error
                        shutil.rmtree(cache_zarr_path, ignore_errors=True)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore())

        rgb_keys = list()
        tactile_keys = list()
        lowdim_keys = list()
        for key, attr in shape_meta['obs'].items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'tactile':
                tactile_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys + tactile_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.tactile_keys = tactile_keys
        self.lowdim_keys = lowdim_keys
        self.tactile_normalize = tactile_normalize
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        for key in self.tactile_keys:
            if self.tactile_normalize == 'identity':
                normalizer[key] = _identity_normalizer()
            else:
                normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of obs (the rest would
        # be discarded anyway). When n_obs_steps is None the slice is a no-op.
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys + self.tactile_keys:
            # T,H,W,C (channel-last in zarr) -> T,C,H,W float32
            obs_dict[key] = np.moveaxis(
                data[key][T_slice], -1, 1).astype(np.float32)
            del data[key]  # save ram
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps actions
        # (observations are already taken care of by T_slice)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }


def _get_replay_buffer(dataset_path, shape_meta, store):
    lowdim_keys = list()
    image_keys = list()
    for key, attr in shape_meta['obs'].items():
        type = attr.get('type', 'low_dim')
        if type in ('rgb', 'tactile'):
            image_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    import cv2
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = sim_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            lowdim_keys=lowdim_keys,
            image_keys=image_keys)
    return replay_buffer
