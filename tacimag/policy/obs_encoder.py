"""Multi-image obs encoder with an optional *imagined* tactile stream.

Identical to diffusion_policy's MultiImageObsEncoder for the keys declared in
shape_meta, plus one extra ResNet branch for the imagined tactile obs
(`imagination_key`), which is injected into the obs dict at runtime by the
stage-2 policy and therefore does not appear in shape_meta.

Set imagination_key=null to get a plain vision(+state) encoder — the branch
is only built (and only consumes parameters) when a key is configured, and
only runs when that key is present in the obs dict.

This replaces the original multi_image_obs_encoder_imagination.py, where the
key and shape were hardcoded to the tactile-RGB modality; here both come from
the config, so switching to the force field is a config change, not a code
edit.
"""
from typing import Dict, Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
import torchvision

from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules


class ImaginationObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str, nn.Module]],
            imagination_key: Optional[str] = None,
            imagination_shape: Optional[Tuple[int, int, int]] = None,
            resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
            crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
            random_crop: bool = True,
            use_group_norm: bool = False,
            share_rgb_model: bool = False,
            imagenet_norm: bool = False
        ):
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        self.imagination_key = imagination_key

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        # 1. standard keys from shape_meta
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                this_model = self._build_resnet(
                    key, rgb_model, share_rgb_model, use_group_norm)
                if this_model is not None:
                    key_model_map[key] = this_model
                key_transform_map[key] = self._build_transform(
                    key, shape, resize_shape, crop_shape, random_crop, imagenet_norm)
            elif type == 'low_dim':
                low_dim_keys.append(key)

        # 2. the imagined tactile stream (runtime-injected, not in shape_meta)
        if self.imagination_key is not None:
            assert imagination_shape is not None, \
                "imagination_shape is required when imagination_key is set"
            shape = tuple(imagination_shape)
            key_shape_map[self.imagination_key] = shape
            this_model = self._build_resnet(
                self.imagination_key, rgb_model, share_rgb_model, use_group_norm)
            if this_model is not None:
                key_model_map[self.imagination_key] = this_model
            key_transform_map[self.imagination_key] = self._build_transform(
                self.imagination_key, shape, resize_shape, crop_shape,
                random_crop, imagenet_norm)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map

    def _build_resnet(self, key, rgb_model, share, use_group_norm):
        if share:
            return None
        if isinstance(rgb_model, dict):
            this_model = rgb_model.get(
                key, copy.deepcopy(next(iter(rgb_model.values()))))
        else:
            this_model = copy.deepcopy(rgb_model)

        if use_group_norm:
            this_model = replace_submodules(
                root_module=this_model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features)
            )
        return this_model

    def _build_transform(self, key, shape, resize_shape, crop_shape,
                         random_crop, imagenet_norm):
        this_resizer = nn.Identity()
        if resize_shape is not None:
            h, w = resize_shape[key] if isinstance(resize_shape, dict) else resize_shape
            this_resizer = torchvision.transforms.Resize(size=(h, w))
            shape = (shape[0], h, w)

        this_randomizer = nn.Identity()
        if crop_shape is not None:
            h, w = crop_shape[key] if isinstance(crop_shape, dict) else crop_shape
            if random_crop:
                this_randomizer = CropRandomizer(
                    input_shape=shape, crop_height=h, crop_width=w,
                    num_crops=1, pos_enc=False)
            else:
                this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))

        this_normalizer = nn.Identity()
        if imagenet_norm:
            this_normalizer = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return nn.Sequential(this_resizer, this_randomizer, this_normalizer)

    def forward(self, obs_dict):
        features = list()

        if self.share_rgb_model:
            imgs = [self.key_transform_map[k](obs_dict[k]) for k in self.rgb_keys]
            if self.imagination_key is not None and self.imagination_key in obs_dict:
                imgs.append(self.key_transform_map[self.imagination_key](
                    obs_dict[self.imagination_key]))
            imgs_cat = torch.cat(imgs, dim=0)
            feature = self.key_model_map['rgb'](imgs_cat)
            feature = feature.reshape(
                -1, imgs_cat.shape[0] // len(imgs), *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1).reshape(feature.shape[1], -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = self.key_transform_map[key](obs_dict[key])
                features.append(self.key_model_map[key](img))
            if self.imagination_key is not None and self.imagination_key in obs_dict:
                img = self.key_transform_map[self.imagination_key](
                    obs_dict[self.imagination_key])
                features.append(self.key_model_map[self.imagination_key](img))

        for key in self.low_dim_keys:
            features.append(obs_dict[key])

        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = {
            k: torch.zeros((1,) + v, dtype=self.dtype, device=self.device)
            for k, v in self.key_shape_map.items()}
        return self.forward(example_obs_dict).shape[1:]
