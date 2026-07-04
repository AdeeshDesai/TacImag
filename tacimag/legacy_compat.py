"""Compatibility aliases for checkpoints trained before TacImag existed
(in the original tacdiff / tacdiff_FF / TVB research repos).

Those checkpoints embed their Hydra config, whose `_target_` strings point at
module paths that only existed in the lab's private forks. Importing this
module registers equivalent classes under those paths, so the checkpoints
load through the unified TacImag implementation unchanged (module attribute
names were kept identical, so state dicts match one-to-one).

Fresh TacImag training runs never need this module — their checkpoints
reference `tacimag.*` targets directly. `load_imagination_model` imports it
automatically when it sees a legacy target.
"""
import sys
import types

from omegaconf import OmegaConf

from tacimag.imagination.policy import DiffusionImaginationPolicy
from tacimag.imagination.workspace import TrainImaginationWorkspace
from tacimag.policy.obs_encoder import ImaginationObsEncoder
from tacimag.policy.imagined_tactile_policy import DiffusionUnetImaginationPolicy


# ---------------------------------------------------------------------------
# Stage-1 policies: the three original forks map to the unified policy with
# the modality flags each fork hardcoded.
# ---------------------------------------------------------------------------
class _LegacyTacRGBPolicy(DiffusionImaginationPolicy):
    """tacdiff imagination_policy.DiffusionUnetImagePolicy (tactile RGB)."""
    def __init__(self, **kwargs):
        kwargs.setdefault('layout', 'image')
        kwargs.setdefault('target_normalize', True)
        super().__init__(**kwargs)


class _LegacyTacFFPolicy(DiffusionImaginationPolicy):
    """tacdiff_FF diffusion_unet_tacff_policy.DiffusionUnetTacFFPolicy
    (force field, diffusion in raw force units)."""
    def __init__(self, **kwargs):
        kwargs.setdefault('layout', 'field')
        kwargs.setdefault('target_normalize', False)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Stage-2 policy: original hardcoded the tactile-RGB pred key and used the
# `tactile_ckpt_path` kwarg name.
# ---------------------------------------------------------------------------
class _LegacyStage2Policy(DiffusionUnetImaginationPolicy):
    def __init__(self, tactile_ckpt_path=None, **kwargs):
        kwargs.setdefault('stage1_ckpt_path', tactile_ckpt_path)
        if kwargs.get('stage1_ckpt_path') is not None:
            kwargs.setdefault(
                'imagination_key', 'pred_right_tactile_camera_taxim')
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Stage-2 obs encoder: original hardcoded key + shape as defaults.
# ---------------------------------------------------------------------------
class _LegacyImaginationEncoder(ImaginationObsEncoder):
    def __init__(self, **kwargs):
        kwargs.setdefault('imagination_key', 'pred_right_tactile_camera_taxim')
        kwargs.setdefault('imagination_shape', (3, 320, 240))
        super().__init__(**kwargs)


def _register(module_path: str, **attrs):
    """Register a synthetic module under a legacy import path.

    Only used for paths that do NOT exist in the installed diffusion_policy
    package, so nothing real is shadowed.
    """
    if module_path in sys.modules:
        return
    mod = types.ModuleType(module_path)
    for name, value in attrs.items():
        setattr(mod, name, value)
    sys.modules[module_path] = mod


_register('diffusion_policy.policy.imagination_policy',
          DiffusionUnetImagePolicy=_LegacyTacRGBPolicy)
_register('diffusion_policy.policy.diffusion_unet_tacff_policy',
          DiffusionUnetTacFFPolicy=_LegacyTacFFPolicy)
_register('diffusion_policy.policy.diffusion_unet_image_tactile_policy',
          DiffusionUnetImagePolicy=_LegacyStage2Policy)
_register('diffusion_policy.model.vision.multi_image_obs_encoder_imagination',
          MultiImageObsEncoder=_LegacyImaginationEncoder)
_register('diffusion_policy.workspace.imagination_workspace',
          TrainDiffusionUnetImageWorkspace=TrainImaginationWorkspace)
_register('diffusion_policy.workspace.train_diffusion_unet_tacff_workspace',
          TrainDiffusionUnetImageWorkspace=TrainImaginationWorkspace)


# ---------------------------------------------------------------------------
# Upstream MultiImageObsEncoder rejects obs type 'tactile' (the lab forks
# patched it to skip). Legacy stage-1 configs pass the full shape_meta —
# including the tactile target — to the conditioning encoder, so make the
# installed encoder tolerate it by filtering those entries out. Identical
# module tree afterwards -> state dicts still match.
# ---------------------------------------------------------------------------
import diffusion_policy.model.vision.multi_image_obs_encoder as _mioe  # noqa: E402

if not getattr(_mioe, '_tacimag_tactile_tolerant', False):
    _OrigEncoder = _mioe.MultiImageObsEncoder

    class _TactileTolerantEncoder(_OrigEncoder):
        def __init__(self, shape_meta, **kwargs):
            meta = OmegaConf.to_container(shape_meta, resolve=True) \
                if OmegaConf.is_config(shape_meta) else dict(shape_meta)
            obs = {k: v for k, v in meta['obs'].items()
                   if (v.get('type', 'low_dim') != 'tactile')}
            super().__init__(
                shape_meta={**meta, 'obs': obs}, **kwargs)

    _mioe.MultiImageObsEncoder = _TactileTolerantEncoder
    _mioe._tacimag_tactile_tolerant = True
