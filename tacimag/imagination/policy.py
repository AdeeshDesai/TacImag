"""Stage-1 tactile imagination policy.

One diffusion model, two modalities, selected by config (see
tacimag/config/stage1/modality/):

  tacrgb — imagine the tactile RGB (GelSight/Taxim) image [3, 320, 240].
           layout='image': the target is bilinearly downsampled by
           `downsample_factor`, flattened row-major to a [H/d, C*W/d]
           trajectory, and the diffusion runs in *normalized* image space.
  tacff  — imagine the contact force field [3, 10, 14] (ch0 normal, ch1/2
           shear). layout='field': the target is flattened column-major to a
           [W, C*H] trajectory and padded along the sequence axis to a
           UNet-friendly length. Runs in raw force units
           (`target_normalize=false`).

The class supersedes the original `imagination_policy.DiffusionUnetImagePolicy`
(tactile RGB) and `diffusion_unet_tacff_policy.DiffusionUnetTacFFPolicy`
(force field): all shape constants that were hardcoded there are derived here
from `shape_meta`. Module attribute names (`imagine_model`,
`imagine_obs_encoder`, ...) are kept identical, so state dicts of checkpoints
trained with the original classes load unchanged.
"""
from typing import Dict, Optional
import math

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


def _find_tactile_key(shape_meta: dict) -> str:
    """The imagination target is the single obs entry with type 'tactile'."""
    keys = [k for k, attr in shape_meta['obs'].items()
            if attr.get('type', 'low_dim') == 'tactile']
    if len(keys) != 1:
        raise ValueError(
            f"shape_meta must contain exactly one obs of type 'tactile' "
            f"(the imagination target), found: {keys}")
    return keys[0]


class DiffusionImaginationPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            imagine_obs_encoder: MultiImageObsEncoder,
            horizon=1,
            n_action_steps=1,
            n_obs_steps=1,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # ---- imagination target (all derivable from shape_meta) ----
            target_key: Optional[str] = None,
            layout: str = 'auto',            # 'image' | 'field' | 'auto'
            target_normalize: bool = True,   # diffuse in normalized vs raw space
            downsample_factor: int = 4,      # layout='image' only
            # parameters passed to scheduler.step
            **kwargs):
        super().__init__()

        # ---- resolve the target and its trajectory geometry ----
        if target_key is None:
            target_key = _find_tactile_key(shape_meta)
        C, H, W = tuple(shape_meta['obs'][target_key]['shape'])
        if layout == 'auto':
            # high-res tactile images diffuse downsampled row-major;
            # small force-field grids diffuse column-major, padded.
            layout = 'image' if min(H, W) >= 32 else 'field'

        if layout == 'image':
            d = downsample_factor
            assert H % d == 0 and W % d == 0, \
                f"target {H}x{W} not divisible by downsample_factor={d}"
            seq_len = H // d          # e.g. 320/4 = 80
            feat_dim = C * (W // d)   # e.g. 3*60  = 180
            pad = 0
        elif layout == 'field':
            seq_len = int(math.ceil(W / 8) * 8)  # e.g. 14 -> 16
            feat_dim = C * H                      # e.g. 3*10 = 30
            pad = seq_len - W
        else:
            raise ValueError(f"Unknown layout: {layout}")

        self.target_key = target_key
        self.target_shape = (C, H, W)
        self.layout = layout
        self.target_normalize = target_normalize
        self.downsample_factor = downsample_factor
        self._pad = pad

        # trajectory dims (named action_dim/horizon for parity with DP code)
        action_dim = feat_dim
        horizon = seq_len

        # get feature dim
        obs_feature_dim = imagine_obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        imagine_model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.imagine_obs_encoder = imagine_obs_encoder
        self.imagine_model = imagine_model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= target <-> trajectory =========
    def _target_to_traj(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] target -> [B, seq, feat] diffusion trajectory."""
        if self.layout == 'image':
            x = F.interpolate(
                x, scale_factor=1.0 / self.downsample_factor,
                mode='bilinear', align_corners=False)
            return rearrange(x, 'b c h w -> b h (c w)')
        x = rearrange(x, 'b c h w -> b w (c h)')
        if self._pad:
            x = F.pad(x, (0, 0, 0, self._pad))
        return x

    def _traj_to_target(self, traj: torch.Tensor) -> torch.Tensor:
        """[B, seq, feat] trajectory -> [B, C, H, W] target."""
        C, H, W = self.target_shape
        if self.layout == 'image':
            x = rearrange(traj, 'b h (c w) -> b c h w', c=C)
            return F.interpolate(
                x, scale_factor=float(self.downsample_factor),
                mode='bilinear', align_corners=False)
        if self._pad:
            traj = traj[:, :W]
        return rearrange(traj, 'b w (c h) -> b c h w', c=C)

    # ========= inference =========
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs):
        imagine_model = self.imagine_model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # 2. predict clean target / noise (per scheduler prediction_type)
            model_output = imagine_model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond)
            # 3. x_t -> x_{t-1}
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _sample_target(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the diffusion and return the imagined target [B, C, H, W]
        in the space the diffusion was trained in (normalized or raw)."""
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps

        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.imagine_obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)

        cond_data = torch.zeros(
            size=(B, self.horizon, self.action_dim),
            device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=None, global_cond=global_cond,
            **self.kwargs)
        return self._traj_to_target(nsample)

    def imagine(self, obs_dict: Dict[str, torch.Tensor],
                unnormalize: bool = False) -> torch.Tensor:
        """Imagine the tactile target from vision/proprio obs.

        Returns [B, 1, C, H, W]. By default the output stays in the space
        the diffusion runs in (that is what the stage-2 policy conditions
        on); pass unnormalize=True for raw sensor units.
        """
        pred = self._sample_target(obs_dict)
        if unnormalize and self.target_normalize:
            pred = self.normalizer[self.target_key].unnormalize(pred)
        return pred.unsqueeze(1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Evaluation entrypoint: returns the imagined target in raw sensor
        units plus the ground truth (when present in obs) for comparison."""
        assert 'past_action' not in obs_dict  # not implemented yet
        pred = self._sample_target(obs_dict)
        if self.target_normalize:
            pred = self.normalizer[self.target_key].unnormalize(pred)
        result = {'action_pred': pred.unsqueeze(1)}
        if self.target_key in obs_dict:
            result['img'] = obs_dict[self.target_key]
        return result

    # ========= training =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        batch_size = next(iter(nobs.values())).shape[0]

        # diffusion target: normalized or raw sensor values
        target_src = nobs if self.target_normalize else batch['obs']
        target = target_src[self.target_key].squeeze(1)   # [B, C, H, W]
        trajectory = self._target_to_traj(target)
        cond_data = trajectory

        # condition on the (normalized) non-tactile obs
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.imagine_obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # forward diffusion
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.imagine_model(
            noisy_trajectory, timesteps, local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target_out = noise
        elif pred_type == 'sample':
            target_out = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target_out, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()
