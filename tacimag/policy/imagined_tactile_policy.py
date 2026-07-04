"""Stage-2 manipulation Diffusion Policy with imagined-tactile conditioning.

A standard image-conditioned Diffusion Policy (DDPM over action chunks,
epsilon-prediction) that can additionally condition on an IMAGINED tactile
signal: a frozen stage-1 imagination model runs in the loop — at train time
and at rollout — and its output is injected into the observation dict under
`imagination_key`, where the ImaginationObsEncoder picks it up. No tactile
sensor is needed at deployment.

Modality is pure config (see tacimag/config/stage2/tactile/):
  vision            stage1_ckpt_path=null, imagination_key=null
  rgb_real/ff_real  no stage-1 either; the REAL tactile key sits in shape_meta
  rgb_imag/ff_imag  stage1_ckpt_path=<ckpt>, imagination_key=pred_<target>

The frozen stage-1 model is rebuilt directly from its own checkpoint payload
(cfg + EMA weights) — no workspace class round-trip.
"""
from typing import Dict, Optional

import dill
import hydra
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply

from tacimag.policy.obs_encoder import ImaginationObsEncoder


def load_imagination_model(ckpt_path: str):
    """Rebuild a frozen stage-1 imagination policy from its checkpoint.

    The checkpoint payload carries its own config; the policy is instantiated
    from it and loaded with the EMA weights (falling back to the raw model).
    """
    payload = torch.load(
        open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    if str(cfg.policy._target_).startswith('diffusion_policy.'):
        # checkpoint from the pre-TacImag research repos
        import tacimag.legacy_compat  # noqa: F401
    policy = hydra.utils.instantiate(cfg.policy)

    state_dicts = payload['state_dicts']
    weights = state_dicts.get('ema_model')
    if weights is None:
        weights = state_dicts['model']
    policy.load_state_dict(weights)

    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


class DiffusionUnetImaginationPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: ImaginationObsEncoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # ---- imagined tactile conditioning ----
            stage1_ckpt_path: Optional[str] = None,
            imagination_key: Optional[str] = None,
            # parameters passed to scheduler.step
            **kwargs):
        super().__init__()

        # ---- frozen stage-1 imagination model (optional) ----
        if stage1_ckpt_path is not None:
            assert imagination_key is not None, \
                "imagination_key is required when stage1_ckpt_path is set " \
                "(and must match the encoder's imagination_key)"
            print(f"[TacImag] loading frozen stage-1 imagination model: "
                  f"{stage1_ckpt_path}")
            # attach as a submodule; don't .to() here — the workspace moves
            # the whole policy to the training device.
            self.tactile_policy = load_imagination_model(stage1_ckpt_path)
            expected = 'pred_' + self.tactile_policy.target_key
            if imagination_key != expected:
                print(f"[TacImag] WARNING: imagination_key='{imagination_key}' "
                      f"!= 'pred_' + stage-1 target ('{expected}')")
        else:
            self.tactile_policy = None
        self.imagination_key = imagination_key

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
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

    # ========= imagined tactile =========
    def _imagine_tactile(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the frozen stage-1 model per obs step.

        obs_dict values are [B, To, ...]; each of the B*To steps is imagined
        independently, so the result [B*To, 1, C, H, W] lines up with the
        flattened per-step layout the obs encoder consumes.
        """
        B, To = next(iter(obs_dict.values())).shape[:2]
        obs_single = {
            k: v.reshape(B * To, 1, *v.shape[2:]) for k, v in obs_dict.items()}
        with torch.no_grad():
            return self.tactile_policy.imagine(obs_single)

    # ========= inference =========
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs):
        model = self.model
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
            # 2. predict model output
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond)
            # 3. x_t -> x_{t-1}
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """obs_dict: must include "obs" key. result: must include "action" key."""
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # imagine the tactile signal from the raw obs
        if self.tactile_policy is not None:
            nobs[self.imagination_key] = self._imagine_tactile(obs_dict)

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(
                size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=local_cond, global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {'action': action, 'action_pred': action_pred}

    # ========= training =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # imagine the tactile signal from the raw obs
        if self.tactile_policy is not None:
            nobs[self.imagination_key] = self._imagine_tactile(batch['obs'])

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # forward diffusion
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()
