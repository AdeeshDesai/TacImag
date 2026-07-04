"""Stage-1 training workspace for the tactile imagination model.

Single workspace for both modalities: the sample-time visualization is picked
from the policy's layout ('image' -> tactile RGB PNGs, 'field' -> force-field
shear plots). Checkpoint payload format is identical to diffusion_policy's
BaseWorkspace, so checkpoints interchange with the wider DP tooling.
"""
import copy
import os
import pathlib
import random
import threading

import dill
import hydra
import numpy as np
import torch
import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from tacimag.imagination.viz import save_tactile_image_pair, save_force_field_pair

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainImaginationWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 10
            cfg.training.checkpoint_every = 1
            cfg.training.sample_every = 1

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset), \
            f"dataset must be BaseImageDataset, got {type(dataset)}"
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs)
                // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch,
            # huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(start_method="thread"),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})

        # configure checkpoint manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save one batch for sampling visualization
        train_sampling_batch = None

        while self.epoch < cfg.training.num_epochs:
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                    if not is_last_batch:
                        # log of last step is combined with sampling
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps - 1):
                        break

            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.ema_model if cfg.training.use_ema else self.model
            policy.eval()

            # run diffusion sampling on a training batch, save GT/pred pairs
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    batch = dict_apply(
                        train_sampling_batch,
                        lambda x: x.to(device, non_blocking=True))
                    result = policy.predict_action(batch['obs'])
                    pred = result['action_pred']   # [B, 1, C, H, W]
                    gt = result['img']             # [B, 1, C, H, W]

                    sample_dir = os.path.join(self.output_dir, "samples")
                    save_pair = (save_tactile_image_pair
                                 if policy.layout == 'image'
                                 else save_force_field_pair)
                    for i in range(min(16, gt.shape[0])):
                        save_pair(gt[i, 0], pred[i, 0],
                                  epoch=self.epoch, b=i, base_dir=sample_dir)
                    del batch, result, pred, gt

            # no env rollout for imagination training: proxy score for topk
            step_log['test_mean_score'] = -train_loss

            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 \
                    and cfg.checkpoint.save_ckpt:
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = {
                    key.replace('/', '_'): value
                    for key, value in step_log.items()}
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()

            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    # ========= checkpointing (diffusion_policy BaseWorkspace format) =========
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, include_keys=None, use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath(
                'checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'cfg': self.cfg, 'state_dicts': dict(), 'pickles': dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                if key not in exclude_keys:
                    payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(
                    payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath(
            'checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(
            path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(
            payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path,
            exclude_keys=None, include_keys=None, **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """Quick full-state save for research; assumes code stays identical.
        Use save_checkpoint for long-term storage."""
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
