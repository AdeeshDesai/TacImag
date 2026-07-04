"""TacImag stage 2 — train the manipulation Diffusion Policy.

  python train_stage2.py tactile=ff_imag task=usb            # ours (imagined FF)
  python train_stage2.py tactile=vision  task=usb            # baseline
  python train_stage2.py tactile=ff_imag task=usb stage1_ckpt=/path/to/latest.ckpt
"""
# IsaacGym must be imported BEFORE torch (same as ManiFeel's train.py) —
# otherwise the env runner fails with "PyTorch was imported before isaacgym".
# Its native library links against the conda env's libpython, which the
# system linker can't find on its own; preloading it here removes the need
# for users to export LD_LIBRARY_PATH=$CONDA_PREFIX/lib.
import ctypes
import os as _os
import sys as _sys

_libpython = _os.path.join(
    _sys.prefix, 'lib',
    f'libpython{_sys.version_info.major}.{_sys.version_info.minor}.so.1.0')
if _os.path.exists(_libpython):
    ctypes.CDLL(_libpython, mode=ctypes.RTLD_GLOBAL)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass  # simulator not installed; fails later only if a rollout is requested

import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ManiFeel's config dir (IsaacGym task configs, composed by its env runner at
# rollout time) is exposed via hydra_plugins/tacimag_searchpath_plugin.py.


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent / 'tacimag' / 'config' / 'stage2'),
    config_name='config')
def main(cfg: OmegaConf):
    # resolve immediately so all interpolations are baked into checkpoints
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
