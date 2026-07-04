"""TacImag stage 1 — train the tactile imagination model.

  python train_stage1.py modality=tacff  task=usb
  python train_stage1.py modality=tacrgb task=usb training.seed=43
"""
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent / 'tacimag' / 'config' / 'stage1'),
    config_name='config')
def main(cfg: OmegaConf):
    # resolve immediately so all interpolations are baked into checkpoints
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
