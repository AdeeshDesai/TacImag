<div align="center">

<h2>Imagining the Sense of Touch:<br>
Touch-Informed Manipulation via Imagined Tactile Representations</h2>

<a href="https://tacimag.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2607.01684"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
<a href="https://huggingface.co/datasets/Adeesh09/TacImag/tree/main"><img src="https://img.shields.io/badge/Checkpoints-HuggingFace-yellow" alt="Checkpoints"></a>

</div>

<p align="center">
<a href="https://zhangzhiyuanzhang.github.io/personal_website/">Zhiyuan Zhang</a>*,
<a href="https://scholar.google.com/citations?user=A880yg0AAAAJ&hl=en&oi=ao">Adeesh Desai</a>*,
<a href="https://vict0rhu.github.io/">Jyun-Chi Hu</a>,
<a href="https://www.linkedin.com/in/yosuke-saka-446410283/?locale=en">Yosuke Saka</a>,
<a href="https://quan-luu.github.io/">Quan Khanh Luu</a>,
<br>
<a href="https://scholar.google.com/citations?user=X52xke0AAAAJ&hl=en">Jiuzhou Lei</a>,
<a href="https://scholar.google.com/citations?user=Rmadq64AAAAJ&hl=en">Davood Soleymanzadeh</a>,
<a href="https://www.linkedin.com/in/bihao-zhang-861754352/">Bihao Zhang</a>,
<a href="https://zh.engr.tamu.edu/">Minghui Zheng</a>,
<a href="https://www.purduemars.com/">Yu She<sup>&dagger;</sup></a>
<br>
* Equal Contribution &nbsp; &dagger; Corresponding Author
</p>

<p style="width: 80%; margin: 0 auto; text-align: justify;">
<strong>TacImag</strong> enables a manipulation policy to <em>imagine</em> its sense of touch.
During training, paired visual-tactile demonstrations supervise a vision-to-touch diffusion model.
At deployment, the frozen model predicts task-relevant tactile representations
(either a tactile force field or a tactile RGB image) from vision, and a diffusion
policy conditions on the imagined touch.
<strong>No tactile sensor is required at deployment.</strong>
</p>

<p align="center">
<img src="media/sim_denoising.gif" alt="TacImag: imagining tactile signals via diffusion denoising" width="90%" />
</p>

## Highlights

- **Tactile imagination**: touch is predicted, not sensed. Stage 1 imagines the
  tactile signal from vision; stage 2 acts on it.
- **One switch per stage** covers every experiment:
  - Stage 1: `modality = tacff | tacrgb`
  - Stage 2: `tactile = vision | rgb_real | rgb_imag | ff_real | ff_imag`
- Ready-to-run script for every task and modality, plus one-command sweeps.

## Installation

```bash
git clone https://github.com/AdeeshDesai/TacImag.git
cd TacImag
bash install_tacimag.sh
conda activate tacimag
```

The installation script sets up the third-party dependencies (simulator and
policy-learning stack under `thirdparty/`), creates the `tacimag` conda
environment, and installs the package in editable mode. Stage-1 training needs
no simulator; use `SKIP_ISAACGYM_DOWNLOAD=true` if you only want imagination
training.

## Dataset and Checkpoint Download Link

💾 [huggingface link](https://huggingface.co/datasets/Adeesh09/TacImag/tree/main)

```bash
bash scripts/download_data.sh all        # datasets (shared by both stages)
bash scripts/download_ckpts.sh all       # pretrained stage-1 checkpoints
```

For how to use the datasets and pretrained checkpoints, including running
stage 2 directly from the released stage-1 checkpoints vs training everything
from scratch, please refer to the [data README](download_dataset_ckpts.md).

## Training

Ready-to-run scripts for every task and modality:

```text
training/<task>/<stage>_<modality>.sh
```

**Stage 1: train the tactile imagination model**

```bash
bash training/usb/stage1_imagine_ff.sh      # imagine the contact force field
bash training/usb/stage1_imagine_rgb.sh     # imagine the tactile RGB image
```

**Stage 2: train the manipulation policy** (baselines: `vision`, `rgb_real`,
`ff_real`; ours: `rgb_imag`, `ff_imag`)

```bash
bash training/usb/stage2_ff_imag.sh         # ours: imagined force field
bash training/usb/stage2_vision.sh          # baseline: vision only
bash training/usb/stage2_ff_real.sh         # baseline: real force-field sensor
```

Each script takes an optional seed and Hydra overrides, for example
`bash training/usb/stage2_ff_imag.sh 43 stage1_ckpt=/path/to/latest.ckpt`.

Tasks: `usb`, `plug`, `pih`, `gear`, `bulb`, `sorting`.

To sweep a stage over all tasks (and optionally all modalities):

```bash
bash scripts/run_all.sh stage1 all
bash scripts/run_all.sh stage2 all 42       # all five modalities, seed 42
```

### Logging

Training logs to [Weights & Biases](https://wandb.ai) by default (projects
`tacimag_stage1` and `tacimag_stage2`; stage 2 also uploads success rate and
rollout videos every 50 epochs). To run without a wandb account, disable it:

```bash
bash training/usb/stage2_ff_imag.sh 42 logging.mode=offline
```

## Repository structure

```
TacImag/
├── training/                 # ready-to-run: training/<task>/<stage>_<modality>.sh
├── train_stage1.py / train_stage2.py    # the two entrypoints
├── tacimag/
│   ├── imagination/          # STAGE 1: diffusion tactile imagination
│   ├── policy/               # STAGE 2: policy w/ imagined-tactile conditioning
│   ├── config/               # Hydra configs (the two modality switches)
│   └── legacy_compat.py
├── scripts/                  # sweeps, data/checkpoint download
├── thirdparty/               # simulator + policy stack land here at install
└── install_tacimag.sh
```

## Citation

```bibtex
@article{zhang2026tacimag,
  title={Imagining the Sense of Touch: Touch-Informed Manipulation via Imagined Tactile Representations},
  author={Zhang, Zhiyuan and Desai, Adeesh and Hu, Jyun-Chi and Saka, Yosuke and Luu, Quan Khanh and Lei, Jiuzhou and Soleymanzadeh, Davood and Zhang, Bihao and Zheng, Minghui and She, Yu},
  journal={arXiv preprint arXiv:2607.01684},
  year={2026}
}
```

## Acknowledgments

TacImag builds on [ManiFeel](https://github.com/purdue-mars/manifeel),
[TacSL](https://iakinola23.github.io/tacsl/), and
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy);
the task datasets are hosted by the
[ManiFeel dataset](https://huggingface.co/datasets/purdue-mars/manifeel).
