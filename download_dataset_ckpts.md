# Download and Extract

USB insertion example:

- Dataset: [usb_quan_Aug05.zip](https://huggingface.co/datasets/purdue-mars/manifeel/blob/main/data/usb_quan_Aug05.zip)
- Pretrained stage-1 checkpoints: [tacff_usb](https://huggingface.co/datasets/Adeesh09/TacImag/blob/main/checkpoints/stage1/tacff_usb/latest.ckpt), [tacrgb_usb](https://huggingface.co/datasets/Adeesh09/TacImag/blob/main/checkpoints/stage1/tacrgb_usb/latest.ckpt)

The helper scripts download AND place everything automatically:

```bash
bash scripts/download_data.sh usb      # dataset  (~4 GB per task; or: all)
bash scripts/download_ckpts.sh usb     # stage-1 checkpoints (or: all)
```

### Dataset

`download_data.sh` creates:

```text
data/
└── usb_quan_Aug05/
    ├── data/     # per-key episode arrays (wrist, state, action,
    │             #   right_tactile_camera_taxim, tactile_force_field_right, ...)
    └── meta/
```

One dataset per task is shared by **both stages**. Stage 1 reads the tactile
keys as its target, stage 2 reads the vision/state/action keys.

| task | dataset directory |
|---|---|
| usb | `data/usb_quan_Aug05` |
| plug | `data/plug_quan_Aug02` |
| pih | `data/pih_quan_June06` |
| gear | `data/gear_quan_Sep15` |
| bulb | `data/bulb_quan_Sep19` |
| sorting | `data/sorting_quan_Aug8` |

### Checkpoints

`download_ckpts.sh` creates:

```text
data/
└── checkpoints/
    └── stage1/
        ├── tacff_usb/latest.ckpt      # force-field imagination
        └── tacrgb_usb/latest.ckpt     # tactile-RGB imagination
```

These are exactly the paths the stage-2 configs default to, so no arguments
are needed afterwards.

## Using the pretrained checkpoints

Skip stage 1 entirely and train the policy directly:

```bash
bash training/usb/stage2_ff_imag.sh          # ours: imagined force field
bash training/usb/stage2_rgb_imag.sh         # ours: imagined tactile RGB
bash training/usb/stage2_vision.sh           # baselines need no checkpoint
```

## Training from scratch

Train your own stage-1 imagination model first; its checkpoint is written to
`data/outputs/stage1/<modality>_<task>/<seed>/checkpoints/latest.ckpt`:

```bash
bash training/usb/stage1_imagine_ff.sh
bash training/usb/stage2_ff_imag.sh 42 \
    stage1_ckpt=data/outputs/stage1/tacff_usb/42/checkpoints/latest.ckpt
```
