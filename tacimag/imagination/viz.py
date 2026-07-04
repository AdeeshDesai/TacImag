"""Sample-time visualizations for the imagination model.

save_tactile_image_pair  — GT / predicted tactile RGB image as PNGs
save_force_field_pair    — GT / predicted force field as TacSL-style
                           colored-arrow shear plots (green->red by normal
                           force, arrow direction/length from shear)
"""
import os

import numpy as np
import torch


def _to_numpy_chw(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def save_tactile_image_pair(gt_img, pred_img, epoch, b, base_dir):
    """gt_img, pred_img: [C, H, W], values in [0, 1]."""
    from PIL import Image

    save_dir = os.path.join(base_dir, str(epoch))
    os.makedirs(save_dir, exist_ok=True)

    def to_pil(x):
        x = _to_numpy_chw(x)
        if x.ndim == 3:
            x = x.transpose(1, 2, 0)  # HWC
        x = np.clip(x * 255.0 if x.max() <= 1.0 else x, 0, 255).astype(np.uint8)
        return Image.fromarray(x)

    to_pil(gt_img).save(os.path.join(save_dir, f"b_{b}_gt.png"))
    to_pil(pred_img).save(os.path.join(save_dir, f"b_{b}_pred.png"))


def _visualize_tactile_shear_image(tactile_normal_force, tactile_shear_force,
                                   normal_force_threshold=0.0008,
                                   shear_force_threshold=0.001,
                                   resolution=24):
    """TacSL shear-field viz. normal: [H,W], shear: [H,W,2].
    Returns BGR float image in [0,1]."""
    import cv2
    nrows, ncols = tactile_normal_force.shape[0], tactile_normal_force.shape[1]
    img = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=np.float32)
    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            loc1_x = loc0_x + tactile_shear_force[row, col][0] / shear_force_threshold * resolution
            loc1_y = loc0_y + tactile_shear_force[row, col][1] / shear_force_threshold * resolution
            color = (0.,
                     max(0., 1. - tactile_normal_force[row][col] / normal_force_threshold),
                     min(1., tactile_normal_force[row][col] / normal_force_threshold))
            cv2.arrowedLine(img, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)),
                            color, 4, tipLength=0.4)
    return img


def save_force_field_pair(gt_force, pred_force, epoch, b, base_dir,
                          normal_force_threshold=0.0008,
                          shear_force_threshold=0.001,
                          resolution=24):
    """gt_force, pred_force: [3, H, W] raw force fields."""
    import cv2
    save_dir = os.path.join(base_dir, str(epoch))
    os.makedirs(save_dir, exist_ok=True)

    def render(force):
        f = _to_numpy_chw(force).transpose(1, 2, 0)  # [H,W,3]
        img = _visualize_tactile_shear_image(
            f[..., 0], f[..., 1:],
            normal_force_threshold=normal_force_threshold,
            shear_force_threshold=shear_force_threshold,
            resolution=resolution)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(save_dir, f"b_{b}_gt.png"), render(gt_force))
    cv2.imwrite(os.path.join(save_dir, f"b_{b}_pred.png"), render(pred_force))
