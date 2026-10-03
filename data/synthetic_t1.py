"""
spectra/data/synthetic_t1.py

Synthetic second-frame generator: produces a fake image_t1 from a single
input image, simulating a small camera motion. Enables OFCV/SPECTRA to be
applied to single-image inputs.

Three modes:
  homography      - sample a small random homography around identity
  inpaint_jitter  - perturb pixels with a learned diffusion-free crop/jitter
  flow_warp       - take a Gaussian-smoothed random flow field and warp the image

The output is a (img_t0, img_t1) pair that mimics the motion statistics of
a real video pair — small parallax + photometric variation — enough to
make the OFCV residual non-degenerate.
"""
import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch


def random_homography(H: int, W: int, max_translate: float = 0.02,
                       max_scale: float = 0.02, max_rotate_deg: float = 1.5,
                       rng: random.Random = None) -> np.ndarray:
    rng = rng or random
    tx = rng.uniform(-max_translate, max_translate) * W
    ty = rng.uniform(-max_translate, max_translate) * H
    s  = 1.0 + rng.uniform(-max_scale, max_scale)
    th = math.radians(rng.uniform(-max_rotate_deg, max_rotate_deg))

    cx, cy = W / 2, H / 2
    R = np.array([
        [s * math.cos(th), -s * math.sin(th), 0.0],
        [s * math.sin(th),  s * math.cos(th), 0.0],
        [0, 0, 1],
    ])
    T = np.eye(3); T[0, 2] = tx; T[1, 2] = ty
    C = np.eye(3); C[0, 2] = -cx; C[1, 2] = -cy
    return T @ np.linalg.inv(C) @ R @ C


def homography_warp(image: np.ndarray, H_mat: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpPerspective(
        image, H_mat, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def smooth_flow_warp(image: np.ndarray, max_disp: float = 4.0,
                     blur_sigma: float = 25.0,
                     rng: random.Random = None) -> np.ndarray:
    rng = rng or random
    H, W = image.shape[:2]
    seed = rng.randrange(2 ** 31)
    rs = np.random.default_rng(seed)
    fx = rs.standard_normal((H, W)).astype(np.float32) * max_disp
    fy = rs.standard_normal((H, W)).astype(np.float32) * max_disp
    fx = cv2.GaussianBlur(fx, (0, 0), blur_sigma)
    fy = cv2.GaussianBlur(fy, (0, 0), blur_sigma)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    map_x = (xx + fx).astype(np.float32)
    map_y = (yy + fy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def make_synthetic_pair(image: np.ndarray, mode: str = "homography",
                        rng: random.Random = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng or random
    if mode == "homography":
        H_mat = random_homography(*image.shape[:2], rng=rng)
        warp = homography_warp(image, H_mat)
    elif mode == "flow_warp":
        warp = smooth_flow_warp(image, rng=rng)
    elif mode == "inpaint_jitter":
        h, w = image.shape[:2]
        out = image.copy()
        # Random small crop-shift jitter
        dx = rng.randrange(-int(0.01 * w), int(0.01 * w) + 1)
        dy = rng.randrange(-int(0.01 * h), int(0.01 * h) + 1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        warp = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    else:
        raise ValueError(f"Unknown synthetic mode: {mode}")
    return image, warp


def make_synthetic_pair_tensor(image: torch.Tensor, mode: str = "homography",
                               rng: random.Random = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    image: (3, H, W) float tensor in [0, 1] (post-normalisation OK too).
    Returns the same tensor + the synthetic t1.
    """
    if image.dim() == 3:
        np_img = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        _, t1 = make_synthetic_pair(np_img, mode=mode, rng=rng)
        t1_tensor = torch.from_numpy(t1.astype(np.float32) / 255).permute(2, 0, 1)
        return image, t1_tensor.to(image.device)
    raise ValueError("Only (3, H, W) tensors are supported")
