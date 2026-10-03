"""
spectra/data/augmentation.py
Albumentations-based augmentation pipeline.
Supports dual-frame inputs (image + image_t1) with consistent spatial
transforms applied to both, while applying independent photometric
transforms (colour jitter, etc.) to each frame separately — mimicking
realistic video variation between frames.
"""
from typing import Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Helper: apply same spatial transform to both frames + mask
# ---------------------------------------------------------------------------

class DualFrameTransform:
    """
    Wrapper that applies:
      - Shared spatial transforms (flip, crop, resize) to image, image_t1, mask
      - Independent photometric transforms to image and image_t1 separately
    """

    def __init__(
        self,
        spatial: A.Compose,
        photometric: A.Compose,
        to_tensor: A.Compose,
    ):
        self.spatial     = spatial
        self.photometric = photometric
        self.to_tensor   = to_tensor

    def __call__(
        self,
        image: np.ndarray,
        mask:  np.ndarray,
        image_t1: np.ndarray,
    ) -> Dict:
        # 1. Shared spatial — fixes replay seed so both frames get identical crop/flip
        spatial_params = self.spatial.to_dict()
        # Apply spatial transform using the same random seed for both frames
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        aug_t0 = self.spatial(image=image, mask=mask)
        image  = aug_t0["image"]
        mask   = aug_t0["mask"]

        np.random.seed(seed)
        aug_t1   = self.spatial(image=image_t1)
        image_t1 = aug_t1["image"]
        # mask from t1 is discarded — we use t0 mask as GT

        # 2. Independent photometric per frame
        image    = self.photometric(image=image)["image"]
        image_t1 = self.photometric(image=image_t1)["image"]

        # 3. To tensor
        image    = self.to_tensor(image=image)["image"]
        image_t1 = self.to_tensor(image=image_t1)["image"]
        mask     = torch.from_numpy(mask).long()
        return {"image": image, "mask": mask, "image_t1": image_t1}


def get_train_transforms(image_size: Tuple[int, int] = (448, 448)) -> DualFrameTransform:
    H, W = image_size

    spatial = A.Compose([
        A.Resize(
            height=H,
            width=W,
            interpolation=cv2.INTER_LINEAR
        ),

        A.HorizontalFlip(p=0.5),
    ])

    photometric = A.Compose([
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05, p=0.8),
        
        # ── Reflection & Confusion Augmentations ──────────────────────────
        # Goal: Force the model to use OFCV/BRF physics instead of simple texture.
        A.OneOf([
            # Simulates glare/specular highlights on glass
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
            # Simulates direct light sources causing lens flare
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=1.0),
        ], p=0.4),
        
        A.OneOf([
            # Simulates refractive distortion through glass
            A.OpticalDistortion(distort_limit=0.2, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3),

        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        # A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
    ])

    to_tensor = A.Compose([ToTensorV2()])

    return DualFrameTransform(spatial=spatial, photometric=photometric, to_tensor=to_tensor)


def get_val_transforms(image_size: Tuple[int, int] = (448, 448)) -> DualFrameTransform:
    H, W = image_size

    spatial = A.Compose([
        A.Resize(height=H, width=W, interpolation=cv2.INTER_LINEAR),
    ])

    photometric = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
    ])

    to_tensor = A.Compose([ToTensorV2()])

    return DualFrameTransform(spatial=spatial, photometric=photometric, to_tensor=to_tensor)
