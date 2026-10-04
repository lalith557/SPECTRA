"""
spectra/eval/per_image_eval.py

Produce a per-image metric record (one entry per test image) for any model.
Used as input to eval/statistical_significance.py.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_config, to_device, load_checkpoint
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_eval_dataset
from eval.metrics import compute_iou, compute_mae


@torch.no_grad()
def per_image(
    config_path: str, checkpoint_path: str, output_path: str,
    dataset: str = "trans10k", split: str = "test", threshold: float = 0.5,
):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SPECTRA(cfg, use_gnn=False).to(device).eval()
    load_checkpoint(checkpoint_path, model, device=str(device))

    ds = build_eval_dataset(name=dataset, root=cfg.data.root, split=split,
                            image_size=tuple(cfg.data.image_size))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    out: List[dict] = []
    H, W = cfg.data.image_size
    for batch in tqdm(loader, desc="per-image"):
        batch = to_device(batch, device)
        outputs = model(batch["image"], batch["image_t1"])
        prob = F.interpolate(outputs["seg_prob"], size=(H, W),
                             mode="bilinear", align_corners=False)
        prob = prob.squeeze().clamp(0, 1)
        gt   = batch["mask"].squeeze().float()

        pred_bin = (prob > threshold).long()
        iou = compute_iou(pred_bin, gt.long())
        mae = compute_mae(prob, gt)

        # f-measure / ber for single image
        tp = ((pred_bin == 1) & (gt == 1)).sum().item()
        fp = ((pred_bin == 1) & (gt == 0)).sum().item()
        fn = ((pred_bin == 0) & (gt == 1)).sum().item()
        tn = ((pred_bin == 0) & (gt == 0)).sum().item()
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f = (1 + 0.3) * precision * recall / (0.3 * precision + recall + eps)
        ber = 0.5 * ((fn / (tp + fn + eps)) + (fp / (fp + tn + eps)))

        out.append({
            "image_path": batch["image_path"][0],
            "iou": iou, "mae": mae, "f_measure": f, "ber": ber,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {output_path} ({len(out)} records)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--dataset",    default="trans10k")
    p.add_argument("--split",      default="test")
    p.add_argument("--threshold",  type=float, default=0.5)
    args = p.parse_args()
    per_image(args.config, args.checkpoint, args.output, args.dataset,
              args.split, args.threshold)
