"""
spectra/eval/eval_checkpoints.py
Run validation on a list of saved checkpoints and report IoU/F/MAE/BER plus
mean OFCV variance and mean BRF activation on the val set.

Used after the causal-model 5-epoch run to populate the per-epoch metrics
table that STEP 5 (decision point) depends on.
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import amp

from utils import load_config, get_logger, to_device
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_dataloaders
from eval.metrics import TransparentObjectMetrics

logger = get_logger("spectra.eval_ckpts")


@torch.no_grad()
def evaluate_checkpoint(model, val_loader, device):
    model.eval()
    metrics = TransparentObjectMetrics()
    ofcv_vars = []
    brf_means = []
    n = 0

    for batch in val_loader:
        batch = to_device(batch, device)
        with amp.autocast("cuda"):
            outputs = model(
                image=batch["image"],
                image_t1=batch["image_t1"],
                return_intermediates=True,
            )
        metrics.update(outputs["seg_prob"], batch["mask"])
        ofcv = outputs.get("ofcv_map")
        if ofcv is not None:
            # Per-image variance — measures image-dependent structure
            for i in range(ofcv.shape[0]):
                ofcv_vars.append(ofcv[i].var().item())
        brf = outputs.get("brf_map")
        if brf is not None:
            for i in range(brf.shape[0]):
                brf_means.append(brf[i].mean().item())
        n += 1

    results = metrics.compute()
    if ofcv_vars:
        import statistics as st
        results["ofcv_var_mean"] = sum(ofcv_vars) / len(ofcv_vars)
        results["ofcv_var_std"] = st.pstdev(ofcv_vars) if len(ofcv_vars) > 1 else 0.0
    if brf_means:
        import statistics as st
        results["brf_mean"] = sum(brf_means) / len(brf_means)
        results["brf_std"] = st.pstdev(brf_means) if len(brf_means) > 1 else 0.0
    return results


def main(cfg_path, checkpoints, out_json):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = build_dataloaders(cfg)

    # Build model once; reload weights per checkpoint
    model = SPECTRA(cfg, use_gnn=False, use_ofcv=True, use_brf=True).to(device)

    all_results = {}
    for ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            logger.warning(f"missing: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            ep = ckpt.get("epoch", "?")
        else:
            model.load_state_dict(ckpt, strict=False)
            ep = "?"

        logger.info(f"=== {Path(ckpt_path).name} (epoch {ep}) ===")
        r = evaluate_checkpoint(model, val_loader, device)
        all_results[Path(ckpt_path).name] = {"epoch": ep, **r}
        logger.info(
            f"IoU={r['iou']:.4f} F={r['f_measure']:.4f} MAE={r['mae']:.4f} "
            f"BER={r['ber']:.4f} ofcv_var_mean={r.get('ofcv_var_mean', 0):.3e} "
            f"brf_mean={r.get('brf_mean', 0):.3e}"
        )

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"saved: {out_json}")
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", default="results/causal_model/per_epoch_metrics.json")
    args = p.parse_args()
    main(args.config, args.checkpoints, args.out)
