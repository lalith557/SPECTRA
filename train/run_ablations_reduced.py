"""
spectra/train/run_ablations_reduced.py
Reduced ablation: 3 variants x 5 epochs to establish marginal contribution
of OFCV and BRF modules. Cheap enough to fit in one session while still
training each variant from scratch (vs. inference-time toggling, which only
shows reliance, not compensability).

Variants:
  - full      : use_ofcv=True,  use_brf=True
  - no_ofcv   : use_ofcv=False, use_brf=True
  - no_brf    : use_ofcv=True,  use_brf=False

Each variant saves its own checkpoint dir under results/causal_model/ablations/.
After all 3 finish, eval_checkpoints.py is invoked on each E5 checkpoint to
produce a single JSON with IoU/F/MAE/BER/ofcv_var per variant.
"""
import sys
import json
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.train_baseline import train
from utils import load_config, get_logger

logger = get_logger("spectra.ablations_reduced")

VARIANTS = [
    # "full" already completed in a prior run (Val E5 IoU=0.9133); skip to save time.
    # Re-add the dict above if you need to retrain it.
    {"name": "no_ofcv", "use_ofcv": False, "use_brf": True},
    {"name": "no_brf",  "use_ofcv": True,  "use_brf": False},
]


def run_variant(cfg_path, variant, base_save_dir):
    name = variant["name"]
    save_dir = Path(base_save_dir) / name / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    cfg.train.epochs = 5
    cfg.train.warmup_epochs = 1
    cfg.logging.save_dir = str(save_dir)
    cfg.experiment = f"ablation_{name}"

    logger.info("=" * 60)
    logger.info(f" VARIANT: {name}")
    logger.info(f"  use_ofcv = {variant['use_ofcv']}, use_brf = {variant['use_brf']}")
    logger.info(f"  save_dir = {save_dir}")
    logger.info("=" * 60)

    t0 = time.time()
    train(
        cfg_override=cfg,
        experiment_name=f"ablation_{name}_5ep",
        use_ofcv=variant["use_ofcv"],
        use_brf=variant["use_brf"],
        use_gnn=False,
    )
    dt = (time.time() - t0) / 60
    logger.info(f"VARIANT {name} done in {dt:.1f} min")
    return save_dir / "checkpoint_epoch005.pth"


def main():
    cfg_path = "configs/config.yaml"
    base_dir = "results/causal_model/ablations"
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    ckpts = {}
    for variant in VARIANTS:
        ckpt_path = run_variant(cfg_path, variant, base_dir)
        ckpts[variant["name"]] = str(ckpt_path)

    summary = {"variants": VARIANTS, "checkpoints": ckpts}
    out_json = Path(base_dir) / "variants_index.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"All variants done. Index: {out_json}")


if __name__ == "__main__":
    main()
