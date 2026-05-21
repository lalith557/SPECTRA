"""
spectra/train/run_ablations.py
Orchestrates the running of 5 model variants to establish
the individual contribution of each module.
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.train_baseline import train
from utils import load_config
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


ABLATIONS = [
    {
        "name": "ablation_dinov2",
        "flags": {"use_ofcv": False, "use_brf": False, "use_gnn": False},
        "desc": "Backbone + FusionHead (no OFCV, no BRF)"
    },
    {
        "name": "ablation_brf",
        "flags": {"use_ofcv": False, "use_brf": True, "use_gnn": False},
        "desc": "Backbone + BRF + FusionHead"
    },
    {
        "name": "ablation_ofcv",
        "flags": {"use_ofcv": True, "use_brf": False, "use_gnn": False},
        "desc": "Backbone + OFCV + FusionHead"
    },
    {
        "name": "ablation_gnn",
        "flags": {"use_ofcv": False, "use_brf": False, "use_gnn": True},
        "desc": "Backbone + MBP-GNN + FusionHead"
    },
    {
        "name": "ablation_full",
        "flags": {"use_ofcv": True, "use_brf": True, "use_gnn": True},
        "desc": "All modules"
    }
]

def run_all_ablations(cfg_path: str):
    print("=" * 60)
    print(" SPECTRA Ablation Runner")
    print("=" * 60)
    
    cfg = load_config(cfg_path)
    # Override epochs for ablation (typically shorter than full train, e.g. 30)
    ablation_epochs = cfg.train.get("ablation_epochs", 30)
    cfg.train.epochs = ablation_epochs
    
    print(f"Running {len(ABLATIONS)} variants for {ablation_epochs} epochs each.\n")
    
    results = []

    for variant in ABLATIONS:
        print("-" * 60)
        print(f" Starting Variant: {variant['name']}")
        print(f" Description:      {variant['desc']}")
        print(f" Flags:            {variant['flags']}")
        print("-" * 60)
        
        # We must load a fresh config object in case it was mutated
        v_cfg = load_config(cfg_path)
        v_cfg.train.epochs = ablation_epochs
        
        # Redirect output or just let it print
        try:
            train(
                cfg_override=v_cfg,
                experiment_name=variant["name"],
                use_ofcv=variant["flags"]["use_ofcv"],
                use_brf=variant["flags"]["use_brf"],
                use_gnn=variant["flags"]["use_gnn"],
            )
            # Find best checkpoint for this run to get the best IoU
            # We don't return best_iou directly from train(), so we parse the checkpoints dir
            ckpt_dir = Path(v_cfg.logging.save_dir)
            # Let's assume the train loop saved a file, but to be safe we'll record that it completed
            results.append({
                "Variant": variant["name"],
                "Status": "Completed",
                "Description": variant["desc"]
            })
        except Exception as e:
            print(f"Variant {variant['name']} failed with error: {e}")
            results.append({
                "Variant": variant["name"],
                "Status": f"Failed: {e}",
                "Description": variant["desc"]
            })
    
    print("=" * 60)
    print(" Ablation Suite Complete")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    df.to_csv("ablation_results.csv", index=False)
    print("Results saved to ablation_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    run_all_ablations(args.config)
