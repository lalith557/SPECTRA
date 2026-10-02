"""
scripts/cloud_preflight.py  — CLOUD pre-flight verification. RUN ON THE CLOUD GPU, not the laptop.

Verifies the environment, datasets, checkpoints, corrected RAFT, and output writability
BEFORE any expensive training. FAILS LOUDLY (non-zero exit) on any problem.

It does NOT train and does NOT run an expensive evaluation. It may instantiate RAFT/DINOv2
to confirm they load (one-time weight download) only if --load-models is passed; by default
it performs import/cache/code checks without heavy forward passes.

Usage:
  python scripts/cloud_preflight.py                 # checks only
  python scripts/cloud_preflight.py --load-models   # also instantiate RAFT+DINOv2 (downloads weights)
"""
import argparse, importlib, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL = []
OKS = []


def ok(m):  OKS.append(m);  print(f"  [OK]   {m}")
def bad(m): FAIL.append(m); print(f"  [FAIL] {m}")


def check_python():
    v = sys.version_info
    (ok if v >= (3, 10) else bad)(f"Python {sys.version.split()[0]} (need >=3.10)")


def check_pkg(name, imp=None, minver=None):
    try:
        mod = importlib.import_module(imp or name)
        ver = getattr(mod, "__version__", "?")
        ok(f"{name} {ver}")
        return mod
    except Exception as e:
        bad(f"{name} import failed: {type(e).__name__}: {e}")
        return None


def check_torch_cuda():
    try:
        import torch
        ok(f"torch {torch.__version__} (cuda build {torch.version.cuda})")
        if not torch.cuda.is_available():
            bad("CUDA not available (torch.cuda.is_available()=False)"); return
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"GPU: {name}  VRAM: {vram:.1f} GB")
        if vram < 8:
            bad(f"VRAM {vram:.1f} GB < 8 GB minimum for 448/bs4")
        elif vram < 12:
            ok("VRAM >= 8 GB (P1 at 448/bs4 fits but is tight; consider bs via protocol only if authorized)")
        else:
            ok("VRAM >= 12 GB (comfortable)")
    except Exception as e:
        bad(f"torch/cuda check failed: {e}")


def check_raft_corrected():
    rw = (ROOT / "flow" / "raft_wrapper.py").read_text(encoding="utf-8")
    if "x * 2.0 - 1.0" in rw and "(x * 255.0).clamp(0, 255)" not in rw:
        ok("RAFT normalization = corrected [-1,1] (old [0,255] absent)")
    else:
        bad("RAFT normalization is NOT the corrected [-1,1] version — fix flow/raft_wrapper.py before running")
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights  # noqa
        ok("torchvision RAFT weights enum importable")
    except Exception as e:
        bad(f"torchvision RAFT import failed: {e}")


def check_datasets():
    checks = [
        ("Trans10K train images", ROOT / "datasets/Trans10K/train/images", 5000),
        ("Trans10K train masks",  ROOT / "datasets/Trans10K/train/masks", 5000),
        ("Trans10K val images",   ROOT / "datasets/Trans10K/validation/images", 1000),
        ("Trans10K val masks",    ROOT / "datasets/Trans10K/validation/masks", 1000),
        ("Trans10K test images",  ROOT / "datasets/Trans10K/test/images", 4428),
        ("Trans10K test masks",   ROOT / "datasets/Trans10K/test/masks", 4428),
    ]
    for label, d, expect in checks:
        n = len(list(d.glob("*"))) if d.exists() else 0
        (ok if n == expect else bad)(f"{label}: {n} (expect {expect}) @ {d.relative_to(ROOT)}")
    vg = ROOT / "VSGD_dataset/test"
    nclips = len([p for p in vg.glob("*") if p.is_dir()]) if vg.exists() else 0
    (ok if nclips >= 100 else bad)(f"VGSD test clips: {nclips} (expect ~105) [needed for P3 / cross-dataset]")
    cp = len(list((ROOT / "clearpose_dataset").rglob("scene*"))) if (ROOT / "clearpose_dataset").exists() else 0
    (ok if cp > 0 else bad)(f"ClearPose scenes: {cp} [needed for P1.5 cross-dataset]")


def check_outputs():
    try:
        d = ROOT / "EXPERIMENTS" / "_preflight_write_test"
        d.mkdir(parents=True, exist_ok=True)
        (d / "t.txt").write_text("ok", encoding="utf-8")
        (d / "t.txt").unlink(); d.rmdir()
        ok("EXPERIMENTS/ is writable")
    except Exception as e:
        bad(f"EXPERIMENTS/ not writable: {e}")


def check_config():
    try:
        sys.path.insert(0, str(ROOT))
        from utils import load_config
        cfg = load_config("configs/config.yaml")
        assert list(cfg.data.image_size) == [448, 448]
        assert cfg.train.epochs == 10 and cfg.train.batch_size == 4 and cfg.seed == 42
        ok("config.yaml protocol: 448 / 10ep / bs4 / seed42")
        if cfg.logging.get("wandb_project"):
            bad("wandb_project is set — will attempt network init; set to null for offline")
        else:
            ok("wandb disabled (offline-safe)")
    except Exception as e:
        bad(f"config check failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-models", action="store_true", help="also instantiate RAFT+DINOv2 (downloads weights)")
    args = ap.parse_args()
    print("=== SPECTRA cloud preflight ===")
    print("[python]");   check_python()
    print("[packages]")
    for n, imp in [("torch", None), ("torchvision", None), ("numpy", None), ("opencv-python", "cv2"),
                   ("albumentations", None), ("einops", None), ("timm", None), ("pyyaml", "yaml"),
                   ("scikit-image", "skimage"), ("scipy", None), ("matplotlib", None)]:
        check_pkg(n, imp)
    print("[cuda/gpu]");  check_torch_cuda()
    print("[raft]");      check_raft_corrected()
    print("[datasets]");  check_datasets()
    print("[config]");    check_config()
    print("[outputs]");   check_outputs()
    if args.load_models:
        print("[models] instantiating (downloads weights)…")
        try:
            sys.path.insert(0, str(ROOT))
            import torch
            from utils import load_config
            from models.spectra_model import SPECTRA
            cfg = load_config("configs/config.yaml")
            m = SPECTRA(cfg, use_gnn=False, use_ofcv=True, use_brf=True)
            n = sum(p.numel() for p in m.parameters())
            ok(f"SPECTRA instantiated ({n:,} params); DINOv2+RAFT loaded")
            del m
        except Exception as e:
            bad(f"model instantiation failed: {e}")

    print(f"\n=== RESULT: {len(OKS)} OK, {len(FAIL)} FAIL ===")
    if FAIL:
        print("PREFLIGHT FAILED — resolve the above before launching any experiment.")
        for f in FAIL:
            print("  - " + f)
        sys.exit(1)
    print("PREFLIGHT PASSED — environment is ready. Proceed to the RAFT sanity test, then authorize P1.")


if __name__ == "__main__":
    main()
