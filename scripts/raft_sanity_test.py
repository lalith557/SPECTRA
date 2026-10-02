"""
scripts/raft_sanity_test.py  — CHEAP RAFT sanity test. RUN ON THE CLOUD GPU, not the laptop.

Confirms the corrected RAFT pipeline is active and behaves correctly:
  1. the code path uses [-1,1] (corrected); the old [0,255] is NOT present
  2. identical frames -> ~0 flow  (sanity: a correct pipeline MUST give near-zero)
  3. a real VGSD frame pair -> plausible non-zero flow
Uses the model's own RAFTWrapper (so it tests exactly what training will use).

This is a tiny diagnostic: a handful of forward passes, no training, no checkpoint needed.
FAILS LOUDLY (non-zero exit) if the corrected pipeline is not behaving as expected.

Usage:  python scripts/raft_sanity_test.py
"""
import os, sys, glob
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np, cv2, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flow.raft_wrapper import RAFTWrapper

IMEAN = np.array([0.485, 0.456, 0.406], np.float32); ISTD = np.array([0.229, 0.224, 0.225], np.float32)
FAIL = []


def load(p, size=448):
    img = cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (size, size))
    x = (img.astype(np.float32) / 255.0 - IMEAN) / ISTD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def mag(f):
    return float((f ** 2).sum(1).sqrt().mean())


def main():
    # 1. static code check
    rw = (ROOT / "flow" / "raft_wrapper.py").read_text(encoding="utf-8")
    assert "x * 2.0 - 1.0" in rw, "[-1,1] correction missing"
    assert "(x * 255.0).clamp(0, 255)" not in rw, "old [0,255] line still present"
    print("[1/3] code check: corrected [-1,1] active, old [0,255] absent  OK")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raft = RAFTWrapper(model_name="raft_large", iters=20).to(dev).eval()

    # pick a real VGSD pair (consecutive frames)
    clips = sorted(p for p in (ROOT / "VSGD_dataset" / "test").iterdir() if p.is_dir())
    fr = sorted(glob.glob(str(clips[3] / "JPEGImages" / "*.jpg")))
    x0, x1 = load(fr[len(fr) // 2]).to(dev), load(fr[len(fr) // 2 + 1]).to(dev)

    with torch.no_grad():
        dup = mag(raft(x0, x0)[0])    # identical frames
        real = mag(raft(x0, x1)[0])   # real consecutive pair
    print(f"[2/3] identical-frame flow = {dup:.3f} px   (expect ~0; must be < 1.0)")
    print(f"[3/3] real-pair flow       = {real:.3f} px   (expect plausible > 0.3)")

    if dup >= 1.0:
        FAIL.append(f"identical-frame flow {dup:.3f} px is NOT ~0 — pipeline broken (old [0,255]?)")
    if real <= 0.3:
        FAIL.append(f"real-pair flow {real:.3f} px implausibly low")
    if real <= dup * 3:
        FAIL.append(f"real ({real:.3f}) not clearly > identical ({dup:.3f}) — flow not discriminating motion")

    if FAIL:
        print("\nRAFT SANITY FAILED:")
        for f in FAIL:
            print("  - " + f)
        sys.exit(1)
    print("\nRAFT SANITY PASSED — corrected flow confirmed (identical~0, real>0).")


if __name__ == "__main__":
    main()
