"""
eval/diag_raft_norm.py  — controlled RAFT normalization + identical-frame sanity test.

Tests, on the SAME VGSD pairs, four flow computations:
  A) wrapper path ([0,255] input)  on REAL pair   (I_t, I_{t+1})
  B) wrapper path ([0,255] input)  on DUP  pair    (I_t, I_t)
  C) official torchvision transform on REAL pair
  D) official torchvision transform on DUP  pair

Sanity law: for IDENTICAL frames, flow MUST be ~0 for any correct pipeline
(identical features -> correlation peaks at zero displacement). If B >> 0 and
D ~ 0, the [0,255] input is the culprit (F3). If BOTH B and D >> 0, the anomaly
is in the harness, not the normalization. Read-only, tiny.
"""
import os, sys, glob
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, cv2, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMEAN = np.array([0.485, 0.456, 0.406], np.float32); ISTD = np.array([0.229, 0.224, 0.225], np.float32)


def load_norm(path, size=448):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    x = (img.astype(np.float32) / 255.0 - IMEAN) / ISTD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def mag(f):
    return float((f ** 2).sum(1).sqrt().mean())


def main():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    w = Raft_Large_Weights.DEFAULT
    net = raft_large(weights=w).eval().to(DEV)
    for p in net.parameters():
        p.requires_grad_(False)
    tf = w.transforms()  # official: expects uint8/float, returns model-ready [-1,1]

    clips = sorted(p for p in (ROOT / "VSGD_dataset" / "test").iterdir() if p.is_dir())[:8]
    A, B, C, D = [], [], [], []
    for clip in clips:
        fr = sorted(glob.glob(str(clip / "JPEGImages" / "*.jpg")))
        if len(fr) < 3:
            continue
        i = len(fr) // 2
        x0 = load_norm(fr[i]).to(DEV); x1 = load_norm(fr[i + 1]).to(DEV)
        # wrapper [0,255] path
        def denorm(x):
            m = torch.tensor(IMEAN, device=x.device).view(1, 3, 1, 1)
            s = torch.tensor(ISTD, device=x.device).view(1, 3, 1, 1)
            return ((x * s + m) * 255.0).clamp(0, 255)
        t0, t1 = denorm(x0), denorm(x1)
        with torch.no_grad():
            A.append(mag(net(t0, t1, num_flow_updates=20)[-1]))
            B.append(mag(net(t0, t0, num_flow_updates=20)[-1]))
            # official transform path (needs [0,1] or uint8 CHW float; transforms handles scaling)
            # rebuild [0,1] RGB tensors
            def raw01(path, size=448):
                img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
                return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            r0 = raw01(fr[i]).to(DEV); r1 = raw01(fr[i + 1]).to(DEV)
            c0, c1 = tf(r0, r1)
            C.append(mag(net(c0, c1, num_flow_updates=20)[-1]))
            cc0, cc1 = tf(r0, r0)
            D.append(mag(net(cc0, cc1, num_flow_updates=20)[-1]))
        print(f"  {clip.name}: A[0-255,real]={A[-1]:.2f}  B[0-255,DUP]={B[-1]:.2f}  "
              f"C[official,real]={C[-1]:.2f}  D[official,DUP]={D[-1]:.2f}", flush=True)
    import statistics as st
    print("\n== MEANS ==")
    print(f"  A [0-255 input, REAL pair] : {st.mean(A):.2f} px")
    print(f"  B [0-255 input, DUP  pair] : {st.mean(B):.2f} px   (should be ~0 if pipeline correct)")
    print(f"  C [official   , REAL pair] : {st.mean(C):.2f} px")
    print(f"  D [official   , DUP  pair] : {st.mean(D):.2f} px   (should be ~0)")
    verdict = ("F3 CONFIRMED: [0-255] breaks identical-frame flow" if st.mean(B) > 5 and st.mean(D) < 5
               else "HARNESS/OTHER: identical-frame flow nonzero for BOTH paths" if st.mean(D) > 5
               else "RAFT OK on identical frames for both; anomaly elsewhere")
    print("  VERDICT:", verdict)


if __name__ == "__main__":
    main()
