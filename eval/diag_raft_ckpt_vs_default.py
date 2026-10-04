"""
Resolve the contradiction: does the CHECKPOINT's RAFT differ from torchvision DEFAULT,
and does a FRESH (ablation-style) SPECTRA feed RAFT broken [0,255] flow?

Compares, on the SAME identical-frame (duplicate) pair:
  (1) model.flow_model AFTER loading spectra_best.pth   (Phase-3/4 path)
  (2) a FRESH RAFTWrapper, no checkpoint                 (VGSD-ablation path)
and measures the weight difference between the two RAFT nets.
Read-only.
"""
import os, sys, glob
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, cv2, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from utils import load_config
from models.spectra_model import SPECTRA
from flow.raft_wrapper import RAFTWrapper
IMEAN = np.array([0.485,0.456,0.406],np.float32); ISTD=np.array([0.229,0.224,0.225],np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_norm(p,size=448):
    img=cv2.resize(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB),(size,size))
    x=(img.astype(np.float32)/255.0-IMEAN)/ISTD
    return torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
def mag(f): return float((f**2).sum(1).sqrt().mean())

cfg=load_config("configs/config.yaml")
clip=sorted(p for p in (ROOT/"VSGD_dataset"/"test").iterdir() if p.is_dir())[3]
fr=sorted(glob.glob(str(clip/"JPEGImages"/"*.jpg"))); x0=load_norm(fr[len(fr)//2]).to(DEV)

# (2) fresh wrapper (ablation path)
fresh=RAFTWrapper(model_name=cfg.flow.model, iters=cfg.flow.iters).to(DEV).eval()
with torch.no_grad():
    ff,_=fresh(x0,x0); print(f"(2) FRESH RAFTWrapper  [ablation path]  flow on DUP pair = {mag(ff):.3f} px")

# (1) checkpoint-loaded model
model=SPECTRA(cfg,use_gnn=False,use_ofcv=True,use_brf=True).to(DEV).eval()
sd=torch.load(ROOT/"checkpoints"/"spectra_best.pth",map_location="cpu",weights_only=False)
sd=sd.get("model_state_dict") or sd
model.load_state_dict(sd,strict=False)
with torch.no_grad():
    ff2,_=model.flow_model(x0,x0); print(f"(1) CKPT-loaded model  [Phase3/4 path]  flow on DUP pair = {mag(ff2):.3f} px")

# weight diff between the two RAFT nets
d_fresh=dict(fresh.raft.state_dict()); d_ck=dict(model.flow_model.raft.state_dict())
tot=0.0; n=0; maxd=0.0
for k in d_fresh:
    if k in d_ck and d_fresh[k].shape==d_ck[k].shape and d_fresh[k].dtype.is_floating_point:
        dd=(d_fresh[k].float()-d_ck[k].float()).abs()
        tot+=float(dd.sum()); n+=dd.numel(); maxd=max(maxd,float(dd.max()))
print(f"RAFT weight L1 diff (fresh vs ckpt): mean={tot/max(n,1):.3e}  max={maxd:.3e}  over {n} params")
# how many keys of ckpt are under flow_model.raft
nkeys=sum(1 for k in sd if k.startswith("flow_model.raft."))
print(f"checkpoint keys under flow_model.raft.* = {nkeys}")
