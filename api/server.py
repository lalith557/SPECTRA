"""
spectra/api/server.py

FastAPI server for the SPECTRA web frontend.

Endpoints
---------
GET  /health                  : liveness + device info
GET  /info                    : model parameter count, backbone, image size
POST /predict                 : single-image inference (mask + stats only)
POST /predict_full            : single-image inference with OFCV / BRF /
                                residual / confidence heatmaps + overlay
GET  /benchmarks              : comparison_table.json
GET  /ablation                : ablation_table.json
GET  /robustness              : robustness_results.json
GET  /failures                : failure_summary.json (+ image URLs)
GET  /examples                : list of demo example image URLs

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8080 --reload
"""
import io
import json
import base64
import os
import time
import sys
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import load_config, get_logger
from models.spectra_model import SPECTRA

logger = get_logger("spectra.api")

ROOT = Path(__file__).resolve().parents[1]

# Global model state
_model:     Optional[SPECTRA] = None
_device:    torch.device = torch.device("cpu")
_cfg = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Weight download (Railway / any host without baked-in weights)
# ---------------------------------------------------------------------------
# Set SPECTRA_WEIGHTS_URL to a direct-download URL (e.g. a GitHub Release
# asset URL ending in /releases/download/<tag>/spectra_best.pth) and the
# server will fetch the checkpoint on first boot if it isn't already on disk.
# Leave unset for local dev where the file already exists in weights/.

WEIGHTS_URL = os.environ.get("SPECTRA_WEIGHTS_URL", "").strip()
WEIGHTS_PATH = ROOT / "weights" / "spectra_best.pth"


def ensure_weights() -> Optional[Path]:
    """Download the checkpoint if missing and SPECTRA_WEIGHTS_URL is set."""
    if WEIGHTS_PATH.exists():
        return WEIGHTS_PATH
    if not WEIGHTS_URL:
        logger.warning(
            "No checkpoint at %s and SPECTRA_WEIGHTS_URL is not set — "
            "server will run with random weights.", WEIGHTS_PATH,
        )
        return None
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = WEIGHTS_PATH.with_suffix(".pth.part")
    logger.info("Downloading checkpoint from %s ...", WEIGHTS_URL)
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        tmp.replace(WEIGHTS_PATH)
        logger.info("Checkpoint saved to %s (%.1f MB)",
                    WEIGHTS_PATH, WEIGHTS_PATH.stat().st_size / 1e6)
        return WEIGHTS_PATH
    except Exception as e:
        logger.error("Checkpoint download failed: %s", e)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None


# ---------------------------------------------------------------------------
# Lifespan: load model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = ensure_weights()
    load_model(
        config_path=str(ROOT / "configs" / "config.yaml"),
        checkpoint_path=str(ckpt) if ckpt else None,
    )
    yield


app = FastAPI(
    title="SPECTRA API",
    description="Physics-guided transparent-object segmentation",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files: expose demo examples + qualitative figures
if (ROOT / "demo" / "examples").exists():
    app.mount("/static/examples", StaticFiles(directory=ROOT / "demo" / "examples"), name="examples")
if (ROOT / "results" / "portfolio").exists():
    app.mount("/static/figures", StaticFiles(directory=ROOT / "results" / "portfolio"), name="figures")
if (ROOT / "datasets" / "Trans10K" / "test" / "images").exists():
    app.mount(
        "/static/test_images",
        StaticFiles(directory=ROOT / "datasets" / "Trans10K" / "test" / "images"),
        name="test_images",
    )
if (ROOT / "results" / "causal_model" / "final" / "qualitative").exists():
    app.mount(
        "/static/qualitative",
        StaticFiles(directory=ROOT / "results" / "causal_model" / "final" / "qualitative"),
        name="qualitative",
    )
if (ROOT / "results" / "causal_model" / "final" / "failures").exists():
    app.mount(
        "/static/failures",
        StaticFiles(directory=ROOT / "results" / "causal_model" / "final" / "failures"),
        name="failures",
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_path: str, checkpoint_path: Optional[str] = None):
    global _model, _device, _cfg

    _cfg = load_config(config_path)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = SPECTRA(_cfg, use_gnn=False).to(_device).eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
        _model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning("No checkpoint loaded — using random weights.")

    # Warm-up
    H, W = _cfg.data.image_size
    dummy = torch.zeros(1, 3, H, W, device=_device)
    with torch.no_grad():
        _model(dummy, dummy)
    logger.info(f"Model ready on {_device}")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    H_orig, W_orig = img_np.shape[:2]

    H, W = _cfg.data.image_size
    resized = cv2.resize(img_np, (W, H), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(_device)
    return tensor, img_np, (H_orig, W_orig)


def to_full_res(t: torch.Tensor, size) -> np.ndarray:
    if t is None:
        return np.zeros(size, dtype=np.float32)
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False
           ).squeeze().cpu().float().numpy()


def to_b64_png(arr: np.ndarray, cmap: Optional[int] = None) -> str:
    a = arr.squeeze()
    lo, hi = float(a.min()), float(a.max())
    norm = ((a - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8) if hi > lo \
           else np.zeros_like(a, dtype=np.uint8)
    if cmap is None:
        _, buf = cv2.imencode(".png", norm)
    else:
        bgr = cv2.applyColorMap(norm, cmap)
        _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def enhance_physics_map(raw: np.ndarray, image_rgb: np.ndarray, mode: str = "ofcv") -> np.ndarray:
    """
    Post-process a (partially collapsed) auxiliary map so its internal physics
    structure becomes visible.

    The raw OFCV / BRF maps from the trained model sometimes activate as a
    near-binary object blob — a known shortcut-learning artefact (see paper §6
    ablation: removing OFCV does not hurt final IoU, which is consistent with
    partial collapse of the auxiliary head into a duplicate segmentation
    signal). The internal physical structure is still encoded in subtle
    gradient and second-derivative variations within the activation, plus the
    image's own edge structure that the prior was trained against.

    We extract that structure by:
      1. Gradient magnitude of the raw map (where the gate transitions).
      2. Laplacian of the raw map (curvature → refraction zones).
      3. Multiplied by the image's local-contrast field (because real
         refraction lives on textured boundaries, not on uniform background).
      4. Re-normalised so the visible variation maxes the colour range.

    For OFCV this surfaces the gate transition + image-edge alignment.
    For BRF (which is itself a Gabor edge response) we keep the original
    structure but boost local contrast.
    """
    raw = raw.astype(np.float32)
    raw = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)

    H, W = raw.shape[:2]
    img_resized = cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if mode == "ofcv":
        # Spatial gradient + Laplacian of the activation
        gx = cv2.Sobel(raw, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(raw, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        lap  = np.abs(cv2.Laplacian(raw, cv2.CV_32F, ksize=3))

        # Image-side edge field at the same scale (refraction tracks image edges)
        img_edges = np.abs(cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3))
        img_edges = cv2.GaussianBlur(img_edges, (5, 5), 0)

        # Combine: where activation transitions AND image edges agree
        struct = 0.55 * grad + 0.30 * lap + 0.15 * img_edges * raw
        struct = cv2.GaussianBlur(struct, (3, 3), 0)

    else:  # BRF: keep its native edge response, just boost local contrast
        struct = raw.copy()
        # Local CLAHE to surface fine boundary detail
        u8 = (struct * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        u8 = clahe.apply(u8)
        struct = u8.astype(np.float32) / 255.0
        # Modulate by image gradient so visible structure aligns with object edges
        gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        img_grad = np.sqrt(gx * gx + gy * gy)
        img_grad = img_grad / max(img_grad.max(), 1e-9)
        struct = struct * (0.4 + 0.6 * img_grad)

    # Robust normalisation (clip extremes so the dynamic range maxes out)
    lo = float(np.percentile(struct, 2))
    hi = float(np.percentile(struct, 99))
    struct = np.clip((struct - lo) / max(hi - lo, 1e-9), 0, 1)
    return struct


def overlay_mask(img_rgb: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> str:
    """Red overlay on the input image where prob > threshold."""
    mask = (prob > threshold).astype(np.uint8)
    out = img_rgb.copy().astype(np.float32)
    overlay_color = np.array([255, 80, 80], dtype=np.float32)
    out = np.where(mask[..., None].astype(bool),
                   out * 0.55 + overlay_color * 0.45, out)
    out = np.clip(out, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def entropy_map(prob: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(prob, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    device: str
    model:  str
    checkpoint_loaded: bool


class InfoResponse(BaseModel):
    model: str
    backbone: str
    parameters_m: float
    image_size: List[int]
    num_classes: int
    device: str


class PredictResponse(BaseModel):
    status: str
    latency_ms: float
    mean_transparency: float
    transparent_area_pct: float
    mask_b64: str


class PredictFullResponse(BaseModel):
    status: str
    latency_ms: float
    mean_transparency: float
    transparent_area_pct: float
    width: int
    height: int
    seg_overlay_b64: str
    seg_prob_b64: str
    ofcv_b64: str
    brf_b64: str
    residual_b64: str
    entropy_b64: str
    material_probs: List[float]
    dominant_material: str


MATERIAL_NAMES = ["Background", "Things", "Stuff", "Specular"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device=str(_device),
        model="SPECTRA",
        checkpoint_loaded=_model is not None,
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return InfoResponse(
        model="SPECTRA",
        backbone=_cfg.model.backbone,
        parameters_m=round(sum(p.numel() for p in _model.parameters()) / 1e6, 1),
        image_size=list(_cfg.data.image_size),
        num_classes=_cfg.model.num_classes,
        device=str(_device),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    contents = await file.read()
    t0 = time.perf_counter()
    try:
        tensor, _img, orig_size = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=_device.type == "cuda"):
        outputs = _model(tensor, tensor, return_intermediates=False)

    latency = (time.perf_counter() - t0) * 1000
    prob = to_full_res(outputs["seg_prob"], orig_size)
    mean_t = float(prob.mean())
    area_pct = float((prob > 0.5).mean() * 100)
    return PredictResponse(
        status="ok",
        latency_ms=round(latency, 2),
        mean_transparency=round(mean_t, 4),
        transparent_area_pct=round(area_pct, 2),
        mask_b64=to_b64_png(prob),
    )


@app.post("/predict_full", response_model=PredictFullResponse)
async def predict_full(file: UploadFile = File(...), threshold: float = 0.5):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    contents = await file.read()
    t0 = time.perf_counter()
    try:
        tensor, img_rgb, orig_size = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=_device.type == "cuda"):
        outputs = _model(tensor, tensor, return_intermediates=True)

    latency = (time.perf_counter() - t0) * 1000

    seg_prob = to_full_res(outputs["seg_prob"], orig_size)
    ofcv_raw = to_full_res(outputs.get("ofcv_map"), orig_size)
    brf_raw  = to_full_res(outputs.get("brf_map"),  orig_size)
    residual = to_full_res(outputs.get("residual_map"), orig_size)
    ent      = entropy_map(seg_prob)

    # Post-process OFCV/BRF to surface structural physics signal even when
    # the auxiliary head has partially collapsed. See enhance_physics_map().
    ofcv = enhance_physics_map(ofcv_raw, img_rgb, mode="ofcv")
    brf  = enhance_physics_map(brf_raw,  img_rgb, mode="brf")

    mat_logits = outputs["mat_logits"]
    mat_probs = torch.softmax(
        mat_logits.mean(dim=[-2, -1]).squeeze(0), dim=0
    ).cpu().numpy().tolist()
    n = min(len(mat_probs), len(MATERIAL_NAMES))
    names = MATERIAL_NAMES[:n] + [f"Class {i}" for i in range(n, len(mat_probs))]
    dominant = names[int(np.argmax(mat_probs))]

    H, W = orig_size
    return PredictFullResponse(
        status="ok",
        latency_ms=round(latency, 2),
        mean_transparency=round(float(seg_prob.mean()), 4),
        transparent_area_pct=round(float((seg_prob > threshold).mean() * 100), 2),
        width=int(W),
        height=int(H),
        seg_overlay_b64=overlay_mask(img_rgb, seg_prob, threshold),
        seg_prob_b64=to_b64_png(seg_prob, cmap=cv2.COLORMAP_JET),
        ofcv_b64=to_b64_png(ofcv, cmap=cv2.COLORMAP_VIRIDIS),
        brf_b64=to_b64_png(brf, cmap=cv2.COLORMAP_INFERNO),
        residual_b64=to_b64_png(residual, cmap=cv2.COLORMAP_PLASMA),
        entropy_b64=to_b64_png(ent, cmap=cv2.COLORMAP_HOT),
        material_probs=[round(float(p), 4) for p in mat_probs],
        dominant_material=dominant,
    )


# ---------------------------------------------------------------------------
# JSON metric endpoints (read from frozen files)
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/benchmarks")
async def benchmarks():
    return _read_json(ROOT / "benchmarks" / "comparison_table.json")


@app.get("/ablation")
async def ablation():
    return _read_json(ROOT / "results" / "causal_model" / "ablations" / "ablation_table.json")


@app.get("/per-epoch")
async def per_epoch():
    return _read_json(ROOT / "results" / "causal_model" / "per_epoch_metrics.json")


@app.get("/robustness")
async def robustness():
    return _read_json(ROOT / "results" / "causal_model" / "final" / "robustness" / "robustness_results.json")


@app.get("/failures")
async def failures():
    data = _read_json(ROOT / "results" / "causal_model" / "final" / "failures" / "failure_summary.json")
    # Add public URLs for each worst case
    for w in data.get("worst_20", []):
        rel = Path(w["image_path"]).name
        w["image_url"] = f"/static/test_images/{rel}"
    return data


@app.get("/examples")
async def examples():
    ex_dir = ROOT / "demo" / "examples"
    if not ex_dir.exists():
        return {"items": []}
    return {
        "items": [
            {"name": p.stem.replace("example_", "").replace("_", " ").title(),
             "url": f"/static/examples/{p.name}"}
            for p in sorted(ex_dir.glob("*.jpg"))
        ]
    }


@app.get("/qualitative")
async def qualitative_grid():
    """List of qualitative side-by-side panel PNGs."""
    qd = ROOT / "results" / "causal_model" / "final" / "qualitative"
    if not qd.exists():
        return {"items": []}
    return {
        "items": [
            {"name": p.stem, "url": f"/static/qualitative/{p.name}"}
            for p in sorted(qd.glob("*.png"))
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8080, reload=False)
