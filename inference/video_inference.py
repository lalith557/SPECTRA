"""
spectra/inference/video_inference.py
Real-time video and webcam inference for SPECTRA.
Handles temporal frame pairs for OFCV — which requires motion.
Target: ≥ 15 FPS on RTX 3080 / A10G.

Usage:
    python inference/video_inference.py --source webcam
    python inference/video_inference.py --source path/to/video.mp4
    python inference/video_inference.py --source path/to/video.mp4 --save
"""
import sys
import time
import argparse
import collections
from pathlib import Path
from typing import Optional, Tuple, Deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils import load_config, get_logger
from models.spectra_model import SPECTRA
from data.augmentation import get_val_transforms
from inference.vis_utils import (
    overlay_mask_on_image,
    draw_hud,
    colorise_probability_map,
)

logger = get_logger("spectra.video")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Frame pre-processing (fast, no albumentations overhead per frame)
# ---------------------------------------------------------------------------

def preprocess_frame(
    frame_bgr: np.ndarray,
    target_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    BGR uint8 frame → normalised (1, 3, H, W) tensor on device.
    Avoids albumentations overhead for real-time performance.
    """
    H, W = target_size
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    norm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD   # (H, W, 3)
    t    = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(device)   # (1, 3, H, W)
    return t


def postprocess_prob(
    seg_prob: torch.Tensor,   # (1, 1, H, W)
    orig_hw:  Tuple[int, int],
) -> np.ndarray:
    """Return probability map as float32 (H, W) at original resolution."""
    prob = F.interpolate(
        seg_prob,
        size=orig_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().float().numpy()
    return prob


# ---------------------------------------------------------------------------
# SPECTRA video runner
# ---------------------------------------------------------------------------

class SPECTRAVideoRunner:
    """
    Stateful video inference runner.
    Maintains a rolling frame buffer for temporal flow computation.

    Args:
        model:       loaded SPECTRA model (eval mode)
        device:      inference device
        target_size: (H, W) for model input
        temporal_stride: number of frames between t and t+1 for flow
        half:        use FP16 for speed
        conf_threshold: binary mask threshold
    """

    def __init__(
        self,
        model:           SPECTRA,
        device:          torch.device,
        target_size:     Tuple[int, int] = (512, 512),
        temporal_stride: int = 2,
        half:            bool = False,
        conf_threshold:  float = 0.5,
    ):
        self.model           = model.eval()
        self.device          = device
        self.target_size     = target_size
        self.temporal_stride = temporal_stride
        self.half            = half and device.type == "cuda"
        self.conf_threshold  = conf_threshold

        if self.half:
            self.model = self.model.half()

        # Rolling buffer of preprocessed tensors
        self._buffer: Deque[torch.Tensor] = collections.deque(
            maxlen=temporal_stride + 1
        )
        self._frame_count = 0

        # FPS tracking
        self._fps_times: Deque[float] = collections.deque(maxlen=30)

    def _push_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess and push to buffer, return current tensor."""
        t = preprocess_frame(frame_bgr, self.target_size, self.device)
        if self.half:
            t = t.half()
        self._buffer.append(t)
        return t

    @torch.no_grad()
    def process_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Process one video frame.

        Returns:
            vis_frame: (H, W, 3) BGR visualisation frame
            stats:     dict with fps, transparency_pct, latency_ms
        """
        t0 = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]

        frame_t = self._push_frame(frame_bgr)

        # For the first few frames we don't have a t-1 yet — use self as pair
        if len(self._buffer) < self.temporal_stride + 1:
            frame_t1 = frame_t
        else:
            frame_t1 = self._buffer[0]   # oldest frame in buffer

        with torch.cuda.amp.autocast(enabled=self.half):
            outputs = self.model(
                image=frame_t,
                image_t1=frame_t1,
                return_intermediates=True,
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        # FPS
        self._fps_times.append(time.perf_counter())
        if len(self._fps_times) >= 2:
            fps = len(self._fps_times) / (self._fps_times[-1] - self._fps_times[0] + 1e-8)
        else:
            fps = 0.0

        # Postprocess
        prob_map = postprocess_prob(outputs["seg_prob"], (orig_h, orig_w))
        mask     = (prob_map > self.conf_threshold).astype(np.uint8)

        # Visualise
        vis = overlay_mask_on_image(frame_bgr, mask, prob_map, alpha=0.45)

        transparency_pct = float(mask.mean() * 100)

        # HUD overlay
        stats = {
            "fps":               round(fps, 1),
            "latency_ms":        round(latency_ms, 1),
            "transparency_pct":  round(transparency_pct, 1),
            "frame":             self._frame_count,
        }
        vis = draw_hud(vis, stats)
        self._frame_count += 1

        return vis, stats, {
            "prob_map":      prob_map,
            "ofcv_map":      outputs.get("ofcv_map"),
            "brf_map":       outputs.get("brf_map"),
            "residual_map":  outputs.get("residual_map"),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_video(
    config_path:     str = "configs/config.yaml",
    checkpoint_path: Optional[str] = None,
    source:          str = "webcam",     # "webcam" | path to video file
    save:            bool = False,
    save_path:       str  = "outputs/spectra_output.mp4",
    half:            bool = True,
    conf_threshold:  float = 0.5,
    show_physics:    bool = True,        # show OFCV/BRF side panels
):
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W   = cfg.data.image_size

    logger.info(f"Device: {device} | Half: {half} | Source: {source}")

    # Load model
    model = SPECTRA(cfg, use_gnn=False).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    model.eval()

    runner = SPECTRAVideoRunner(
        model=model,
        device=device,
        target_size=(H, W),
        temporal_stride=cfg.flow.temporal_stride,
        half=half,
        conf_threshold=conf_threshold,
    )

    # Open video source
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency for webcam

    # Video writer
    writer = None
    if save:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        w_out   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_out   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer  = cv2.VideoWriter(save_path, fourcc, fps_out, (w_out * 2 if show_physics else w_out, h_out))

    logger.info("Press 'q' to quit, 's' to save snapshot, 'p' to toggle physics panel")

    show_panel = show_physics
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream.")
            break

        vis, stats, physics = runner.process_frame(frame)

        if show_panel:
            # Side-by-side: main vis | physics heatmap
            from inference.vis_utils import make_physics_panel
            panel = make_physics_panel(
                frame,
                physics["prob_map"],
                physics["ofcv_map"],
                physics["brf_map"],
                height=frame.shape[0],
            )
            display = np.concatenate([vis, panel], axis=1)
        else:
            display = vis

        if writer is not None:
            writer.write(display)

        cv2.imshow("SPECTRA — Transparent Object Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap_path = f"outputs/snapshot_{runner._frame_count:06d}.jpg"
            Path("outputs").mkdir(exist_ok=True)
            cv2.imwrite(snap_path, display)
            logger.info(f"Snapshot saved: {snap_path}")
        elif key == ord("p"):
            show_panel = not show_panel

    cap.release()
    if writer:
        writer.release()
        logger.info(f"Video saved: {save_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPECTRA video inference")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    parser.add_argument("--source",     default="webcam",     help="'webcam' or path to video")
    parser.add_argument("--save",       action="store_true",  help="Save output video")
    parser.add_argument("--save-path",  default="outputs/spectra_output.mp4")
    parser.add_argument("--half",       action="store_true",  help="FP16 inference")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--no-physics", action="store_true",  help="Hide physics side panel")
    args = parser.parse_args()

    run_video(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        source=args.source,
        save=args.save,
        save_path=args.save_path,
        half=args.half,
        conf_threshold=args.threshold,
        show_physics=not args.no_physics,
    )
