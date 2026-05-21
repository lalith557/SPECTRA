"""
spectra/graph/superpixel.py
Converts images + feature maps into PyTorch Geometric Data objects.
Each node = one SLIC superpixel region.
Edges = Region Adjacency Graph (RAG).
"""
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from skimage.segmentation import slic
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# SLIC → RAG → PyG Data
# ---------------------------------------------------------------------------

def compute_slic_segments(
    image_np: np.ndarray,         # (H, W, 3) uint8 or float [0,1]
    n_segments: int = 512,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Run SLIC superpixel segmentation.

    Returns:
        segments: (H, W) int array, labels 0..n_actual-1
    """
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)

    segments = slic(
        image_np,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=2,
    )
    return segments


def build_rag_edges(segments: np.ndarray) -> np.ndarray:
    """
    Build Region Adjacency Graph edges from superpixel labels.
    Two regions are adjacent if they share at least one pixel boundary.

    Returns:
        edges: (2, E) int array of (src, dst) pairs (undirected, both directions)
    """
    H, W = segments.shape
    edge_set = set()

    for y in range(H - 1):
        for x in range(W - 1):
            c = segments[y, x]
            r = segments[y, x + 1]   # right neighbour
            d = segments[y + 1, x]   # down neighbour
            if c != r:
                edge_set.add((min(c, r), max(c, r)))
            if c != d:
                edge_set.add((min(c, d), max(c, d)))

    if len(edge_set) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    edges = np.array(list(edge_set), dtype=np.int64).T  # (2, E)
    # Add reverse edges for undirected graph
    edges = np.concatenate([edges, edges[[1, 0]]], axis=1)  # (2, 2E)
    return edges


def pool_features_to_superpixels(
    features: Tensor,     # (C, H, W) — e.g. DINOv2 patch tokens upsampled
    segments: np.ndarray, # (H, W) int
    n_nodes:  int,
) -> Tensor:
    """
    Mean-pool feature map values within each superpixel.

    Returns:
        node_feats: (n_nodes, C)
    """
    C, H, W = features.shape
    device   = features.device
    segs_t   = torch.from_numpy(segments.astype(np.int64)).to(device)  # (H, W)

    node_feats = torch.zeros(n_nodes, C, device=device, dtype=features.dtype)
    counts     = torch.zeros(n_nodes, device=device, dtype=torch.float32)

    # Flatten spatial dims
    flat_feats = features.view(C, -1).T          # (H*W, C)
    flat_segs  = segs_t.view(-1)                 # (H*W,)

    node_feats.scatter_add_(0, flat_segs.unsqueeze(1).expand(-1, C), flat_feats)
    counts.scatter_add_(0, flat_segs, torch.ones(H * W, device=device))

    # Avoid division by zero
    counts = counts.clamp(min=1.0).unsqueeze(1)  # (n_nodes, 1)
    return node_feats / counts                    # (n_nodes, C)


def compute_node_centroids(segments: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Compute normalised (x, y) centroid for each superpixel node.

    Returns:
        centroids: (n_nodes, 2) in [0, 1]
    """
    H, W = segments.shape
    sums   = np.zeros((n_nodes, 2), dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.float64)

    for y in range(H):
        for x in range(W):
            lbl = segments[y, x]
            sums[lbl, 0] += x / (W - 1)
            sums[lbl, 1] += y / (H - 1)
            counts[lbl]  += 1

    counts = np.maximum(counts, 1)[:, None]
    return sums / counts   # (n_nodes, 2) in [0, 1]


# ---------------------------------------------------------------------------
# Node feature builder
# ---------------------------------------------------------------------------

def encode_node_features(
    dino_features:   Tensor,     # (C, h, w) patch tokens (spatial resolution of ViT)
    ofcv_map:        Tensor,     # (1, H, W)
    brf_map:         Tensor,     # (1, H, W)
    image_tensor:    Tensor,     # (3, H, W) original image (for colour stats)
    segments:        np.ndarray, # (H, W)
    n_nodes:         int,
    feat_proj_dim:   int = 256,
    feat_proj:       torch.nn.Module = None,
) -> Tensor:
    """
    Assemble per-node feature vector from all modalities.

    Node feature composition:
      [projected_dino (feat_proj_dim)] + [ofcv_score (1)] + [brf_energy (1)]
      + [mean_rgb (3)] + [centroid_xy (2)]
    Total = feat_proj_dim + 7
    """
    device = dino_features.device
    _, H, W = image_tensor.shape

    # Upsample DINOv2 tokens to match image spatial resolution
    dino_up = torch.nn.functional.interpolate(
        dino_features.unsqueeze(0),   # add batch dim
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)   # (C, H, W)

    # Project DINOv2 dim if projector provided, else truncate/pad
    if feat_proj is not None:
        with torch.no_grad():
            dino_proj = feat_proj(dino_up.unsqueeze(0)).squeeze(0)  # (proj_dim, H, W)
    else:
        dino_proj = dino_up[:feat_proj_dim]   # simple truncation (fallback)

    # Pool each modality to superpixels
    dino_pooled   = pool_features_to_superpixels(dino_proj,              segments, n_nodes)
    ofcv_pooled   = pool_features_to_superpixels(ofcv_map.squeeze(0).unsqueeze(0),  segments, n_nodes)
    brf_pooled    = pool_features_to_superpixels(brf_map.squeeze(0).unsqueeze(0),   segments, n_nodes)
    rgb_pooled    = pool_features_to_superpixels(image_tensor,           segments, n_nodes)

    # Centroid (computed on CPU, then moved to device)
    centroids_np  = compute_node_centroids(segments, n_nodes)
    centroids     = torch.from_numpy(centroids_np).float().to(device)

    # Concatenate all node features
    node_feats = torch.cat([
        dino_pooled,     # (N, proj_dim)
        ofcv_pooled,     # (N, 1)
        brf_pooled,      # (N, 1)
        rgb_pooled,      # (N, 3)
        centroids,       # (N, 2)
    ], dim=-1)           # (N, proj_dim + 7)

    return node_feats


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

class SuperpixelGraphBuilder(torch.nn.Module):
    """
    Converts a batch of images + feature maps into a batched PyG graph.

    Args:
        n_segments:    target number of SLIC superpixels per image
        compactness:   SLIC compactness (higher = more square segments)
        feat_proj_dim: output dim for DINOv2 feature projection
    """

    def __init__(
        self,
        n_segments:   int = 512,
        compactness:  float = 10.0,
        feat_proj_dim: int = 256,
        dino_dim:     int = 1024,
    ):
        super().__init__()
        self.n_segments    = n_segments
        self.compactness   = compactness
        self.feat_proj_dim = feat_proj_dim

        # Small conv to project DINOv2 spatial features to feat_proj_dim
        self.feat_proj = torch.nn.Conv2d(
            dino_dim, feat_proj_dim, kernel_size=1, bias=False
        )

    def build_single_graph(
        self,
        image_t:      Tensor,   # (3, H, W)
        dino_t:       Tensor,   # (C, h, w)  patch tokens
        ofcv_t:       Tensor,   # (1, H, W)
        brf_t:        Tensor,   # (1, H, W)
        ofcv_node_scores: Tensor = None,  # pre-pooled ofcv (N,) if available
        flow_cons_t:  Tensor = None,       # (1, H, W) flow consistency
    ) -> Data:
        device = image_t.device
        H, W   = image_t.shape[-2:]

        # Image to numpy for SLIC
        img_np = image_t.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        segments  = compute_slic_segments(img_np, self.n_segments, self.compactness)
        n_nodes   = segments.max() + 1
        edges_np  = build_rag_edges(segments)

        # Node features
        node_feats = encode_node_features(
            dino_features=dino_t,
            ofcv_map=ofcv_t,
            brf_map=brf_t,
            image_tensor=image_t,
            segments=segments,
            n_nodes=n_nodes,
            feat_proj_dim=self.feat_proj_dim,
            feat_proj=self._spatial_feat_proj,
        )   # (N, feat_proj_dim + 7)

        edge_index = torch.from_numpy(edges_np).long().to(device)  # (2, 2E)

        # Physics edge weights
        if edge_index.numel() > 0:
            # Pool OFCV and flow consistency to superpixels for edge weight computation
            ofcv_node = pool_features_to_superpixels(
                ofcv_t, segments, n_nodes
            ).squeeze(-1)   # (N,)

            if flow_cons_t is not None:
                cons_node = pool_features_to_superpixels(
                    flow_cons_t, segments, n_nodes
                ).squeeze(-1)
            else:
                cons_node = torch.ones(n_nodes, device=device)

            from graph.mbp_gnn import compute_physics_edge_weights
            edge_weights = compute_physics_edge_weights(
                ofcv_scores=ofcv_node,
                flow_consist=cons_node,
                edge_index=edge_index,
            ).unsqueeze(-1)   # (E, 1)
        else:
            # Isolated graph (shouldn't happen in practice)
            edge_weights = torch.ones(0, 1, device=device)

        # Superpixel-to-pixel membership map (for unprojecting predictions)
        seg_tensor = torch.from_numpy(segments.astype(np.int64)).to(device)  # (H, W)

        return Data(
            x=node_feats.float(),
            edge_index=edge_index,
            edge_attr=edge_weights.float(),
            segments=seg_tensor,
            n_nodes=n_nodes,
        )

    def _spatial_feat_proj(self, dino_spatial: Tensor) -> Tensor:
        """Project (1, C, H, W) → (1, proj_dim, H, W)."""
        return self.feat_proj(dino_spatial)

    def forward(
        self,
        images:   Tensor,    # (B, 3, H, W)
        dino_feats: Tensor,  # (B, C, h, w)
        ofcv_maps:  Tensor,  # (B, 1, H, W)
        brf_maps:   Tensor,  # (B, 1, H, W)
        flow_cons_maps: Tensor = None,  # (B, 1, H, W)
    ) -> "Batch":
        from torch_geometric.data import Batch
        graphs = []
        B = images.shape[0]
        for i in range(B):
            fc = flow_cons_maps[i] if flow_cons_maps is not None else None
            g = self.build_single_graph(
                image_t=images[i],
                dino_t=dino_feats[i],
                ofcv_t=ofcv_maps[i],
                brf_t=brf_maps[i],
                flow_cons_t=fc,
            )
            graphs.append(g)
        return Batch.from_data_list(graphs)


def unproject_to_pixels(
    node_predictions: Tensor,   # (N,) per-node logits or probabilities
    segments_batch:   Tensor,   # (H, W) superpixel label map
) -> Tensor:
    """
    Map per-node predictions back to pixel space.

    Args:
        node_predictions: (N,) — scalar value per superpixel
        segments_batch:   (H, W) — label for each pixel

    Returns:
        pixel_map: (H, W) — prediction value at each pixel
    """
    H, W = segments_batch.shape
    flat_segs = segments_batch.view(-1)   # (H*W,)

    pixel_flat = node_predictions[flat_segs]  # (H*W,)
    return pixel_flat.view(H, W)
