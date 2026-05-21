"""
spectra/graph/mbp_gnn.py
Novel Contribution C3 — Material Belief Propagation GNN (MBP-GNN).

A graph neural network where:
  - Nodes   = SLIC superpixels (spatial regions)
  - Edges   = adjacency between neighbouring superpixels
  - Node features = DINOv2 + OFCV score + BRF energy + colour stats + centroid
  - Edge weights  = PHYSICS-derived flow disagreement score (NOT learned attention)
                    This is the key novelty vs standard GAT/GCN

Message passing propagates "material beliefs" (glass / water / plastic / metal)
across the graph. High flow-disagreement edges (likely material boundaries) gate
the propagation — beliefs stay within material regions rather than bleeding across
boundaries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, softmax
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Physics-informed edge weight computation
# ---------------------------------------------------------------------------

def compute_physics_edge_weights(
    ofcv_scores: Tensor,    # (N,) per-node OFCV violation score
    flow_consist: Tensor,   # (N,) per-node flow consistency score
    edge_index:   Tensor,   # (2, E) source → target
    sigma:        float = 0.5,
) -> Tensor:
    """
    Compute physics-derived edge weights for the MBP-GNN.

    Key insight: if two adjacent superpixels have high flow inconsistency
    AND high OFCV scores, the edge between them likely crosses a material
    boundary (transparent ↔ opaque or two different transparent materials).
    Such edges should have HIGH weight — material beliefs should NOT propagate
    freely across them (the glass boundary is a hard barrier).

    Conversely, pairs with consistent flow and low OFCV are likely within
    the same material region — beliefs should propagate freely (low weight
    = less gating).

    w_ij = σ( OFCV_i + OFCV_j - flow_consistency_i - flow_consistency_j )

    Returns:
        weights: (E,) edge weights in (0, 1)
    """
    src, dst = edge_index[0], edge_index[1]

    ofcv_src  = ofcv_scores[src]
    ofcv_dst  = ofcv_scores[dst]
    cons_src  = flow_consist[src]
    cons_dst  = flow_consist[dst]

    # High OFCV + low consistency → high weight (strong material boundary signal)
    raw = (ofcv_src + ofcv_dst) - (cons_src + cons_dst)
    weights = torch.sigmoid(raw / sigma)   # (E,) in (0, 1)
    return weights


# ---------------------------------------------------------------------------
# MBP Convolution layer
# ---------------------------------------------------------------------------

class MBPConv(MessagePassing):
    """
    Single MBP message-passing layer.

    Aggregation:
        m_ij = W_msg · x_j * edge_weight_ij      (physics-gated message)
        h_i  = MLP(concat[x_i, Σ_j m_ij])        (update)

    Args:
        in_channels:  input node feature dim
        out_channels: output node feature dim
        edge_dim:     edge attribute dim (1 for scalar weight + optional features)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        edge_dim:     int = 1,
        dropout:      float = 0.1,
    ):
        super().__init__(aggr="add")  # sum aggregation
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # Message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # Skip connection projection if dims differ
        if in_channels != out_channels:
            self.skip = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x:          Tensor,    # (N, in_channels)
        edge_index: Tensor,    # (2, E)
        edge_attr:  Tensor,    # (E, edge_dim) — physics edge weights
    ) -> Tensor:
        # Add self-loops so each node also aggregates from itself
        edge_index_sl, edge_attr_sl = add_self_loops(
            edge_index,
            edge_attr=edge_attr,
            fill_value=1.0,   # self-loop weight = 1 (no gating)
            num_nodes=x.size(0),
        )

        out = self.propagate(edge_index_sl, x=x, edge_attr=edge_attr_sl)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return out + self.skip(x)   # residual connection

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        x_j: (E, in_channels) — neighbour features
        edge_attr: (E, edge_dim) — physics gate value
        """
        # Concatenate neighbour feature with edge physics weight
        msg_in = torch.cat([x_j, edge_attr], dim=-1)   # (E, in_channels + edge_dim)
        return self.msg_mlp(msg_in)                      # (E, out_channels)


# ---------------------------------------------------------------------------
# Full MBP-GNN
# ---------------------------------------------------------------------------

class MBPGNN(nn.Module):
    """
    Material Belief Propagation GNN.

    Stack of MBPConv layers with residual connections.
    After the final layer, each node gets:
      - A binary transparent/opaque prediction
      - A 4-class material type prediction (glass/water/plastic/specular_metal)

    Args:
        node_in_dim:   input node feature dimensionality
        hidden_dim:    GNN hidden dimension
        n_layers:      number of MBPConv layers
        num_classes:   number of material classes (including background)
        dropout:       dropout probability
    """

    def __init__(
        self,
        node_in_dim: int = 258,    # 256 (proj features) + 1 (OFCV) + 1 (BRF)
        hidden_dim:  int = 256,
        n_layers:    int = 4,
        num_classes: int = 5,      # background + 4 material types
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Stack of MBPConv layers
        self.convs = nn.ModuleList([
            MBPConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=1,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output heads
        self.seg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),   # binary: transparent or not
        )

        self.mat_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x:              Tensor,   # (N, node_in_dim) node features
        edge_index:     Tensor,   # (2, E)
        edge_attr:      Tensor,   # (E, 1) physics edge weights
        batch:          Tensor,   # (N,) batch assignment
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            seg_logits: (N, 1)        binary segmentation logit per node
            mat_logits: (N, num_classes) material class logit per node
        """
        h = self.input_proj(x)

        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)

        seg_logits = self.seg_head(h)   # (N, 1)
        mat_logits = self.mat_head(h)   # (N, num_classes)

        return seg_logits, mat_logits
