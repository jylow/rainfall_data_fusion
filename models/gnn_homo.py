"""
models/gnn_homo.py
==================
Homogeneous GNN baseline for multi-source rainfall fusion.

Purpose
-------
GNNHomoBaseline is an intentionally weak baseline that quantifies how much
the heterogeneous graph design of GNNInductiveHetero helps.  It removes
the type-aware message-passing by merging all node types (raingauge, CML,
radar) into a single homogeneous node set with zero-padded features:

  raingauge node : [rg_feat_0 .. rg_feat_F  | 0 .. 0        | 0 .. 0   ]
  CML node       : [0 .. 0                   | cml_feat_0..F | 0 .. 0   ]
  radar node     : [0 .. 0                   | 0 .. 0        | radar_val]

All edges (within-type and cross-type) are then merged into a single
edge_index over the unified node space, and a standard GraphConv stack
propagates messages without any type distinction.

Interface
---------
Identical to GNNInductiveHetero — takes (x_dict, edge_index_dict,
edge_attr_dict) and returns {'raingauge': predictions} — so the training
loops in training/logic_hetero.py work without modification.

One extra step: call model.set_graph_sizes(heterodata) once before each
train / val / test phase, because the raingauge node count differs across
splits within a fold.

Node index remapping
---------------------
For a PyG batch of B graphs, each with N_rg raingauge, N_cml CML and
N_radar radar nodes:

  Hetero batch space (per node-type):
    raingauge: indices 0 .. B*N_rg-1
    cml:       indices 0 .. B*N_cml-1
    radar:     indices 0 .. B*N_radar-1

  Unified homo batch space:
    N_total = N_rg + N_cml + N_radar   (nodes per graph)
    graph b occupies indices [b*N_total, (b+1)*N_total)
      raingauge: b*N_total + 0           ..  b*N_total + N_rg-1
      cml:       b*N_total + N_rg        ..  b*N_total + N_rg+N_cml-1
      radar:     b*N_total + N_rg+N_cml  ..  b*N_total + N_total-1

  For a hetero node index h_i of type t (N_t nodes per graph):
    graph_b  = h_i // N_t
    local    = h_i %  N_t
    homo_idx = graph_b * N_total + node_offset[t] + local
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, Linear


class GNNHomoBaseline(nn.Module):
    """
    Homogeneous GNN baseline — all node types share one zero-padded feature
    vector and one set of graph-convolution weights.

    Parameters
    ----------
    in_channels_dict : dict   {node_type: num_input_features}
                              Same format as GNNInductiveHetero.
                              Missing types default to 0 features.
    hidden_channels  : int
    out_channels     : int
    num_layers       : int
    edge_types       : ignored — kept for API parity with GNNInductiveHetero
    dropout          : float  applied after each conv layer's ReLU
    """

    # Fixed node-type ordering that determines the layout of the zero-padded
    # feature vector and the node-index block order within each graph.
    NODE_ORDER = ('raingauge', 'cml', 'radar')

    def __init__(self, in_channels_dict: dict, hidden_channels: int,
                 out_channels: int, num_layers: int,
                 edge_types=None, dropout: float = 0.0):
        super().__init__()

        self.in_channels_dict = {k: in_channels_dict.get(k, 0)
                                 for k in self.NODE_ORDER}
        D = sum(self.in_channels_dict.values())

        self.config = dict(
            in_channels_dict=in_channels_dict,
            total_in_channels=D,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(p=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(D, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))

        self.lin = Linear(hidden_channels, out_channels)

        # Populated by set_graph_sizes() before each train/val/test phase.
        self._npg: dict | None = None

    # ------------------------------------------------------------------
    # Graph-size registration
    # ------------------------------------------------------------------

    def set_graph_sizes(self, nodes_per_graph: dict) -> None:
        """
        Register the per-type node count for the current graph split.

        Must be called before each train / val / test phase because the
        raingauge count differs between splits (train ≈ 35, val ≈ 43,
        test = 51 for the Singapore dataset).

        Parameters
        ----------
        nodes_per_graph : dict  {node_type: N_nodes_per_single_graph}
            Typically obtained as:
              {nt: heterodata[nt].x.shape[0]
               for nt in heterodata.node_types}
        """
        self._npg = {k: nodes_per_graph.get(k, 0) for k in self.NODE_ORDER}

    # ------------------------------------------------------------------
    # Hetero → homo conversion
    # ------------------------------------------------------------------

    def _build_homo_graph(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Convert heterogeneous inputs into a single homogeneous graph.

        Returns
        -------
        x_homo         : [B*N_total, D]
        edge_index_homo: [2, E_total]
        edge_weight    : [E_total]
        node_offset    : dict {node_type: offset within one graph block}
        N_rg           : raingauge nodes per graph
        N_total        : total nodes per graph
        B              : batch size (number of graphs in batch)
        """
        npg    = self._npg
        device = next(iter(x_dict.values())).device

        # ---- derive layout constants ----
        N_rg    = npg['raingauge']
        N_total = sum(npg.values())
        B       = x_dict['raingauge'].shape[0] // N_rg if N_rg > 0 else 1

        # Feature slots: where each type's channels start in the D-dim vector
        feat_offset: dict[str, int] = {}
        off = 0
        for nt in self.NODE_ORDER:
            feat_offset[nt] = off
            off += self.in_channels_dict[nt]
        D = off

        # Node slots: where each type's nodes start within one graph block
        node_offset: dict[str, int] = {}
        off = 0
        for nt in self.NODE_ORDER:
            node_offset[nt] = off
            off += npg[nt]

        # ---- build combined node-feature tensor [B*N_total, D] ----
        x_homo = torch.zeros(B * N_total, D, device=device, dtype=torch.float32)

        for nt in self.NODE_ORDER:
            if nt not in x_dict or npg[nt] == 0:
                continue
            N_t  = npg[nt]
            fs   = feat_offset[nt]
            fe   = fs + self.in_channels_dict[nt]
            h    = torch.arange(B * N_t, device=device)
            homo = (h // N_t) * N_total + node_offset[nt] + (h % N_t)
            x_homo[homo, fs:fe] = x_dict[nt].to(dtype=torch.float32)

        # ---- remap all edges into unified index space ----
        all_edges:   list[torch.Tensor] = []
        all_weights: list[torch.Tensor] = []

        for edge_type, ei in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            N_src = npg.get(src_type, 0)
            N_dst = npg.get(dst_type, 0)
            if N_src == 0 or N_dst == 0 or ei.shape[1] == 0:
                continue

            src_h = ei[0]
            dst_h = ei[1]
            homo_src = (src_h // N_src) * N_total + node_offset[src_type] + (src_h % N_src)
            homo_dst = (dst_h // N_dst) * N_total + node_offset[dst_type] + (dst_h % N_dst)
            all_edges.append(torch.stack([homo_src, homo_dst]))

            # Collapse edge attributes to a 1-D scalar weight per edge
            ea = edge_attr_dict.get(edge_type, None)
            if ea is not None and ea.numel() > 0:
                w = ea.float().squeeze(-1) if ea.dim() > 1 else ea.float()
            else:
                w = torch.ones(ei.shape[1], device=device)
            all_weights.append(w)

        edge_index_homo = torch.cat(all_edges,   dim=1)
        edge_weight     = torch.cat(all_weights, dim=0)

        return x_homo, edge_index_homo, edge_weight, node_offset, N_rg, N_total, B

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Parameters
        ----------
        x_dict           : {node_type: [N_t, F_t]}   (per-type node features)
        edge_index_dict  : {edge_type: [2, E]}        (per-type edge indices)
        edge_attr_dict   : {edge_type: [E]}           (per-type edge weights)

        Returns
        -------
        {'raingauge': Tensor[N_rg_total, out_channels]}
        """
        if self._npg is None:
            raise RuntimeError(
                "Call model.set_graph_sizes(heterodata) before the first "
                "forward pass."
            )

        x, edge_index, edge_weight, node_offset, N_rg, N_total, B = \
            self._build_homo_graph(x_dict, edge_index_dict, edge_attr_dict)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.dropout(F.relu(x))

        # Extract raingauge nodes from each graph in the batch
        rg_off = node_offset['raingauge']
        rg_idx = torch.cat([
            torch.arange(b * N_total + rg_off,
                         b * N_total + rg_off + N_rg,
                         device=x.device)
            for b in range(B)
        ])

        return {'raingauge': self.lin(x[rg_idx])}
