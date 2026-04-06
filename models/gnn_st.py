"""
GNNInductiveHeteroST
====================
Spatio-Temporal Heterogeneous GNN that combines a per-node-type LSTM for
temporal encoding with a stack of HeteroConv GNN layers for spatial
message-passing.

Architecture (per forward pass)
--------------------------------
1. LSTM encoder  — processes the context window [B*N, W, F] for each node
   type and produces a temporal embedding [B*N, lstm_hidden].
2. Feature fusion — concatenate current-timestep features [B*N, F] with the
   temporal embedding → [B*N, F + lstm_hidden], then project to
   hidden_channels via a per-node-type Linear layer.
3. HeteroConv GNN layers — num_layers rounds of GraphConv message-passing
   over the heterogeneous graph.
4. Output head  — Linear + Softplus → predicted rainfall [B*N, out_channels].

Notes
-----
* During training the current-timestep features of the *target* node are
  zeroed (leave-one-out masking) by the training loop **before** calling
  forward().  The context window is NOT masked, which is the source of the
  temporal signal.
* GraphConv is used (same conv as GNNInductiveHetero) so the two models are
  directly comparable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv, GATv2Conv


class GNNInductiveHeteroST(nn.Module):
    """
    Spatio-Temporal HGNN: LSTM temporal encoder + HeteroConv GNN.

    Args:
        in_channels_dict: Mapping from node type to its raw feature dimension
            at a single timestep (same values as used for GNNInductiveHetero).
        hidden_channels: Hidden dimension shared by all GNN layers.
        out_channels: Output dimension (1 for single-value regression).
        num_layers: Number of HeteroConv layers.
        edge_types: List of edge-type tuples, e.g.
            ``[('raingauge','connects','raingauge'), ...]``.
        window_size: Number of past timesteps in the context window (W).
            Stored for reference but not used in the forward pass directly.
        lstm_hidden: Hidden state dimension of each LSTM encoder.
        lstm_layers: Number of stacked LSTM layers.
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        edge_types: list,
        window_size: int,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        cml_edge_dim: int | None = None,
    ):
        super().__init__()

        self.window_size = window_size
        self.lstm_hidden = lstm_hidden
        self.node_types = list(in_channels_dict.keys())
        self.dropout = nn.Dropout(p=dropout)
        self.cml_edge_dim = cml_edge_dim

        # ------------------------------------------------------------------
        # 1. LSTM encoders — one per node type
        # ------------------------------------------------------------------
        self.lstm_encoders = nn.ModuleDict(
            {
                ntype: nn.LSTM(
                    input_size=in_ch,
                    hidden_size=lstm_hidden,
                    num_layers=lstm_layers,
                    batch_first=True,  # input: [batch, seq_len, features]
                )
                for ntype, in_ch in in_channels_dict.items()
            }
        )

        # ------------------------------------------------------------------
        # 2. Input projection: (in_ch + lstm_hidden) → hidden_channels
        # ------------------------------------------------------------------
        self.input_proj = nn.ModuleDict(
            {
                ntype: nn.Linear(in_ch + lstm_hidden, hidden_channels)
                for ntype, in_ch in in_channels_dict.items()
            }
        )

        # ------------------------------------------------------------------
        # 3. GNN layers — operate at hidden_channels after projection
        # ------------------------------------------------------------------
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in edge_types:
                _, rel, _ = et
                if rel == 'cml_link' and cml_edge_dim is not None:
                    conv_dict[et] = GATv2Conv(
                        (-1, -1), hidden_channels,
                        heads=1, concat=False,
                        edge_dim=cml_edge_dim,
                        add_self_loops=False,
                    )
                else:
                    conv_dict[et] = GraphConv((-1, -1), hidden_channels)
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # ------------------------------------------------------------------
        # 4. Output head
        # ------------------------------------------------------------------
        self.lin = nn.Linear(hidden_channels, out_channels)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_dict, x_context_dict, edge_index_dict, edge_attr_dict):
        """
        Args:
            x_dict: ``{node_type: Tensor[B*N, F]}``
                Current-timestep node features.  For the target node the data
                features (rain value + validity flag) have been zeroed by the
                training loop; LPE columns are preserved.
            x_context_dict: ``{node_type: Tensor[B*N, W, F]}``
                Context window of W preceding timesteps.  These are *never*
                masked — the target node's own history is intentionally visible
                to the LSTM.
            edge_index_dict: ``{edge_type: LongTensor[2, E]}``
            edge_attr_dict:  ``{edge_type: Tensor[E]}``

        Returns:
            ``{'raingauge': Tensor[B*N, out_channels]}`` — softplus-activated
            rainfall predictions.
        """
        h_dict = {}

        for ntype, x_cur in x_dict.items():
            # ---- Temporal encoding ----
            if ntype in self.lstm_encoders and ntype in x_context_dict:
                x_ctx = x_context_dict[ntype]  # [B*N, W, F]
                # LSTM returns (output, (h_n, c_n))
                # h_n shape: [num_layers, B*N, lstm_hidden]
                _, (h_n, _) = self.lstm_encoders[ntype](x_ctx)
                temp_emb = h_n[-1]  # take top layer: [B*N, lstm_hidden]
            else:
                temp_emb = torch.zeros(
                    x_cur.shape[0], self.lstm_hidden, device=x_cur.device
                )

            # ---- Feature fusion & projection ----
            x_aug = torch.cat([x_cur, temp_emb], dim=-1)  # [B*N, F + lstm_hidden]
            h_dict[ntype] = self.dropout(F.relu(self.input_proj[ntype](x_aug)))  # [B*N, hidden_channels]

        # ---- GNN message-passing ----
        # GraphConv takes scalar edge_weight; GATv2Conv takes vector edge_attr.
        # Split so each conv type receives the correct kwarg.
        for conv in self.convs:
            scalar_weights = {
                et: ea for et, ea in edge_attr_dict.items()
                if et[1] != 'cml_link' or self.cml_edge_dim is None
            }
            vector_attrs = {
                et: ea for et, ea in edge_attr_dict.items()
                if et[1] == 'cml_link' and self.cml_edge_dim is not None
            }
            h_dict = conv(
                h_dict, edge_index_dict,
                edge_weight_dict=scalar_weights,
                edge_attr_dict=vector_attrs,
            )
            h_dict = {k: self.dropout(F.relu(v)) for k, v in h_dict.items()}

        # No output activation — allows gradients to flow freely during training.
        # Clamp to >= 0 is applied in the validation/test loops.
        return {"raingauge": self.lin(h_dict["raingauge"])}
