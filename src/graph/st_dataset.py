"""
SpatioTemporalDataset
=====================
A PyTorch Dataset that wraps a HeteroData object (features shaped [N, T, F])
and yields (context_window, current_snapshot) pairs for spatio-temporal
rainfall interpolation.

Dense-grid design
-----------------
The dataset assumes the caller has already left-joined all data sources onto
a complete 15-minute time grid (done in train_st.py).  There are therefore
no gaps to worry about — every window of W consecutive positions is
contiguous by construction.

Valid target selection
----------------------
A timestep t is a valid training target if and only if at least one raingauge
node has a non-zero validity flag (feature index 1) at t.  Timesteps where
every raingauge reading is imputed (validity = 0) are excluded because there
is no ground truth to learn from.

The context window [t-W, ..., t-1] may contain imputed steps (validity = 0).
These are kept as-is; the LSTM learns to down-weight them via the validity
flag signal embedded in the feature vector.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

# Raingauge validity flag is at feature index 1 (same as in GaugeGraphNew)
_VALIDITY_IDX = 1


class SpatioTemporalDataset(Dataset):
    """
    Dataset for spatio-temporal rainfall interpolation on a dense 15-min grid.

    Each sample contains:
      - ``x_context``  [N, W, F]  — W preceding timesteps (unmasked, may
                                    include imputed steps with validity=0)
      - ``x``          [N, F]     — current/target timestep features
      - ``y``          [N, 1]     — ground-truth rainfall at current timestep
      - ``mask``                  — which nodes are evaluation targets for
                                    this split (same convention as
                                    HeterogeneousWeatherGraphDatasetInductive)

    Args:
        heterodata  : HeteroData whose node features are shaped ``[N, T, F]``.
        window_size : Number of preceding timesteps to include as context (W).
    """

    def __init__(self, heterodata: HeteroData, window_size: int = 6):
        self.heterodata  = heterodata
        self.window_size = window_size
        self.mask        = heterodata["raingauge"].mask

        self.valid_indices = self._find_valid_indices()

        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No valid target timesteps found with window_size={window_size}.  "
                "Ensure the heterodata was built from a dense 15-min grid and that "
                "at least some raingauge nodes have real data."
            )

    # ------------------------------------------------------------------
    # Valid target discovery
    # ------------------------------------------------------------------

    def _find_valid_indices(self) -> list:
        """
        Return all t in [window_size, T) where at least one raingauge node
        has validity > 0 at timestep t.

        With a dense 15-min grid every window is contiguous, so there is no
        gap check — only the target timestep needs real data.
        """
        # raingauge.x shape: [N, T, F]; validity is at dim-2 index 1
        validity = self.heterodata["raingauge"].x[:, :, _VALIDITY_IDX]  # [N, T]
        T = validity.shape[1]

        valid = [
            t for t in range(self.window_size, T)
            if validity[:, t].sum() > 0
        ]
        return valid

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> HeteroData:
        t       = self.valid_indices[idx]
        ctx_idx = list(range(t - self.window_size, t))

        data = HeteroData()

        # --- Raingauge (primary prediction target) ---
        # x_context: [N, W, F]  — past W timesteps; NOT masked here.
        #   The training loop zeros feature[:2] for the target node at the
        #   *current* timestep only (logic_st.py), so there is no leakage.
        # x        : [N, F]     — current timestep (masking done in logic_st.py)
        # y        : [N, 1]     — ground truth at current timestep
        data["raingauge"].x_context = self.heterodata["raingauge"].x[:, ctx_idx, :]
        data["raingauge"].x         = self.heterodata["raingauge"].x[:, t, :]
        data["raingauge"].y         = self.heterodata["raingauge"].y[:, t, :]
        data["raingauge"].mask      = torch.tensor(self.mask)
        data["raingauge"].num_nodes = self.heterodata["raingauge"].x.shape[0]

        # --- Other node types (radar, cml, …) ---
        for ntype in self.heterodata.node_types:
            if ntype == "raingauge":
                continue
            data[ntype].x_context = self.heterodata[ntype].x[:, ctx_idx, :]
            data[ntype].x         = self.heterodata[ntype].x[:, t, :]
            data[ntype].num_nodes = self.heterodata[ntype].x.shape[0]

        # --- Edges (topology fixed across all timesteps) ---
        for et in self.heterodata.edge_types:
            data[et].edge_index = self.heterodata[et].edge_index
            if hasattr(self.heterodata[et], "edge_attr"):
                data[et].edge_attr = self.heterodata[et].edge_attr

        return data
