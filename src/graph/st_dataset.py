"""
SpatioTemporalDataset
=====================
A PyTorch Dataset that wraps a HeteroData object (features shaped [N, T, F])
and yields (context_window, current_snapshot) pairs for spatio-temporal
rainfall interpolation.

Key design choices
------------------
* The context window contains W *preceding* timesteps for every node.
  The target node's past values are NOT masked — only the current-timestep
  value is zeroed in the training loop (handled in logic_st.py), so there
  is no data leakage.
* Non-contiguous timestamps are handled by building a list of valid target
  indices where the full window [t-W, …, t-1, t] has no gap larger than
  `max_gap_minutes`.  Windows that straddle a gap are silently dropped.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class SpatioTemporalDataset(Dataset):
    """
    Dataset for spatio-temporal rainfall interpolation.

    Each sample consists of:
      - ``x_context``  [N, W, F]  — W preceding timesteps (unmasked)
      - ``x``          [N, F]     — current/target timestep
      - ``y``          [N, 1]     — ground-truth rainfall at current timestep

    The train/val/test split is purely over *nodes* (graph topology), so all
    T timesteps are available in every split.  The ``mask`` attribute identifies
    which nodes are the evaluation targets for that split (same convention as
    :class:`HeterogeneousWeatherGraphDatasetInductive`).

    Args:
        heterodata: HeteroData whose node features are shaped ``[N, T, F]``.
        timestamps: Array-like of length T giving the wall-clock timestamp for
            each column in the T dimension.  Used to detect gaps.
        window_size: Number of preceding timesteps to include as context (W).
        max_gap_minutes: Maximum allowed gap between two consecutive timesteps
            inside a valid window.  Windows containing a larger gap are dropped.
    """

    def __init__(
        self,
        heterodata: HeteroData,
        timestamps,
        window_size: int = 6,
        max_gap_minutes: int = 10,
    ):
        self.heterodata = heterodata
        self.timestamps = pd.to_datetime(timestamps)
        self.window_size = window_size

        # Preserve the split mask exactly as the original dataset does
        self.mask = heterodata["raingauge"].mask

        self.valid_indices = self._find_valid_windows(max_gap_minutes)

        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No valid windows found with window_size={window_size} and "
                f"max_gap_minutes={max_gap_minutes}.  "
                "Check your timestamps or relax max_gap_minutes."
            )

    # ------------------------------------------------------------------
    # Window validity
    # ------------------------------------------------------------------

    def _find_valid_windows(self, max_gap_minutes: int) -> list:
        """Return list of target indices t where window [t-W, …, t] is gap-free."""
        T = len(self.timestamps)
        max_gap = pd.Timedelta(minutes=max_gap_minutes)
        valid = []
        for t in range(self.window_size, T):
            window_ts = self.timestamps[t - self.window_size : t + 1]
            diffs = pd.Series(window_ts.values).diff().dropna()
            if (diffs <= max_gap).all():
                valid.append(t)
        return valid

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> HeteroData:
        t = self.valid_indices[idx]
        ctx_idx = list(range(t - self.window_size, t))

        data = HeteroData()

        # --- Raingauge (primary prediction target) ---
        # x_context: [N, W, F]  — past W timesteps, unmasked
        # x:         [N, F]     — current timestep (masking done in training loop)
        # y:         [N, 1]     — ground truth at current timestep
        data["raingauge"].x_context = self.heterodata["raingauge"].x[:, ctx_idx, :]
        data["raingauge"].x = self.heterodata["raingauge"].x[:, t, :]
        data["raingauge"].y = self.heterodata["raingauge"].y[:, t, :]
        data["raingauge"].mask = torch.tensor(self.mask)
        data["raingauge"].num_nodes = self.heterodata["raingauge"].x.shape[0]

        # --- Other node types (radar, cml, …) ---
        for ntype in self.heterodata.node_types:
            if ntype == "raingauge":
                continue
            data[ntype].x_context = self.heterodata[ntype].x[:, ctx_idx, :]
            data[ntype].x = self.heterodata[ntype].x[:, t, :]
            data[ntype].num_nodes = self.heterodata[ntype].x.shape[0]

        # --- Edges (topology is fixed across all timesteps) ---
        for et in self.heterodata.edge_types:
            data[et].edge_index = self.heterodata[et].edge_index
            if hasattr(self.heterodata[et], "edge_attr"):
                data[et].edge_attr = self.heterodata[et].edge_attr

        return data
