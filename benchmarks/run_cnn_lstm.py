"""
benchmarks/run_cnn_lstm.py
==========================
Train and evaluate CNN, LSTM, and CNN-LSTM rainfall-fusion benchmarks.

Compared with run_cnn.py this script adds:
  1. Channel-wise z-score normalisation fitted from training-fold input grids.
  2. Sequence dataset — each item is a contiguous window of `seq_len`
     timesteps; windows that span a data gap are discarded.
  3. Three model variants selectable via --model {cnn, lstm, cnn_lstm}.

Architecture reference
----------------------
Wu et al. (2020) "A spatiotemporal deep fusion model for merging satellite
and gauge precipitation in China", Journal of Hydrology 584:124664.

Usage
-----
  python benchmarks/run_cnn_lstm.py --model cnn_lstm
  python benchmarks/run_cnn_lstm.py --model lstm
  python benchmarks/run_cnn_lstm.py --model cnn        # normalised CNN baseline
  python benchmarks/run_cnn_lstm.py --model cnn --debug
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import tqdm
import xarray as xr

from src.sampling.main import stratified_spatial_kfold_dual
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_processed_dataset

from benchmarks.models.cnn_lstm import RainfallCNNNorm, RainfallLSTM, RainfallCNNLSTM
from benchmarks.processing.gridify import (
    make_grid_coords,
    precompute_idw_weights,
    apply_idw_weights,
)
# Reuse grid-building helpers and sampling from run_cnn
from benchmarks.run_cnn import (
    _prepare_cml_series,
    _prepare_radar_series,
    sample_field_at_stations,
    weighted_mse,
    _compute_metrics,
)


# ============================================================
# Sequence dataset
# ============================================================

def _build_valid_seqs(timestamps: pd.DatetimeIndex,
                      seq_len: int,
                      max_gap_minutes: int = 15) -> list[int]:
    """
    Return target indices where a contiguous sequence of `seq_len` timesteps
    is available immediately before (and including) that index.

    A window is considered contiguous when every consecutive pair of
    timestamps is within `max_gap_minutes` of each other.  This discards
    sequences that span a data gap.

    Parameters
    ----------
    timestamps      : pd.DatetimeIndex  aligned dataset timestamps
    seq_len         : int               number of timesteps per sequence
    max_gap_minutes : int               maximum allowed gap between consecutive
                                        timestamps within a window (default 15)

    Returns
    -------
    list[int]  valid target indices (each at least seq_len-1 from the start)
    """
    gap_limit = pd.Timedelta(minutes=max_gap_minutes)
    valid = []
    for i in range(seq_len - 1, len(timestamps)):
        window = timestamps[i - seq_len + 1 : i + 1]
        diffs  = window[1:] - window[:-1]
        if all(d <= gap_limit for d in diffs):
            valid.append(i)
    return valid


class SequenceGridDataset(Dataset):
    """
    Dataset where each item is a contiguous sequence of input grids plus the
    target station values at the final (most recent) timestep.

    Parameters
    ----------
    input_grids   : torch.Tensor  [T, C, H, W]
    target_values : torch.Tensor  [T, N]
    valid_indices : list[int]     valid target indices from _build_valid_seqs()
    seq_len       : int
    """

    def __init__(self, input_grids: torch.Tensor,
                 target_values: torch.Tensor,
                 valid_indices: list[int],
                 seq_len: int):
        self.grids   = input_grids
        self.targets = target_values
        self.valid   = valid_indices
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int):
        t   = self.valid[idx]
        seq = self.grids[t - self.seq_len + 1 : t + 1]  # [S, C, H, W]
        return seq, self.targets[t]                       # ([S,C,H,W], [N])


# ============================================================
# Training & evaluation loops
# ============================================================

def _train_epoch(model, loader, optimizer, device,
                 station_lons, station_lats,
                 grid_lons, grid_lats, alpha):
    model.train()
    losses = []
    for seqs, targets in loader:
        seqs    = seqs.to(device)       # [B, S, C, H, W]
        targets = targets.to(device)    # [B, N]

        pred = model(seqs)              # [B, 1, H, W]
        pred_at = sample_field_at_stations(
            pred, station_lons, station_lats, grid_lons, grid_lats
        )                               # [B, N]

        loss = weighted_mse(pred_at, targets, alpha=alpha)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def _eval_epoch(model, loader, device,
                station_lons, station_lats,
                grid_lons, grid_lats, alpha):
    model.eval()
    losses, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for seqs, targets in loader:
            seqs    = seqs.to(device)
            targets = targets.to(device)

            pred    = model(seqs).clamp(min=0.0)
            pred_at = sample_field_at_stations(
                pred, station_lons, station_lats, grid_lons, grid_lats
            )

            losses.append(weighted_mse(pred_at, targets, alpha=alpha).item())
            all_preds.append(pred_at.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if not all_preds:
        return float("nan"), np.array([]), np.array([])
    preds   = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()
    return float(np.mean(losses)) if losses else float("nan"), preds, targets


# ============================================================
# Model factory
# ============================================================

def _build_model(model_type: str, in_channels: int, config: dict) -> nn.Module:
    """Instantiate the requested model using hyper-parameters from config."""
    tp = config.get("temporal_params", {})

    if model_type == "cnn":
        # Match the hidden size used in run_cnn.py
        return RainfallCNNNorm(in_channels=in_channels, hidden=64)

    if model_type == "lstm":
        return RainfallLSTM(
            in_channels=in_channels,
            hidden=tp.get("lstm_hidden", 64),
            num_layers=tp.get("lstm_layers", 1),
        )

    if model_type == "cnn_lstm":
        return RainfallCNNLSTM(
            in_channels=in_channels,
            cnn_hidden=32,
            lstm_hidden=tp.get("lstm_hidden", 64),
            num_layers=tp.get("lstm_layers", 1),
        )

    raise ValueError(f"Unknown model_type '{model_type}'. "
                     "Choose from: cnn, lstm, cnn_lstm")


# ============================================================
# Main entry point
# ============================================================

def train_cnn_lstm(config=None, model_type: str = "cnn_lstm",
                   debug: bool = False):
    """
    Train and evaluate a CNN, LSTM, or CNN-LSTM rainfall fusion model.

    Parameters
    ----------
    config     : dict | None   loaded config; reads config.yaml if None
    model_type : str           'cnn' | 'lstm' | 'cnn_lstm'
    debug      : bool          use 500 timesteps / 1 fold / 3 epochs
    """
    if config is None:
        config = read_config("config.yaml")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config["training_params"]["batch_size"]
    epochs     = config["training_params"]["epochs"]
    patience   = config["training_params"]["early_stop"]
    fold_count = config["training_params"]["fold_count"]
    alpha      = config["training_params"]["weighted_loss_alpha"]
    lr         = float(config["model"]["learning_rate"])
    wd         = float(config["model"]["weight_decay"])
    rain_thr   = 0.5

    tp = config.get("temporal_params", {})
    seq_len         = 1 if model_type == "cnn" else tp.get("window_size", 4)
    max_gap_minutes = tp.get("max_gap_minutes", 15)

    debug_T = 500
    if debug:
        epochs     = 3
        patience   = 2
        fold_count = 1
        batch_size = min(batch_size, 16)

    prefix   = "debug_" if debug else ""
    exp_name = f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}"
    os.makedirs(f"experiments/{exp_name}", exist_ok=True)
    shutil.copy("config.yaml", f"experiments/{exp_name}/config.yaml")
    log_path = f"experiments/{exp_name}/training_log.jsonl"

    def _log(record: dict):
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ---- load raw data ----
    print("Loading data …")
    uptime = config["filters"]["uptime_threshold"]
    start  = config["dataset_parameters"]["start_year"]
    end    = config["dataset_parameters"]["end_year"]

    raingauge_df, mapping_df = load_raingauge_dataset(
        start=start, end=end, uptime_threshold=uptime
    )
    raingauge_df = raingauge_df.resample("15min", closed="left", label="left").mean()
    raingauge_df = raingauge_df[raingauge_df.index.minute % 15 == 0]

    radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")

    cml_nc_path = f"database/{config['dataset_parameters']['cml_folder']}"
    with xr.open_dataset(cml_nc_path, engine="netcdf4") as _ds:
        cml_ts = set(pd.to_datetime(_ds["time"].values).tolist())

    # ---- inner-join timestamps ----
    radar_ts   = set(radar_df["timestamp"].tolist())
    common     = sorted(raingauge_df.index.intersection(radar_ts).intersection(cml_ts))
    timestamps = pd.DatetimeIndex(common)

    raingauge_df = raingauge_df.loc[timestamps]
    radar_df     = radar_df[radar_df["timestamp"].isin(set(timestamps))].copy()

    if debug:
        timestamps   = timestamps[:debug_T]
        raingauge_df = raingauge_df.iloc[:debug_T]
        radar_df     = radar_df[radar_df["timestamp"].isin(set(timestamps))].copy()
        print(f"[DEBUG] {debug_T} timesteps, {fold_count} fold(s), {epochs} epoch(s)")

    T = len(timestamps)
    print(f"Aligned timesteps: {T}   seq_len={seq_len}")

    # ---- build grid ----
    grid_lons, grid_lats = make_grid_coords()
    H, W = len(grid_lats), len(grid_lons)
    print(f"Grid: H={H}  W={W}")

    # ---- precompute fold-independent grids ----
    print("Building CML grid series …")
    t0 = time.time()
    cml_grids = _prepare_cml_series(cml_nc_path, grid_lons, grid_lats, timestamps)
    print(f"  CML done in {time.time()-t0:.1f}s")

    print("Building radar grid series …")
    radar_grids = _prepare_radar_series(radar_df, timestamps, H, W)

    # ---- find valid sequence end-indices (data-gap aware) ----
    valid_all = _build_valid_seqs(timestamps, seq_len, max_gap_minutes)
    print(f"Valid sequence indices: {len(valid_all)} / {T}")

    # ---- spatial folds ----
    split_info = stratified_spatial_kfold_dual(
        mapping_df, seed=config["training_params"]["seed"],
        plot=False, n_splits=fold_count
    )

    fold_metrics = []
    total_start  = time.time()

    for fold_idx in range(fold_count):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} / {fold_count}   model={model_type}")
        print(f"{'='*60}")

        train_ids = split_info[fold_idx]["ml"]["train"]
        val_ids   = split_info[fold_idx]["ml"]["validation"]
        test_ids  = split_info[fold_idx]["ml"]["test"]

        def _get_coords(station_ids):
            rows = mapping_df[mapping_df["id"].isin(station_ids)]
            rows = rows.set_index("id").reindex(station_ids)
            return rows["longitude"].values, rows["latitude"].values

        train_lons, train_lats = _get_coords(train_ids)
        val_lons,   val_lats   = _get_coords(val_ids)
        test_lons,  test_lats  = _get_coords(test_ids)

        # ---- rain gauge grid (train stations only) ----
        print(f"  Building raingauge grid from {len(train_ids)} training stations …")
        rg_weights = precompute_idw_weights(train_lons, train_lats, grid_lons, grid_lats)
        rg_values  = raingauge_df[list(train_ids)].fillna(0).values.astype(np.float32)
        rg_grids   = apply_idw_weights(rg_weights, rg_values, H, W)  # [T, H, W]

        # ---- stack input channels ----
        input_np = np.stack([rg_grids, cml_grids, radar_grids], axis=1)  # [T, 3, H, W]
        input_t  = torch.tensor(input_np, dtype=torch.float32)

        # ---- target tensors ----
        def _targets(ids):
            return torch.tensor(
                raingauge_df[list(ids)].fillna(0).values, dtype=torch.float32
            )  # [T, N]

        train_targets = _targets(train_ids)
        val_targets   = _targets(val_ids)
        test_targets  = _targets(test_ids)

        # ---- sequence datasets ----
        train_ds = SequenceGridDataset(input_t, train_targets, valid_all, seq_len)
        val_ds   = SequenceGridDataset(input_t, val_targets,   valid_all, seq_len)
        test_ds  = SequenceGridDataset(input_t, test_targets,  valid_all, seq_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # ---- build model ----
        model = _build_model(model_type, in_channels=3, config=config).to(device)

        # ---- fit normalisation stats from training inputs ----
        # Use all T input grids (spatial k-fold has no temporal holdout)
        model.norm.fit(input_t)
        print(f"  ChannelNorm stats: mean={model.norm.mean.tolist()}  "
              f"std={model.norm.std.tolist()}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        _log({"type": "model_config", "fold": fold_idx,
              "config": model.config, "seq_len": seq_len,
              "norm_mean": model.norm.mean.tolist(),
              "norm_std":  model.norm.std.tolist(),
              "timestamp": time.time()})

        # ---- training loop with early stopping ----
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        fold_start    = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = _train_epoch(
                model, train_loader, optimizer, device,
                train_lons, train_lats, grid_lons, grid_lats, alpha
            )
            val_loss, _, _ = _eval_epoch(
                model, val_loader, device,
                val_lons, val_lats, grid_lons, grid_lats, alpha
            )

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            _log({"type": "epoch", "fold": fold_idx, "epoch": epoch,
                  "train_loss": train_loss, "val_loss": val_loss,
                  "best_val": best_val_loss, "new_best": improved,
                  "timestamp": time.time()})

            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}"
                  + ("  *" if improved else ""))

            if no_improve >= patience:
                print(f"  Early stop after {no_improve} epochs without improvement.")
                break

        # ---- restore best checkpoint ----
        if best_state is not None:
            model.load_state_dict(best_state)

        ckpt_path = f"experiments/{exp_name}/fold_{fold_idx + 1}_model.pt"
        torch.save({"model_state": model.state_dict(),
                    "model_config": model.config,
                    "seq_len": seq_len,
                    "fold": fold_idx,
                    "norm_mean": model.norm.mean.tolist(),
                    "norm_std":  model.norm.std.tolist()}, ckpt_path)

        # ---- test evaluation ----
        _, test_preds, test_targets_np = _eval_epoch(
            model, test_loader, device,
            test_lons, test_lats, grid_lons, grid_lats, alpha
        )
        metrics   = _compute_metrics(test_preds, test_targets_np, rain_threshold=rain_thr)
        fold_time = time.time() - fold_start

        print(f"\n  Test metrics (fold {fold_idx + 1}):")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"  Fold time: {fold_time:.1f}s")

        _log({"type": "test_metrics", "fold": fold_idx,
              "train_time_sec": fold_time, **metrics, "timestamp": time.time()})
        fold_metrics.append(metrics)

    # ---- aggregate ----
    total_time = time.time() - total_start
    agg     = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    agg_std = {f"{k}_std": float(np.std([m[k] for m in fold_metrics]))
               for k in fold_metrics[0]}

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({model_type}) — mean ± std across folds")
    print(f"{'='*60}")
    for k in fold_metrics[0]:
        print(f"  {k}: {agg[k]:.4f} ± {agg_std[k + '_std']:.4f}")
    print(f"Total time: {total_time:.1f}s")

    summary = {"type": "summary", "model_type": model_type,
               "seq_len": seq_len, "total_time_sec": total_time,
               **agg, **agg_std, "per_fold": fold_metrics,
               "timestamp": time.time()}
    _log(summary)

    results_path = f"experiments/{exp_name}/results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return summary


# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["cnn", "lstm", "cnn_lstm"], default="cnn_lstm",
        help="Model variant to train (default: cnn_lstm)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run with 500 timesteps / 1 fold / 3 epochs for local testing",
    )
    args = parser.parse_args()

    config = read_config("config.yaml")
    train_cnn_lstm(config, model_type=args.model, debug=args.debug)
