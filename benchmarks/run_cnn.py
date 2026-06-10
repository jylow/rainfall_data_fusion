"""
benchmarks/run_cnn.py
=====================
Train and evaluate the CNN rainfall-fusion benchmark.

Architecture reference
----------------------
Wu et al. (2020) "A spatiotemporal deep fusion model for merging satellite
and gauge precipitation in China", Journal of Hydrology 584:124664.

Three input channels (all interpolated to the same 0.01° Singapore grid):
  Ch 0 – Rain gauge IDW grid        (training stations only)
  Ch 1 – CML specific-attenuation IDW grid  (all links; CML is auxiliary)
  Ch 2 – Weather radar reflectivity grid

Supervision: weighted MSE at training station locations (same loss as GNN).
Evaluation : RMSE, MAE, F1, Pearson r at held-out test station locations.
"""

import os
import sys

# Ensure imports and relative file paths resolve from the project root,
# regardless of which directory the script is invoked from.
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import tqdm
import xarray as xr

# ---------- project imports ----------
from src.sampling.main import stratified_spatial_kfold_dual
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_processed_dataset
from benchmarks.models.cnn import RainfallCNN
from benchmarks.processing.gridify import (
    make_grid_coords,
    precompute_idw_weights,
    apply_idw_weights,
)


# ============================================================
# Dataset
# ============================================================

class CNNGridDataset(Dataset):
    """One item = one timestep: (input_grid [C,H,W], target_values [N])."""

    def __init__(self, input_grids: torch.Tensor, target_values: torch.Tensor):
        assert input_grids.shape[0] == target_values.shape[0]
        self.grids = input_grids
        self.targets = target_values

    def __len__(self):
        return self.grids.shape[0]

    def __getitem__(self, idx):
        return self.grids[idx], self.targets[idx]


# ============================================================
# Sampling helper (uses torch.nn.functional.grid_sample)
# ============================================================

def sample_field_at_stations(pred_field, station_lons, station_lats,
                              grid_lons, grid_lats):
    """Bilinear sample a predicted [B, 1, H, W] field at station positions.

    Parameters
    ----------
    pred_field    : torch.Tensor, [B, 1, H, W]
    station_lons  : np.ndarray, shape (N,)
    station_lats  : np.ndarray, shape (N,)
    grid_lons     : np.ndarray, shape (W,)  ascending
    grid_lats     : np.ndarray, shape (H,)  descending

    Returns
    -------
    torch.Tensor, shape [B, N]
    """
    W = len(grid_lons)
    H = len(grid_lats)

    # Normalise to [-1, 1] as required by grid_sample (align_corners=True)
    xi = 2.0 * (station_lons - grid_lons[0]) / (grid_lons[-1] - grid_lons[0]) - 1.0
    yi = 2.0 * (grid_lats[0] - station_lats) / (grid_lats[0] - grid_lats[-1]) - 1.0

    # grid_sample grid: [B, N, 1, 2] of (x, y) normalised coords
    grid = np.stack([xi, yi], axis=1).astype(np.float32)  # [N, 2]
    grid_t = torch.from_numpy(grid).to(pred_field.device)
    B = pred_field.shape[0]
    grid_4d = grid_t.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)  # [B, N, 1, 2]

    sampled = F.grid_sample(
        pred_field, grid_4d, mode="bilinear",
        align_corners=True, padding_mode="border"
    )  # [B, 1, N, 1]
    return sampled[:, 0, :, 0]  # [B, N]


# ============================================================
# Weighted MSE loss (matches training/logic_hetero.py)
# ============================================================

def weighted_mse(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.0):
    weights = 1.0 + alpha * torch.log1p(target.clamp(min=0.0))
    return (weights * (pred - target) ** 2).mean()


# ============================================================
# Data preparation helpers
# ============================================================

def _prepare_cml_series(cml_nc_path: str,
                        grid_lons: np.ndarray,
                        grid_lats: np.ndarray,
                        timestamps: pd.DatetimeIndex):
    """Return [T, H, W] float32 CML specific-attenuation grids.

    Dataset dims: (link_id=414, station=2, time=T_cml).
    Averages over both station endpoints then aligns to target timestamps.
    Attenuation = max(0, mean_TSL - mean_RSL) / link_length  [dB/km].
    """
    H, W = len(grid_lats), len(grid_lons)

    ds = xr.open_dataset(cml_nc_path, engine="netcdf4")
    cml_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))

    # [N_links, 2, T_cml] → mean over station axis → [N_links, T_cml] → T [T_cml, N_links]
    tsl = ds["TSL_AVG"].values.astype(np.float32)   # [N_links, 2, T_cml]
    rsl = ds["RSL_AVG"].values.astype(np.float32)

    atten = np.clip(
        tsl.mean(axis=1) - rsl.mean(axis=1), 0, None
    ).T.astype(np.float32)                           # [T_cml, N_links]

    link_ids_arr = ds["link_id"].values              # [N_links]
    mid_lons = ((ds["site_a_longitude"].values + ds["site_b_longitude"].values) / 2)
    mid_lats = ((ds["site_a_latitude"].values  + ds["site_b_latitude"].values)  / 2)
    lk_lengths = np.maximum(ds["length"].values.astype(np.float32), 1e-6)
    ds.close()

    atten /= lk_lengths                              # specific attenuation [T_cml, N_links]

    aligned = (
        pd.DataFrame(atten, index=cml_times, columns=link_ids_arr)
        .reindex(timestamps).fillna(0.0).values.astype(np.float32)
    )
    cml_weights = precompute_idw_weights(mid_lons, mid_lats, grid_lons, grid_lats)
    return apply_idw_weights(cml_weights, aligned, H, W)  # [T, H, W]


def _prepare_radar_series(radar_df: pd.DataFrame,
                          timestamps: pd.DatetimeIndex,
                          H: int, W: int):
    """Return [T, H, W] float32 array aligned to timestamps."""
    radar_indexed = radar_df.set_index("timestamp")["data"].reindex(timestamps)

    grids = []
    for ts in timestamps:
        row = radar_indexed.get(ts)
        if row is None or (isinstance(row, float) and np.isnan(row)):
            grids.append(np.zeros((H, W), dtype=np.float32))
        else:
            data = np.asarray(row, dtype=np.float32)
            if data.shape != (H, W):
                # Rescale if grid dimensions differ from radar native resolution
                from scipy.ndimage import zoom
                data = zoom(data, (H / data.shape[0], W / data.shape[1]), order=1)
            grids.append(np.clip(data, 0, None))

    return np.stack(grids, axis=0)  # [T, H, W]


# ============================================================
# Training & evaluation loops
# ============================================================

def _train_epoch(model, loader, optimizer, device,
                 station_lons, station_lats,
                 grid_lons, grid_lats, alpha):
    model.train()
    losses = []
    for grids, targets in loader:
        grids   = grids.to(device)          # [B, C, H, W]
        targets = targets.to(device)        # [B, N_train]

        pred = model(grids)                 # [B, 1, H, W]
        pred_at = sample_field_at_stations(
            pred, station_lons, station_lats, grid_lons, grid_lats
        )                                   # [B, N_train]

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
    losses = []
    all_preds, all_targets = [], []

    with torch.no_grad():
        for grids, targets in loader:
            grids   = grids.to(device)
            targets = targets.to(device)

            pred = model(grids).clamp(min=0.0)
            pred_at = sample_field_at_stations(
                pred, station_lons, station_lats, grid_lons, grid_lats
            )

            loss = weighted_mse(pred_at, targets, alpha=alpha)
            losses.append(loss.item())
            all_preds.append(pred_at.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds   = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()
    return float(np.mean(losses)), preds, targets


def _compute_metrics(preds, targets, rain_threshold=0.5):
    mask = ~(np.isnan(preds) | np.isnan(targets))
    p, t = preds[mask], targets[mask]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae  = float(np.mean(np.abs(p - t)))
    r, _ = pearsonr(p, t) if len(p) > 1 else (0.0, 1.0)
    f1   = f1_score(
        (t >= rain_threshold).astype(int),
        (p >= rain_threshold).astype(int),
        zero_division=0,
    )
    return {"rmse": rmse, "mae": mae, "pearson_r": float(r), "f1": float(f1)}


# ============================================================
# Main entry point
# ============================================================

def train_cnn(config=None, debug: bool = False):
    """Train the CNN benchmark.

    Parameters
    ----------
    debug : bool
        When True, cuts the dataset to 500 timesteps, runs 1 fold for 3 epochs,
        and prefixes the experiment folder with 'debug_'. Use on local machines
        to validate the pipeline end-to-end without loading the full dataset.
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

    # Debug overrides — small enough to fit in ~1 GB RAM on an M1 laptop
    debug_T = 500
    if debug:
        epochs     = 3
        patience   = 2
        fold_count = 1
        batch_size = min(batch_size, 16)

    # ---- experiment folder ----
    prefix   = "debug_" if debug else ""
    exp_name = f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}_cnn"
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

    raingauge_df, mapping_df = load_raingauge_dataset(start=start, end=end,
                                                       uptime_threshold=uptime)
    raingauge_df = raingauge_df.resample("15min", closed="left", label="left").mean()
    raingauge_df = raingauge_df[raingauge_df.index.minute % 15 == 0]

    radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")

    cml_nc_path = f"database/{config['dataset_parameters']['cml_folder']}"
    with xr.open_dataset(cml_nc_path, engine="netcdf4") as _ds:
        cml_ts = set(pd.to_datetime(_ds["time"].values).tolist())

    # ---- inner-join timestamps ----
    radar_ts = set(radar_df["timestamp"].tolist())
    common   = sorted(raingauge_df.index.intersection(radar_ts).intersection(cml_ts))
    timestamps = pd.DatetimeIndex(common)

    raingauge_df = raingauge_df.loc[timestamps]
    radar_df     = radar_df[radar_df["timestamp"].isin(set(timestamps))].copy()

    if debug:
        timestamps   = timestamps[:debug_T]
        raingauge_df = raingauge_df.iloc[:debug_T]
        radar_df     = radar_df[radar_df["timestamp"].isin(set(timestamps))].copy()
        print(f"[DEBUG] Using {debug_T} timesteps, {fold_count} fold(s), {epochs} epoch(s)")

    T = len(timestamps)
    print(f"Aligned timesteps: {T}")

    # ---- build grid coords (matches radar 0.01° resolution) ----
    grid_lons, grid_lats = make_grid_coords()
    H = len(grid_lats)
    W = len(grid_lons)
    print(f"Grid: H={H}, W={W}")

    # ---- precompute CML and radar grids (fold-independent) ----
    t0 = time.time()
    print("Building CML grid series …")
    cml_grids = _prepare_cml_series(cml_nc_path, grid_lons, grid_lats, timestamps)  # [T, H, W]
    print(f"  CML grid done in {time.time()-t0:.1f}s")

    print("Building radar grid series …")
    radar_grids = _prepare_radar_series(radar_df, timestamps, H, W)     # [T, H, W]

    # ---- spatial folds (same splits as GNN) ----
    split_info = stratified_spatial_kfold_dual(
        mapping_df, seed=config["training_params"]["seed"],
        plot=False, n_splits=fold_count
    )

    fold_metrics = []
    total_start  = time.time()

    for fold_idx in range(fold_count):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} / {fold_count}")
        print(f"{'='*60}")

        train_ids = split_info[fold_idx]["ml"]["train"]
        val_ids   = split_info[fold_idx]["ml"]["validation"]
        test_ids  = split_info[fold_idx]["ml"]["test"]

        # Station coordinates
        def _get_coords(station_ids):
            rows = mapping_df[mapping_df["id"].isin(station_ids)]
            rows = rows.set_index("id").reindex(station_ids)
            return rows["longitude"].values, rows["latitude"].values

        train_lons, train_lats = _get_coords(train_ids)
        val_lons,   val_lats   = _get_coords(val_ids)
        test_lons,  test_lats  = _get_coords(test_ids)

        # Raingauge grid (train stations only — test/val stations never in input)
        print(f"  Building raingauge grid from {len(train_ids)} training stations …")
        rg_weights = precompute_idw_weights(train_lons, train_lats, grid_lons, grid_lats)
        rg_values  = raingauge_df[list(train_ids)].fillna(0).values.astype(np.float32)
        rg_grids   = apply_idw_weights(rg_weights, rg_values, H, W)  # [T, H, W]

        # Stack into [T, 3, H, W]
        input_np = np.stack([rg_grids, cml_grids, radar_grids], axis=1)  # [T, 3, H, W]
        input_t  = torch.tensor(input_np)

        # Target tensors per split
        def _targets(station_ids):
            return torch.tensor(
                raingauge_df[list(station_ids)].fillna(0).values, dtype=torch.float32
            )  # [T, N]

        train_targets = _targets(train_ids)
        val_targets   = _targets(val_ids)
        test_targets  = _targets(test_ids)

        # Datasets / loaders
        train_ds = CNNGridDataset(input_t, train_targets)
        val_ds   = CNNGridDataset(input_t, val_targets)
        test_ds  = CNNGridDataset(input_t, test_targets)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # Model + optimiser
        model = RainfallCNN(in_channels=3, hidden=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Log model config
        _log({"type": "model_config", "fold": fold_idx,
              "config": model.config, "timestamp": time.time()})

        # Training loop with early stopping
        best_val_loss  = float("inf")
        best_state     = None
        no_improve     = 0
        fold_start     = time.time()

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
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save checkpoint
        ckpt_path = f"experiments/{exp_name}/fold_{fold_idx + 1}_model.pt"
        torch.save({"model_state": model.state_dict(),
                    "model_config": model.config,
                    "fold": fold_idx}, ckpt_path)

        # Evaluate on test set
        _, test_preds, test_targets_np = _eval_epoch(
            model, test_loader, device,
            test_lons, test_lats, grid_lons, grid_lats, alpha
        )
        metrics = _compute_metrics(test_preds, test_targets_np, rain_threshold=rain_thr)
        fold_time = time.time() - fold_start

        print(f"\n  Test metrics (fold {fold_idx + 1}):")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"  Fold time: {fold_time:.1f}s")

        _log({"type": "test_metrics", "fold": fold_idx,
              "train_time_sec": fold_time, **metrics, "timestamp": time.time()})

        fold_metrics.append(metrics)

    # ---- aggregate results ----
    total_time = time.time() - total_start
    agg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    agg_std = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean ± std across folds)")
    print(f"{'='*60}")
    for k in fold_metrics[0]:
        print(f"  {k}: {agg[k]:.4f} ± {agg_std[k + '_std']:.4f}")
    print(f"Total time: {total_time:.1f}s")

    summary = {
        "type": "summary",
        "total_time_sec": total_time,
        **agg,
        **agg_std,
        "per_fold": fold_metrics,
        "timestamp": time.time(),
    }
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
    parser.add_argument("--debug", action="store_true",
                        help="Run with 500 timesteps / 1 fold / 3 epochs for local testing")
    args = parser.parse_args()

    config = read_config("config.yaml")
    train_cnn(config, debug=args.debug)
