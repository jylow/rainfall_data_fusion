"""
run_cnn_saved_weights_inference.py
====================================
Re-run CNN test-set inference from ALREADY-TRAINED checkpoints (no training),
using the updated _compute_metrics() in benchmarks/run_cnn.py which now also
reports POD (Probability of Detection), FAR (False Alarm Ratio) and CSI
(Critical Success Index) alongside the original RMSE/MAE/Pearson r/F1.

Rebuilds the exact same data pipeline, grid construction and fold splits as
benchmarks/run_cnn.py::train_cnn (same join order, same stratified_spatial_kfold_dual
seed, same per-fold IDW raingauge grid), then loads each fold's
fold_{fold}_model.pt and runs evaluation directly instead of training. Since
evaluation is deterministic (model.eval()), this reproduces the original
RMSE/MAE/etc. bit-for-bit while adding the new metrics.

Usage
-----
python run_cnn_saved_weights_inference.py --experiment-dir experiments/20260607_205713_cnn
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_processed_dataset
from src.sampling.main import stratified_spatial_kfold_dual

from benchmarks.models.cnn import RainfallCNN
from benchmarks.processing.gridify import make_grid_coords, precompute_idw_weights, apply_idw_weights
from benchmarks.run_cnn import (
    CNNGridDataset,
    _prepare_cml_series,
    _prepare_radar_series,
    _eval_epoch,
    _compute_metrics,
)


def run_inference(experiment_dir, config_path, rain_threshold=0.5, folds=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = read_config(config_path)

    batch_size = config["training_params"]["batch_size"]
    fold_count = config["training_params"]["fold_count"]
    alpha      = config["training_params"]["weighted_loss_alpha"]

    # ---- load raw data (identical to train_cnn) ----
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

    radar_ts = set(radar_df["timestamp"].tolist())
    common   = sorted(raingauge_df.index.intersection(radar_ts).intersection(cml_ts))
    timestamps = pd.DatetimeIndex(common)

    raingauge_df = raingauge_df.loc[timestamps]
    radar_df     = radar_df[radar_df["timestamp"].isin(set(timestamps))].copy()

    T = len(timestamps)
    print(f"Aligned timesteps: {T}")

    grid_lons, grid_lats = make_grid_coords()
    H, W = len(grid_lats), len(grid_lons)
    print(f"Grid: H={H}, W={W}")

    t0 = time.time()
    print("Building CML grid series …")
    cml_grids = _prepare_cml_series(cml_nc_path, grid_lons, grid_lats, timestamps)
    print(f"  CML grid done in {time.time()-t0:.1f}s")

    print("Building radar grid series …")
    radar_grids = _prepare_radar_series(radar_df, timestamps, H, W)

    split_info = stratified_spatial_kfold_dual(
        mapping_df, seed=config["training_params"]["seed"],
        plot=False, n_splits=fold_count
    )

    requested_folds = folds if folds is not None else list(range(fold_count))
    fold_metrics = []
    total_start = time.time()

    for fold_idx in requested_folds:
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} / {fold_count}")
        print(f"{'='*60}")

        train_ids = split_info[fold_idx]["ml"]["train"]
        test_ids  = split_info[fold_idx]["ml"]["test"]

        def _get_coords(station_ids):
            rows = mapping_df[mapping_df["id"].isin(station_ids)]
            rows = rows.set_index("id").reindex(station_ids)
            return rows["longitude"].values, rows["latitude"].values

        train_lons, train_lats = _get_coords(train_ids)
        test_lons,  test_lats  = _get_coords(test_ids)

        # Raingauge grid (train stations only — matches training-time input)
        print(f"  Building raingauge grid from {len(train_ids)} training stations …")
        rg_weights = precompute_idw_weights(train_lons, train_lats, grid_lons, grid_lats)
        rg_values  = raingauge_df[list(train_ids)].fillna(0).values.astype(np.float32)
        rg_grids   = apply_idw_weights(rg_weights, rg_values, H, W)

        input_np = np.stack([rg_grids, cml_grids, radar_grids], axis=1)  # [T, 3, H, W]
        input_t  = torch.tensor(input_np)

        test_targets = torch.tensor(
            raingauge_df[list(test_ids)].fillna(0).values, dtype=torch.float32
        )
        test_ds     = CNNGridDataset(input_t, test_targets)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        ckpt_path = os.path.join(experiment_dir, f"fold_{fold_idx + 1}_model.pt")
        print(f"  Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = RainfallCNN(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        fold_start = time.time()
        _, test_preds, test_targets_np = _eval_epoch(
            model, test_loader, device,
            test_lons, test_lats, grid_lons, grid_lats, alpha
        )
        metrics = _compute_metrics(test_preds, test_targets_np, rain_threshold=rain_threshold)
        fold_time = time.time() - fold_start

        print(f"\n  Test metrics (fold {fold_idx + 1}):")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"  Inference time: {fold_time:.1f}s")

        fold_metrics.append(metrics)

    total_time = time.time() - total_start
    agg     = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    agg_std = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean ± std across folds)")
    print(f"{'='*60}")
    for k in fold_metrics[0]:
        print(f"  {k}: {agg[k]:.4f} ± {agg_std[k + '_std']:.4f}")
    print(f"Total inference time: {total_time:.1f}s")

    summary = {
        "type": "inference_summary",
        "total_time_sec": total_time,
        **agg,
        **agg_std,
        "per_fold": fold_metrics,
        "timestamp": time.time(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True,
                         help="Path to experiments/<name> containing fold_{fold}_model.pt")
    parser.add_argument("--config", default=None,
                         help="Path to the config.yaml used to train the checkpoints "
                              "(default: <experiment-dir>/config.yaml)")
    parser.add_argument("--rain-threshold", type=float, default=0.5,
                         help="Rainfall threshold (mm) for POD/FAR/CSI/F1")
    parser.add_argument("--output-suffix", default=None,
                         help="If set, write to inference_results_<suffix>.json instead of "
                              "inference_results.json")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                         help="Fold indices (0-based) to run (default: all folds in config)")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.rstrip("/")
    config_path = args.config or os.path.join(experiment_dir, "config.yaml")

    summary = run_inference(
        experiment_dir=experiment_dir,
        config_path=config_path,
        rain_threshold=args.rain_threshold,
        folds=args.folds,
    )

    out_name = "inference_results.json" if not args.output_suffix else f"inference_results_{args.output_suffix}.json"
    out_path = os.path.join(experiment_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
