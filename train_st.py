"""
train_st.py  —  Spatio-Temporal rainfall interpolation training pipeline.

Dense-grid approach
-------------------
All three data sources (raingauge, radar, CML) are left-joined onto a
complete 15-minute time grid that spans the full dataset period.  Missing
entries are zero-padded and a per-node-per-timestep validity flag (feature
index 1 for raingauge / a dedicated column for CML / a second feature
channel for radar) tells the LSTM which context steps are real vs. imputed.

Key differences vs train_fused.py
-----------------------------------
1. Data loading : left-join on complete 15-min grid instead of inner join.
2. Validity flags: every node type carries explicit 0/1 validity information.
3. Dataset       : SpatioTemporalDataset yields context windows; every
                   timestep where at least one raingauge node is valid is a
                   training target — no windows are dropped due to gaps.
4. Model         : GNNInductiveHeteroST (LSTM + GNN).
5. Training loop : per-node validity check before loss computation so the
                   model is never trained to predict an imputed zero value.
"""

from src.sampling.main import stratified_spatial_kfold_dual  # must be first

import gc

import torch
import os
import shutil
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from torch_geometric.loader import DataLoader as GeometricDataLoader

from src.performance_logger import PerformanceLogger
from models.gnn_st import GNNInductiveHeteroST
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_processed_dataset
from src.cml.utils import load_cml_dataset
from training.logic_st import train_epoch_st, validate_st, test_model_st
from src.graph.cmlgraph import CMLGraph
from src.graph.radargraph import RadarGraph
from src.graph.gaugegraphnew import GaugeGraphNew
from src.graph.st_dataset import SpatioTemporalDataset


# ---------------------------------------------------------------------------
# Normalisation helpers  (identical to train_fused.py)
# ---------------------------------------------------------------------------

def compute_norm_stats(heterodata):
    stats = {}
    for node_type in heterodata.node_types:
        x = heterodata[node_type].x  # [N, T, F]
        mean = x.mean(dim=(0, 1))
        std = x.std(dim=(0, 1)).clamp(min=1e-8)
        stats[node_type] = (mean, std)
    return stats


def apply_norm(heterodata, stats):
    """Apply precomputed stats. Only touches .x, never .y."""
    normed = heterodata.clone()
    for node_type in heterodata.node_types:
        if node_type in stats:
            mean, std = stats[node_type]
            normed[node_type].x = (heterodata[node_type].x - mean) / std
    return normed


# ---------------------------------------------------------------------------
# CML expansion helper
# ---------------------------------------------------------------------------

# Columns that are static per (link_id, station) — used to build the full
# (timestamp × link × station) index.  Any subset that is missing from the
# actual CML dataframe is silently ignored.
_CML_STATIC_COLS = [
    "link_id", "station",
    "site_a_latitude", "site_a_longitude",
    "site_b_latitude", "site_b_longitude",
    "length", "frequency", "polarization",
]
_CML_INDEX_COLS = ["timestamp", "link_id", "station"]


def _expand_cml_to_dense_grid(cml_df: pd.DataFrame, complete_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a long-format CML dataframe to cover every timestamp in
    complete_df for every (link_id, station) pair.

    Missing (timestamp, link, station) rows are zero-padded and receive a
    ``valid = 0`` flag.  Real rows receive ``valid = 1``.

    The ``valid`` column is included as a regular feature so CMLGraph's
    pivot_table picks it up automatically.
    """
    present_static = [c for c in _CML_STATIC_COLS if c in cml_df.columns]
    cml_static = (
        cml_df[present_static]
        .drop_duplicates(["link_id", "station"])
        .reset_index(drop=True)
    )

    # Dynamic feature columns (everything except static non-index cols)
    _static_non_index = [c for c in present_static if c not in ["link_id", "station"]]
    dynamic_cols = [c for c in cml_df.columns if c not in _static_non_index]

    cml_dynamic = cml_df[dynamic_cols].drop_duplicates(_CML_INDEX_COLS)

    # Build complete (timestamp × link × station) index
    complete_cml_index = complete_df.merge(
        cml_static[["link_id", "station"]], how="cross"
    )

    # Left-join dynamic features
    cml_expanded = complete_cml_index.merge(
        cml_dynamic, on=_CML_INDEX_COLS, how="left"
    )

    # Restore static columns (may be NaN after merge due to column overlap)
    cml_expanded = cml_expanded.merge(
        cml_static, on=["link_id", "station"], how="left", suffixes=("_drop", "")
    )
    for col in _static_non_index:
        if f"{col}_drop" in cml_expanded.columns:
            cml_expanded = cml_expanded.drop(columns=[f"{col}_drop"])

    # Validity flag: 1 where real data existed, 0 for padded rows
    # Use the first dynamic feature column (not the index cols) to detect presence
    _check_cols = [c for c in cml_dynamic.columns if c not in _CML_INDEX_COLS]
    if _check_cols:
        cml_expanded["valid"] = cml_expanded[_check_cols[0]].notna().astype(float)
    else:
        cml_expanded["valid"] = 1.0

    return cml_expanded.fillna(0)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_st(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config["training_params"]["batch_size"]
    fold_count = config["training_params"]["fold_count"]
    datasources = config["datasources"]

    tp = config.get("temporal_params", {})
    window_size      = tp.get("window_size",      6)
    max_gap_minutes  = tp.get("max_gap_minutes",  30)
    lstm_hidden      = tp.get("lstm_hidden",       32)
    lstm_layers      = tp.get("lstm_layers",       1)
    window_size     = tp.get("window_size",      6)
    lstm_hidden     = tp.get("lstm_hidden",       32)
    lstm_layers     = tp.get("lstm_layers",       1)

    # ------------------------------------------------------------------
    # Experiment folder
    # ------------------------------------------------------------------
    experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_st"
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    shutil.copy("config.yaml", f"experiments/{experiment_name}/config.yaml")
    perf = PerformanceLogger(f"experiments/{experiment_name}/training_log.jsonl")

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    uptime_threshold = config["filters"]["uptime_threshold"]
    start_year       = config["dataset_parameters"]["start_year"]
    end_year         = config["dataset_parameters"]["end_year"]

    raingauge_df_raw, raingauge_station_mappings_df = load_raingauge_dataset(
        start=start_year, end=end_year, uptime_threshold=uptime_threshold
    )
    radar_df_raw = load_processed_dataset("database/processed_radar_dataset.pkl")
    cml_df_raw, cml_coordinates_df = load_cml_dataset(
        config["dataset_parameters"]["cml_folder"]
    )

    # ------------------------------------------------------------------
    # 2. Build complete 15-minute time grid
    #    Sort raw sources by timestamp first — merges preserve left-df order,
    #    so an unsorted source would produce an unsorted T dimension in the
    #    heterodata, causing the context-window slicing in SpatioTemporalDataset
    #    to pick up non-chronological steps and silently produce wrong windows.
    # ------------------------------------------------------------------
    # Raingauge data is at 5-min intervals; radar/CML are at 15-min.
    # Resample to 15-min averages while the timestamp is still the index,
    # then promote it to a column for the subsequent left-join.
    raingauge_df_raw = raingauge_df_raw.resample('15min', closed='left', label='left').mean()
    raingauge_df_raw = raingauge_df_raw[raingauge_df_raw.index.minute % 15 == 0]

    # raingauge_df_raw comes from pivot(index="timestamp"), so timestamp is the
    # index, not a column.  reset_index() promotes it to a column so that
    # sort_values and the subsequent left-join on "timestamp" both work.
    raingauge_df_raw = raingauge_df_raw.reset_index().sort_values("timestamp").reset_index(drop=True)
    radar_df_raw     = radar_df_raw.sort_values("timestamp").reset_index(drop=True)
    cml_df_raw       = cml_df_raw.sort_values("timestamp").reset_index(drop=True)

    all_ts = pd.to_datetime(raingauge_df_raw["timestamp"])
    complete_ts_index = pd.date_range(
        start=all_ts.min(), end=all_ts.max(), freq="15min"
    )
    complete_df = pd.DataFrame({"timestamp": complete_ts_index})
    T = len(complete_ts_index)
    print(f"Complete 15-min grid: {T} timesteps "
          f"({complete_ts_index[0]} → {complete_ts_index[-1]})")

    # ------------------------------------------------------------------
    # 3. Raingauge — left-join onto complete grid
    #    Missing timestamps → NaN → handled by GaugeGraphNew.fill_heterodata
    #    (notna() returns 0 for NaN rows, so validity flag is automatically 0)
    # ------------------------------------------------------------------
    raingauge_deduped = raingauge_df_raw.drop_duplicates("timestamp")
    raingauge_df = complete_df.merge(raingauge_deduped, on="timestamp", how="left").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Radar — left-join, fill missing rows with zero arrays
    #    Track which timestamps had real radar data for the validity flag.
    # ------------------------------------------------------------------
    radar_deduped = radar_df_raw.drop_duplicates("timestamp")
    valid_radar_ts = set(pd.to_datetime(radar_deduped["timestamp"]))
    radar_valid_flags = np.array(
        [1.0 if ts in valid_radar_ts else 0.0 for ts in complete_ts_index],
        dtype=np.float32,
    )  # [T]

    first_valid_data   = radar_deduped["data"].dropna().iloc[0]
    zero_radar_array   = np.zeros_like(first_valid_data)

    radar_full = complete_df.merge(radar_deduped, on="timestamp", how="left")
    radar_full["data"] = [
        v if isinstance(v, np.ndarray) else zero_radar_array.copy()
        for v in radar_full["data"]
    ]
    radar_full["bounds"] = radar_full["bounds"].ffill().bfill()
    radar_df = radar_full

    print(f"Radar: {int(radar_valid_flags.sum())} / {T} timesteps have real data")

    # ------------------------------------------------------------------
    # 5. CML — expand to complete (timestamp × link × station) grid
    #    A 'valid' column is added (1=real, 0=padded) and passes through
    #    CMLGraph.prepare_tensor as an extra feature automatically.
    # ------------------------------------------------------------------
    cml_df = _expand_cml_to_dense_grid(cml_df_raw, complete_df)
    print(f"CML expanded: {len(cml_df)} rows  "
          f"({int(cml_df['valid'].sum())} real, "
          f"{int((cml_df['valid'] == 0).sum())} padded)")

    print(f"raingauge_df : {raingauge_df.shape}")
    print(f"radar_df     : {radar_df.shape}")
    print(f"cml_df       : {cml_df.shape}")

    # ------------------------------------------------------------------
    # 6. Stratified spatial k-fold split
    # ------------------------------------------------------------------
    split_info = stratified_spatial_kfold_dual(
        raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    # ------------------------------------------------------------------
    # 7. Build shared heterodata for non-raingauge sources (once).
    #    Radar and CML tensors are identical across all folds — only the
    #    raingauge split masks differ.  Building them once and reusing the
    #    same tensor objects avoids duplicating large arrays (e.g. radar
    #    [N_radar, T, 2]) for each fold.
    # ------------------------------------------------------------------
    shared_radar_heterodata = None
    shared_radar_coords     = None
    shared_cml_heterodata   = None

    if "radar" in datasources:
        radar_graph             = RadarGraph(radar_df)
        shared_radar_heterodata = radar_graph.get_radar_heterodata()
        shared_radar_coords     = radar_graph.grid_coords

        # Append per-timestep validity flag as a second radar feature.
        # Use expand() without clone() — torch.cat materialises its own
        # contiguous output, so no explicit copy is needed beforehand.
        N_radar = shared_radar_heterodata["radar"].x.shape[0]
        valid_t = torch.tensor(radar_valid_flags).view(1, T, 1).expand(N_radar, -1, -1)
        shared_radar_heterodata["radar"].x = torch.cat(
            [shared_radar_heterodata["radar"].x, valid_t], dim=-1
        )
        del radar_graph, valid_t
        print(f"Radar heterodata shape: {shared_radar_heterodata['radar'].x.shape}")

    if "cml" in datasources:
        cml_graph             = CMLGraph(cml_df, cml_coordinates_df)
        shared_cml_heterodata = cml_graph.get_heterodata()
        del cml_graph

    hidden_channels = config["model"]["hidden_channels"]
    num_layers      = config["model"]["num_layers"]
    dropout         = config["model"].get("dropout", 0.0)
    out_channels    = 1

    # ------------------------------------------------------------------
    # 8. Training helper (defined here so it closes over experiment_name,
    #    config, perf, etc.)
    # ------------------------------------------------------------------

    def train_fold(model, train_loader, val_loader, fold, device="cpu"):
        print(f"\n{'='*50}\n  FOLD {fold}  |  device: {device}\n{'='*50}")
        first_param = next(model.parameters())
        print(f"Initial weight sample: {first_param.data.flatten()[:5]}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        weighted_alpha    = config["training_params"].get("weighted_loss_alpha", 0.0)
        stopping_patience = config["training_params"]["early_stop"]
        total_epochs      = config["training_params"]["epochs"]

        training_loss_arr   = []
        validation_loss_arr = []
        early_counter       = 0
        best_val_loss       = float("inf")
        epochs_run          = 0

        training_start = time.time()
        for epoch in range(total_epochs):
            epoch_start = time.time()
            print(f"\n--- Epoch {epoch + 1} ---")

            train_loss = train_epoch_st(
                model, train_loader, optimizer, device,
                weighted_loss_alpha=weighted_alpha,
            )
            val_loss = validate_st(
                model, val_loader, device,
                weighted_loss_alpha=weighted_alpha,
            )

            training_loss_arr.append(train_loss)
            validation_loss_arr.append(val_loss)
            perf.log_epoch(epoch, train_loss, val_loss)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                early_counter = 0
                torch.save(
                    model.state_dict(),
                    f"experiments/{experiment_name}/weather_gnn_best_{fold}.pth",
                )
                print("  model saved")
            else:
                early_counter += 1

            epochs_run += 1
            print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5
            print(f"  Gradient norm: {total_norm:.6f}")
            print(f"  Epoch time: {time.time() - epoch_start:.1f}s")

            if early_counter >= stopping_patience:
                print("  Early stopping triggered.")
                break

        total_time = time.time() - training_start
        perf.finalise(total_time)
        print(f"\nTraining complete: {total_time:.1f}s over {epochs_run} epochs")

        plt.plot(training_loss_arr,   label="train",      color="blue")
        plt.plot(validation_loss_arr, label="validation", color="red")
        plt.legend()
        plt.savefig(
            f"experiments/{experiment_name}/train_loss_plot_{fold}.png", dpi=300
        )
        plt.close()

    # ------------------------------------------------------------------
    # 9. Process one fold at a time.
    #    Build the gauge graph, model, and DataLoaders; train; test; then
    #    explicitly delete everything and call gc.collect() /
    #    torch.cuda.empty_cache() before the next fold.  Only one fold's
    #    data is in memory at a time.
    # ------------------------------------------------------------------
    fold_metrics    = []
    in_channels_dict = None

    for i in range(fold_count):
        print(f"\n{'='*60}\n  Building graph for fold {i} / {fold_count - 1}\n{'='*60}")

        # Build raingauge graph for this fold only
        gauge_graph = GaugeGraphNew(
            raingauge_df,
            raingauge_station_mappings_df,
            split_info=split_info[i],
            knn=config["layer_connect"]["gauge_gauge"],
        )
        if shared_radar_heterodata is not None:
            gauge_graph.add_heterodata(
                heterodata_layer=shared_radar_heterodata,
                coords=shared_radar_coords,
                layer_name="radar",
                knn=config["layer_connect"]["radar_gauge"],
            )
        if shared_cml_heterodata is not None:
            gauge_graph.add_heterodata(
                heterodata_layer=shared_cml_heterodata,
                coords=cml_coordinates_df,
                layer_name="cml",
                knn=config["layer_connect"]["cml_gauge"],
            )

        # Extract in_channels_dict from the first fold only — feature dims
        # are fold-independent (only the split mask changes between folds).
        if in_channels_dict is None:
            sample_hd = gauge_graph.get_train_heterodata()
            in_channels_dict = {
                ntype: sample_hd[ntype].x.shape[2]
                for ntype in sample_hd.node_types
            }
            print("in_channels_dict:", in_channels_dict)

        # Build a fresh model for this fold
        model = GNNInductiveHeteroST(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            edge_types=gauge_graph.get_train_heterodata().edge_types,
            window_size=window_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        ).to(device=device)

        # Build DataLoaders
        train_data = gauge_graph.get_train_heterodata()
        val_data   = gauge_graph.get_validation_heterodata()
        test_data  = gauge_graph.get_test_heterodata()

        stats = compute_norm_stats(train_data)

        train_dataset = SpatioTemporalDataset(
            apply_norm(train_data, stats),
            window_size=window_size,
        )
        val_dataset = SpatioTemporalDataset(
            apply_norm(val_data, stats),
            window_size=window_size,
        )
        test_dataset = SpatioTemporalDataset(
            apply_norm(test_data, stats),
            window_size=window_size,
        )

        print(
            f"Fold {i}  —  train: {len(train_dataset)} windows, "
            f"val: {len(val_dataset)}, test: {len(test_dataset)}"
        )

        # Drop the un-normalised references; datasets hold their own copies.
        del train_data, val_data, test_data

        train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader   = GeometricDataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
        test_loader  = GeometricDataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        # Train
        train_fold(model, train_loader, val_loader, fold=i, device=device)

        # Load best weights and evaluate
        model.load_state_dict(
            torch.load(
                f"experiments/{experiment_name}/weather_gnn_best_{i}.pth",
                map_location=device,
            )
        )
        metrics_dict = test_model_st(
            model,
            raingauge_station_mappings_df,
            test_loader,
            device,
            fold=i,
            experiment_name=experiment_name,
        )
        fold_metrics.append(metrics_dict)

        # Free this fold's allocations before the next iteration
        del gauge_graph, model
        del train_dataset, val_dataset, test_dataset
        del train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    averaged = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in ["rmse", "mae", "pearson_r", "timestep_rmse",
                    "precision", "recall", "f1"]
    }
    print("\n=== Cross-fold averaged metrics ===")
    for k, v in averaged.items():
        print(f"  {k:20s}: {v:.4f}")

    return averaged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

config = read_config("config.yaml")
train_st(config=config)
