"""
train_st.py  —  Spatio-Temporal rainfall interpolation training pipeline.

Extends train_fused.py with a temporal context window fed to an LSTM encoder
inside the model (GNNInductiveHeteroST).  The spatial graph construction,
k-fold split strategy, normalisation, and evaluation metrics are identical to
train_fused.py so the two experiments are directly comparable.

Key differences vs train_fused.py
-----------------------------------
1. Dataset  : SpatioTemporalDataset (window of W preceding timesteps) instead
              of HeterogeneousWeatherGraphDatasetInductive (single timestep).
2. Model    : GNNInductiveHeteroST (LSTM + GNN) instead of GNNInductiveHetero.
3. Masking  : Only the *current-timestep* value of the target node is zeroed.
              The context window for that node is kept intact — no leakage
              because we are predicting the current value, not a past one.
4. Config   : Reads a [temporal_params] section from config.yaml (added below).
              Falls back to sensible defaults if the section is absent so that
              train_fused.py continues to work unchanged.

Non-contiguous / missing timesteps
------------------------------------
SpatioTemporalDataset automatically filters out target timesteps whose
preceding W-step window spans a gap larger than max_gap_minutes.  These
samples are silently dropped; no imputation is performed.
"""

from src.sampling.main import stratified_spatial_kfold_dual  # must be first

import torch
import os
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
    """Compute per-feature mean and std from one split's node features [N, T, F]."""
    stats = {}
    for node_type in heterodata.node_types:
        x = heterodata[node_type].x  # [N, T, F]
        mean = x.mean(dim=(0, 1))
        std = x.std(dim=(0, 1)).clamp(min=1e-8)
        stats[node_type] = (mean, std)
    return stats


def apply_norm(heterodata, stats):
    """Apply precomputed normalisation stats.  Only touches .x, never .y."""
    normed = heterodata.clone()
    for node_type in heterodata.node_types:
        if node_type in stats:
            mean, std = stats[node_type]
            normed[node_type].x = (heterodata[node_type].x - mean) / std
    return normed


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_st(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config["training_params"]["batch_size"]
    fold_count = config["training_params"]["fold_count"]
    datasources = config["datasources"]

    # Temporal hyperparameters — read from config with defaults
    tp = config.get("temporal_params", {})
    window_size      = tp.get("window_size",      6)
    max_gap_minutes  = tp.get("max_gap_minutes",  10)
    lstm_hidden      = tp.get("lstm_hidden",       32)
    lstm_layers      = tp.get("lstm_layers",       1)

    # ------------------------------------------------------------------
    # Experiment folder
    # ------------------------------------------------------------------
    experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_st"
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    perf = PerformanceLogger(f"experiments/{experiment_name}/training_log.jsonl")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    uptime_threshold = config["filters"]["uptime_threshold"]
    start_year       = config["dataset_parameters"]["start_year"]
    end_year         = config["dataset_parameters"]["end_year"]

    raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(
        start=start_year, end=end_year, uptime_threshold=uptime_threshold
    )
    radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")

    # ------------------------------------------------------------------
    # Inner-join all sources on timestamp
    # ------------------------------------------------------------------
    radar_cols     = radar_df.columns
    raingauge_cols = raingauge_df.columns
    merged_df      = radar_df.merge(raingauge_df, on=["timestamp"], how="inner")

    cml_df, cml_coordinates_df = load_cml_dataset(
        config["dataset_parameters"]["cml_folder"]
    )
    cml_df    = cml_df.fillna(0)
    cml_cols  = cml_df.columns
    merged_df = merged_df.merge(cml_df, on=["timestamp"], how="inner")
    cml_df    = merged_df[cml_cols]

    raingauge_df = (
        pd.concat([merged_df["timestamp"], merged_df[raingauge_cols]], axis=1)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    radar_df = radar_df.drop_duplicates(subset=["timestamp"], keep="first")
    cml_df   = cml_df.drop_duplicates()

    print(f"raingauge_df : {raingauge_df.shape}")
    print(f"radar_df     : {radar_df.shape}")
    print(f"cml_df       : {cml_df.shape}")

    # Timestamps aligned with the T dimension of all heterodata objects
    timestamps = raingauge_df["timestamp"].values

    # ------------------------------------------------------------------
    # Stratified spatial k-fold split
    # ------------------------------------------------------------------
    split_info = stratified_spatial_kfold_dual(
        raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    # ------------------------------------------------------------------
    # Build heterogeneous graphs (one per fold)
    # ------------------------------------------------------------------
    gauge_graph_arr = []
    for i in range(fold_count):
        gauge_graph = GaugeGraphNew(
            raingauge_df,
            raingauge_station_mappings_df,
            split_info=split_info[i],
            knn=config["layer_connect"]["gauge_gauge"],
        )
        if "radar" in datasources:
            radar_graph      = RadarGraph(radar_df)
            radar_heterodata = radar_graph.get_radar_heterodata()
            gauge_graph.add_heterodata(
                heterodata_layer=radar_heterodata,
                coords=radar_graph.grid_coords,
                layer_name="radar",
                knn=config["layer_connect"]["radar_gauge"],
            )
        if "cml" in datasources:
            cml_graph      = CMLGraph(cml_df, cml_coordinates_df)
            cml_heterodata = cml_graph.get_heterodata()
            gauge_graph.add_heterodata(
                heterodata_layer=cml_heterodata,
                coords=cml_coordinates_df,
                layer_name="cml",
                knn=config["layer_connect"]["cml_gauge"],
            )
        gauge_graph_arr.append(gauge_graph)

    # ------------------------------------------------------------------
    # Build models (one per fold)
    # ------------------------------------------------------------------
    raingauge_features = 6 if config["dataset_parameters"]["include_lpe"] else 2
    hidden_channels    = config["model"]["hidden_channels"]
    num_layers         = config["model"]["num_layers"]
    out_channels       = 1

    if "cml" in datasources:
        cml_features = gauge_graph_arr[0].get_train_heterodata()["cml"].x.shape[2]

    model_arr = []
    for i in range(fold_count):
        in_channels_dict = {"raingauge": raingauge_features}
        if "radar" in datasources:
            in_channels_dict["radar"] = 1
        if "cml" in datasources:
            in_channels_dict["cml"] = cml_features

        model_arr.append(
            GNNInductiveHeteroST(
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                edge_types=gauge_graph_arr[i].get_train_heterodata().edge_types,
                window_size=window_size,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
            ).to(device=device)
        )

    # ------------------------------------------------------------------
    # Build DataLoaders using SpatioTemporalDataset
    # ------------------------------------------------------------------
    train_loader_arr = []
    val_loader_arr   = []
    test_loader_arr  = []

    for i in range(fold_count):
        train_data = gauge_graph_arr[i].get_train_heterodata()
        val_data   = gauge_graph_arr[i].get_validation_heterodata()
        test_data  = gauge_graph_arr[i].get_test_heterodata()

        # Compute normalisation stats from training split only
        stats = compute_norm_stats(train_data)

        train_dataset = SpatioTemporalDataset(
            apply_norm(train_data, stats),
            timestamps,
            window_size=window_size,
            max_gap_minutes=max_gap_minutes,
        )
        val_dataset = SpatioTemporalDataset(
            apply_norm(val_data, stats),
            timestamps,
            window_size=window_size,
            max_gap_minutes=max_gap_minutes,
        )
        test_dataset = SpatioTemporalDataset(
            apply_norm(test_data, stats),
            timestamps,
            window_size=window_size,
            max_gap_minutes=max_gap_minutes,
        )

        print(
            f"Fold {i}  —  train: {len(train_dataset)} windows, "
            f"val: {len(val_dataset)}, test: {len(test_dataset)}"
        )

        train_loader_arr.append(
            GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        )
        val_loader_arr.append(
            GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        )
        test_loader_arr.append(
            GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        )

    # ------------------------------------------------------------------
    # Training loop (per fold)
    # ------------------------------------------------------------------

    def train_fold(model, train_loader, val_loader, fold, device="cpu"):
        print(f"\n{'='*50}")
        print(f"  FOLD {fold}  |  device: {device}")
        print(f"{'='*50}")
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

            # Gradient norm diagnostics
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
    # Run all folds
    # ------------------------------------------------------------------
    fold_metrics = []
    for i in range(fold_count):
        train_fold(
            model_arr[i],
            train_loader=train_loader_arr[i],
            val_loader=val_loader_arr[i],
            fold=i,
            device=device,
        )

        # Reload best checkpoint before testing
        model_arr[i].load_state_dict(
            torch.load(
                f"experiments/{experiment_name}/weather_gnn_best_{i}.pth",
                map_location=device,
            )
        )
        metrics_dict = test_model_st(
            model_arr[i],
            raingauge_station_mappings_df,
            test_loader_arr[i],
            device,
            fold=i,
            experiment_name=experiment_name,
        )
        fold_metrics.append(metrics_dict)

    # ------------------------------------------------------------------
    # Average metrics across folds
    # ------------------------------------------------------------------
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
