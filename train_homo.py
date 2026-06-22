"""
train_homo.py
=============
Training script for GNNHomoBaseline — the homogeneous GNN baseline.

Mirrors train_fused.py almost exactly; the two changes are:

  1. GNNInductiveHetero  →  GNNHomoBaseline
  2. model.set_graph_sizes() called before each train / val / test phase
     because the raingauge node count differs across graph splits.

All data loading, normalisation, k-fold setup, and evaluation reuse the
same code as train_fused.py so results are directly comparable.
"""

from src.sampling.main import stratified_spatial_kfold_dual  # must come first

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
from models.gnn_homo import GNNHomoBaseline
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_processed_dataset
from src.cml.utils import load_cml_dataset
from training.logic_hetero import train_epoch, validate, test_model
from src.graph.cmlgraph import CMLGraph
from src.graph.radargraph import RadarGraph
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive


def train_homo(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size  = config['training_params']['batch_size']
    fold_count  = config['training_params']['fold_count']
    datasources = config['datasources']

    # ------------------------------------------------------------------ #
    # Experiment folder
    # ------------------------------------------------------------------ #
    experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_homo_baseline"
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    shutil.copy("config.yaml", f"experiments/{experiment_name}/config.yaml")
    perf = PerformanceLogger(f"experiments/{experiment_name}/training_log.jsonl")

    # ------------------------------------------------------------------ #
    # Load data  (identical to train_fused.py)
    # ------------------------------------------------------------------ #
    uptime_threshold = config['filters']['uptime_threshold']
    start_year       = config['dataset_parameters']['start_year']
    end_year         = config['dataset_parameters']['end_year']

    raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(
        start=start_year, end=end_year, uptime_threshold=uptime_threshold
    )
    raingauge_df = raingauge_df.resample('15min', closed='left', label='left').mean()
    raingauge_df = raingauge_df[raingauge_df.index.minute % 15 == 0]

    radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")

    radar_cols     = radar_df.columns
    raingauge_cols = raingauge_df.columns
    merged_df      = radar_df.merge(raingauge_df, on=['timestamp'], how='inner')

    cml_df, cml_coordinates_df = load_cml_dataset(config['dataset_parameters']['cml_folder'])
    cml_df    = cml_df.fillna(0)
    cml_cols  = cml_df.columns
    merged_df = merged_df.merge(cml_df, on=['timestamp'], how='inner')
    cml_df    = merged_df[cml_cols]

    raingauge_df = pd.concat(
        [merged_df['timestamp'], merged_df[raingauge_cols]], axis=1
    ).drop_duplicates().reset_index(drop=True)
    radar_df = merged_df[radar_cols].drop_duplicates(subset=['timestamp'], keep='first')
    cml_df   = cml_df.drop_duplicates()

    print(raingauge_df.shape)
    print(radar_df.shape)
    print(cml_df.shape)

    # ------------------------------------------------------------------ #
    # Stratified k-fold split
    # ------------------------------------------------------------------ #
    split_info = stratified_spatial_kfold_dual(
        raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    # ------------------------------------------------------------------ #
    # Build graphs
    # ------------------------------------------------------------------ #
    gauge_graph_arr = []
    for i in range(fold_count):
        gauge_graph = GaugeGraphNew(
            raingauge_df, raingauge_station_mappings_df,
            split_info=split_info[i],
            knn=config['layer_connect']['gauge_gauge'],
        )
        if 'radar' in datasources:
            radar_graph    = RadarGraph(radar_df)
            radar_hetero   = radar_graph.get_radar_heterodata()
            gauge_graph.add_heterodata(
                heterodata_layer=radar_hetero,
                coords=radar_graph.grid_coords,
                layer_name='radar',
                knn=config['layer_connect']['radar_gauge'],
            )
        if 'cml' in datasources:
            cml_graph  = CMLGraph(cml_df, cml_coordinates_df)
            cml_hetero = cml_graph.get_heterodata()
            gauge_graph.add_heterodata(
                heterodata_layer=cml_hetero,
                coords=cml_coordinates_df,
                layer_name='cml',
                knn=config['layer_connect']['cml_gauge'],
            )
        gauge_graph_arr.append(gauge_graph)

    # ------------------------------------------------------------------ #
    # Build models
    # ------------------------------------------------------------------ #
    hidden_channels    = config['model']['hidden_channels']
    out_channels       = 1
    num_layers         = config['model']['num_layers']
    dropout            = config['model'].get('dropout', 0.0)
    raingauge_features = 6 if config['dataset_parameters']['include_lpe'] else 2

    if 'cml' in datasources:
        cml_features = gauge_graph_arr[0].get_train_heterodata()['cml'].x.shape[2]

    model_arr = []
    for i in range(fold_count):
        in_channels_dict = {'raingauge': raingauge_features, 'radar': 1}
        if 'cml' in datasources:
            in_channels_dict['cml'] = cml_features

        model_arr.append(
            GNNHomoBaseline(
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                edge_types=gauge_graph_arr[i].get_train_heterodata().edge_types,
                dropout=dropout,
            ).to(device=device)
        )

    # ------------------------------------------------------------------ #
    # Normalisation helpers
    # ------------------------------------------------------------------ #
    def compute_norm_stats(heterodata):
        stats = {}
        for node_type in heterodata.node_types:
            x    = heterodata[node_type].x  # [N, T, F]
            mean = x.mean(dim=(0, 1))
            std  = x.std(dim=(0, 1)).clamp(min=1e-8)
            stats[node_type] = (mean, std)
        return stats

    def apply_norm(heterodata, stats):
        normed = heterodata.clone()
        for node_type in heterodata.node_types:
            if node_type in stats:
                mean, std = stats[node_type]
                normed[node_type].x = (heterodata[node_type].x - mean) / std
        return normed

    # ------------------------------------------------------------------ #
    # Build loaders
    # ------------------------------------------------------------------ #
    train_loader_arr = []
    val_loader_arr   = []
    test_loader_arr  = []
    train_data_arr   = []
    val_data_arr     = []
    test_data_arr    = []

    for i in range(fold_count):
        raw_train = gauge_graph_arr[i].get_train_heterodata()
        raw_val   = gauge_graph_arr[i].get_validation_heterodata()
        raw_test  = gauge_graph_arr[i].get_test_heterodata()

        stats      = compute_norm_stats(raw_train)
        train_data = apply_norm(raw_train, stats)
        val_data   = apply_norm(raw_val,   stats)
        test_data  = apply_norm(raw_test,  stats)

        train_data_arr.append(train_data)
        val_data_arr.append(val_data)
        test_data_arr.append(test_data)

        train_loader_arr.append(GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(train_data),
            batch_size=batch_size, shuffle=False,
        ))
        val_loader_arr.append(GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(val_data),
            batch_size=batch_size, shuffle=False,
        ))
        test_loader_arr.append(GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(test_data),
            batch_size=batch_size, shuffle=False,
        ))

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def _graph_sizes(heterodata):
        """Return {node_type: N_nodes} for set_graph_sizes()."""
        return {nt: heterodata[nt].x.shape[0] for nt in heterodata.node_types}

    def train_fold(model, train_data, val_data,
                   train_loader, val_loader, fold, device="cpu"):
        print("Training")
        print(f"Device type: {device}")
        first_param = next(model.parameters())
        print(f"Initial weight sample: {first_param.data.flatten()[:5]}")

        optimizer      = torch.optim.Adam(
            model.parameters(),
            lr=config['model']['learning_rate'],
            weight_decay=float(config['model']['weight_decay']),
        )
        training_loss_arr   = []
        validation_loss_arr = []
        early              = 0
        mini               = float('inf')
        stopping_condition = config['training_params']['early_stop']
        total_epochs       = config['training_params']['epochs']
        weighted_alpha     = config['training_params'].get('weighted_loss_alpha', 0.0)
        epochs_run         = 0

        print(f"-----FOLD: {fold}-----")
        training_start = time.time()

        for epoch_i in range(total_epochs):
            epoch_start = time.time()
            print(f"-----EPOCH: {epoch_i + 1}-----")

            # Train — raingauge count is smaller in training split
            model.set_graph_sizes(_graph_sizes(train_data))
            train_loss = train_epoch(model, train_loader, optimizer, device,
                                     weighted_loss_alpha=weighted_alpha)
            print(train_loss)

            # Validate — raingauge count includes held-out nodes too
            model.set_graph_sizes(_graph_sizes(val_data))
            validation_loss = validate(model, val_loader, device,
                                       weighted_loss_alpha=weighted_alpha)

            training_loss_arr.append(train_loss)
            validation_loss_arr.append(validation_loss)
            perf.log_epoch(epoch_i, train_loss, validation_loss)

            if mini >= validation_loss:
                mini  = validation_loss
                early = 0
                torch.save(
                    model.state_dict(),
                    f"experiments/{experiment_name}/homo_gnn_best_{fold}.pth",
                )
                print("✅ model weights saved")
            else:
                early += 1

            epochs_run += 1
            if early >= stopping_condition:
                print("Early stop")
                break

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")

            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            print(f"Gradient norm: {total_norm:.6f}")
            print(f"epoch {epoch_i} took {time.time() - epoch_start:.1f}s")

        training_end = time.time()
        total_time   = training_end - training_start
        perf.finalise(total_time)
        print(f"Training took {total_time:.1f}s over {epochs_run} epochs")

        plt.plot(training_loss_arr,   label="training_loss",   color="blue")
        plt.plot(validation_loss_arr, label="validation_loss", color="red")
        plt.legend()
        plt.savefig(
            f"experiments/{experiment_name}/train_loss_plot_{fold}.png", dpi=300
        )
        plt.close()

    # ------------------------------------------------------------------ #
    # Run all folds
    # ------------------------------------------------------------------ #
    fold_metrics = []
    for i in range(fold_count):
        train_fold(
            model_arr[i],
            train_data=train_data_arr[i],
            val_data=val_data_arr[i],
            train_loader=train_loader_arr[i],
            val_loader=val_loader_arr[i],
            fold=i,
            device=device,
        )

        model_arr[i].load_state_dict(
            torch.load(
                f"experiments/{experiment_name}/homo_gnn_best_{i}.pth",
                map_location=device,
            )
        )

        # Test — all 51 stations visible
        model_arr[i].set_graph_sizes(_graph_sizes(test_data_arr[i]))
        metrics_dict = test_model(
            model_arr[i],
            raingauge_station_mappings_df,
            test_loader_arr[i],
            device,
            fold=i,
            experiment_name=experiment_name,
        )
        fold_metrics.append(metrics_dict)

    averaged = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in ["rmse", "mae", "pearson_r", "timestep_rmse",
                    "precision", "recall", "f1"]
    }
    return averaged


config = read_config("config.yaml")
train_homo(config=config)
