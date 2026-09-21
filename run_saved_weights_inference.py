"""
run_saved_weights_inference.py
===============================
Re-run test-set inference from ALREADY-TRAINED checkpoints (no training),
using the updated test_model() in training/logic_hetero.py which now also
reports CSI (Critical Success Index), FAR (False Alarm Ratio) and POD
(Probability of Detection) alongside the original RMSE/MAE/Pearson r/
precision/recall/F1.

Rebuilds the exact same data pipeline, graph structure and fold splits as
train_fused.py (same join order, same stratified_spatial_kfold_dual seed,
same per-fold train-derived normalisation stats), then loads each fold's
weather_gnn_best_{fold}.pth and calls test_model() directly instead of
training. Since evaluation is deterministic (model.eval(), no dropout),
this reproduces the original RMSE/MAE/etc. bit-for-bit while adding the
new metrics.

Usage
-----
python run_saved_weights_inference.py --experiment-dir experiments/raingauge_cml_radar_final

By default this OVERWRITES that folder's per_station_metrics_f*.csv and
test_scatter_plot_*.png with updated versions (same numbers + CSI/FAR/POD
columns) — this is what test_model() always does. Pass --output-suffix to
instead write into a subfolder and leave the original files untouched, e.g.:

python run_saved_weights_inference.py \\
    --experiment-dir experiments/raingauge_cml_radar_final \\
    --output-suffix with_csi_far_pod
"""

from src.sampling.main import stratified_spatial_kfold_dual  # must init first (see train.py)

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_processed_dataset
from src.cml.utils import load_cml_dataset
from training.logic_hetero import test_model
import src.graph.gaugegraphnew as gaugegraphnew_module
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive
from src.graph.cmlgraph import CMLGraph
from src.graph.radargraph import RadarGraph


def compute_norm_stats(heterodata):
    """Same as train_fused.py compute_norm_stats — must match exactly for checkpoint compatibility."""
    stats = {}
    for node_type in heterodata.node_types:
        x = heterodata[node_type].x
        mean = x.mean(dim=(0, 1))
        std = x.std(dim=(0, 1)).clamp(min=1e-8)
        stats[node_type] = (mean, std)
    return stats


def apply_norm(heterodata, stats):
    """Same as train_fused.py apply_norm. Only touches .x, never .y."""
    normed = heterodata.clone()
    for node_type in heterodata.node_types:
        if node_type in stats:
            mean, std = stats[node_type]
            normed[node_type].x = (heterodata[node_type].x - mean) / std
    return normed


def run_inference(experiment_dir, config_path, rain_threshold, output_experiment_name, folds):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = read_config(config_path)

    # GaugeGraphNew reads a MODULE-LEVEL `config` (loaded once from config.yaml
    # in the CWD at import time) for include_lpe / is_directed. Override it so
    # graph construction matches exactly what this checkpoint was trained
    # with, regardless of whatever the root config.yaml currently contains.
    gaugegraphnew_module.config = config

    batch_size = config['training_params']['batch_size']
    fold_count = config['training_params']['fold_count']
    datasources = config['datasources']

    uptime_threshold = config['filters']['uptime_threshold']
    start_year = config['dataset_parameters']['start_year']
    end_year = config['dataset_parameters']['end_year']
    raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(
        start=start_year, end=end_year, uptime_threshold=uptime_threshold
    )

    # Raingauge data is at 5-min intervals; radar/CML are at 15-min. Resample
    # BEFORE the inner join so timestamps align — mirrors train_fused.py.
    raingauge_df = raingauge_df.resample('15min', closed='left', label='left').mean()
    raingauge_df = raingauge_df[raingauge_df.index.minute % 15 == 0]

    radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")

    radar_cols = radar_df.columns
    raingauge_cols = raingauge_df.columns
    merged_df = radar_df.merge(raingauge_df, on=['timestamp'], how='inner')

    cml_df, cml_coordinates_df = load_cml_dataset(config['dataset_parameters']['cml_folder'])
    cml_df = cml_df.fillna(0)
    cml_cols = cml_df.columns
    merged_df = merged_df.merge(cml_df, on=['timestamp'], how='inner')
    cml_df = merged_df[cml_cols]

    raingauge_df = merged_df[raingauge_cols]
    radar_df = merged_df[radar_cols]
    raingauge_df = pd.concat(
        [merged_df['timestamp'], merged_df[raingauge_cols]], axis=1
    ).drop_duplicates().reset_index(drop=True)
    radar_df = radar_df.drop_duplicates(subset=['timestamp'], keep='first')
    cml_df = cml_df.drop_duplicates()

    print(f"raingauge_df: {raingauge_df.shape}, radar_df: {radar_df.shape}, cml_df: {cml_df.shape}")

    split_info = stratified_spatial_kfold_dual(
        raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    hidden_channels = config['model']['hidden_channels']
    out_channels = 1
    num_layers = config['model']['num_layers']
    dropout = config['model'].get('dropout', 0.0)
    raingauge_features = 6 if config['dataset_parameters']['include_lpe'] else 2

    requested_folds = folds if folds is not None else list(range(fold_count))

    fold_metrics = []
    for i in requested_folds:
        print(f"\n=== Fold {i} ===")
        gauge_graph = GaugeGraphNew(
            raingauge_df, raingauge_station_mappings_df,
            split_info=split_info[i], knn=config['layer_connect']['gauge_gauge'],
        )
        if 'radar' in datasources:
            radar_graph = RadarGraph(radar_df)
            radar_heterodata = radar_graph.get_radar_heterodata()
            gauge_graph.add_heterodata(
                heterodata_layer=radar_heterodata, coords=radar_graph.grid_coords,
                layer_name='radar', knn=config['layer_connect']['radar_gauge'],
            )
        if 'cml' in datasources:
            cml_graph = CMLGraph(cml_df, cml_coordinates_df)
            cml_heterodata = cml_graph.get_heterodata()
            gauge_graph.add_heterodata(
                heterodata_layer=cml_heterodata, coords=cml_coordinates_df,
                layer_name='cml', knn=config['layer_connect']['cml_gauge'],
            )

        train_data = gauge_graph.get_train_heterodata()
        test_data = gauge_graph.get_test_heterodata()
        stats = compute_norm_stats(train_data)

        test_loader = GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(apply_norm(test_data, stats)),
            batch_size=batch_size, shuffle=False,
        )

        if 'cml' in datasources:
            cml_features = train_data['cml'].x.shape[2]
            model = GNNInductiveHetero(
                in_channels_dict={
                    "raingauge": raingauge_features,
                    "radar": 1,
                    "cml": cml_features,
                },
                hidden_channels=hidden_channels, out_channels=out_channels,
                num_layers=num_layers, edge_types=train_data.edge_types, dropout=dropout,
            ).to(device=device)
        else:
            model = GNNInductiveHetero(
                in_channels_dict={"raingauge": raingauge_features, "radar": 1},
                hidden_channels=hidden_channels, out_channels=out_channels,
                num_layers=num_layers, edge_types=train_data.edge_types, dropout=dropout,
            ).to(device=device)

        ckpt_path = os.path.join(experiment_dir, f"weather_gnn_best_{i}.pth")
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        metrics_dict = test_model(
            model, raingauge_station_mappings_df, test_loader, device,
            fold=i, experiment_name=output_experiment_name, rain_threshold=rain_threshold,
        )
        fold_metrics.append(metrics_dict)

    averaged_metrics_dict = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in ["rmse", "mae", "pearson_r", "timestep_rmse",
                    "precision", "recall", "f1", "pod", "far", "csi"]
    }

    print("\n=== AVERAGED METRICS ACROSS FOLDS ===")
    for k, v in averaged_metrics_dict.items():
        print(f"  {k:>14s}: {v:.4f}")

    return averaged_metrics_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', required=True,
                         help='Path to experiments/<name> containing weather_gnn_best_{fold}.pth')
    parser.add_argument('--config', default=None,
                         help='Path to the config.yaml used to train the checkpoints '
                              '(default: <experiment-dir>/config.yaml)')
    parser.add_argument('--rain-threshold', type=float, default=0.5,
                         help='Rainfall threshold (mm) for CSI/FAR/POD/precision/recall/F1')
    parser.add_argument('--output-suffix', default=None,
                         help='If set, write outputs to experiments/<name>/<suffix> instead of '
                              'overwriting the original per_station_metrics_f*.csv / plots in place')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                         help='Fold indices to run (default: all folds in config)')
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.rstrip('/')
    config_path = args.config or os.path.join(experiment_dir, 'config.yaml')

    # test_model() saves to f"experiments/{experiment_name}/..." — derive that
    # relative name from experiment_dir so paths land in the right place
    # whether experiment_dir is "experiments/foo" or a bare "foo".
    base_name = experiment_dir[len('experiments/'):] if experiment_dir.startswith('experiments/') else experiment_dir
    output_experiment_name = f"{base_name}/{args.output_suffix}" if args.output_suffix else base_name

    run_inference(
        experiment_dir=experiment_dir,
        config_path=config_path,
        rain_threshold=args.rain_threshold,
        output_experiment_name=output_experiment_name,
        folds=args.folds,
    )


if __name__ == '__main__':
    main()
