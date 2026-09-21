"""
run_homo3_inference.py
=======================
Re-run test-set inference for experiments/homo_3, a raingauge-only
GNNInductiveHetero checkpoint (single node type "raingauge", single edge
type raingauge-connects-raingauge) trained back in Feb 2026.

Why this script exists instead of run_saved_weights_inference.py
------------------------------------------------------------------
homo_3/ has no config.yaml and its weights don't match the CURRENT
src/graph/gaugegraphnew.py (which always emits 2 or 6 raingauge features
for include_lpe False/True). Inspecting the checkpoint's state_dict shows
it expects exactly 1 raingauge feature per node (raw rainfall value, no
LPE, no extra channels), hidden_channels=8, num_layers=3, and only a
raingauge<->raingauge edge type -- no radar/CML. That matches the
raingauge-only GaugeGraphNew implementation that existed around commit
cf68731 (Feb 2026), before LPE / radar / CML fusion was added.

This script vendors a copy of that old GaugeGraphNew (renamed
LegacyGaugeGraphNew) so the graph fed to the model has the exact shape
the checkpoint expects, then calls the CURRENT test_model() (which
already computes RMSE/MAE/Pearson r/Precision/Recall/F1/POD/FAR/CSI).

Caveat
------
The exact script/config that produced homo_3 isn't recoverable from git
(the committed config.yaml at the time had fold_count=1, but homo_3 has
5 folds, so a local uncommitted config was used). Dataset filters here
(start_year=2024, end_year=2025, uptime_threshold=0.9, knn=5, seed=123)
are taken from the current config.yaml, which already matches the values
seen throughout that period's git history. Results should be very close
to the original run but are a best-effort reconstruction, not a
guaranteed bit-exact replay.

Usage
-----
python run_homo3_inference.py
"""

from src.sampling.main import stratified_spatial_kfold_dual  # must init first (see train.py)

import argparse
import os

import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.transforms import ToUndirected

from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from training.logic_hetero import test_model


class LegacyGaugeGraphNew:
    """Vendored copy of src/graph/gaugegraphnew.py as of commit cf68731 --
    the raingauge-only version (1 feature/node, no LPE) that homo_3's
    checkpoints were trained against."""

    def __init__(self, data_df: pd.DataFrame, mapping_df: pd.DataFrame, split_info: dict, knn: int):
        self.dtype = torch.float32
        self.raingauge_df = data_df[mapping_df['id'].values.tolist()]
        self.mapping_df = mapping_df
        self.split_info = split_info
        self.knn = knn
        self.train_gauges = split_info["ml"]['train']
        self.validation_gauges = split_info['ml']['validation']
        self.test_gauges = split_info['ml']['test']

        self.train_mask, self.val_mask, self.test_mask = self.initialise_masks()

        self.train_graph = self.build_graph("train")
        self.validation_graph = self.build_graph("validation")
        self.test_graph = self.build_graph("test")

        self.train_heterodata = self.fill_heterodata("train")
        self.validation_heterodata = self.fill_heterodata("validation")
        self.test_heterodata = self.fill_heterodata("test")

        self.train_heterodata = ToUndirected()(self.train_heterodata)
        self.validation_heterodata = ToUndirected()(self.validation_heterodata)
        self.test_heterodata = ToUndirected()(self.test_heterodata)

    def get_train_heterodata(self):
        return self.train_heterodata

    def get_validation_heterodata(self):
        return self.validation_heterodata

    def get_test_heterodata(self):
        return self.test_heterodata

    def build_graph(self, split: str):
        match split:
            case "train":
                mask = self.train_mask
            case "validation":
                mask = np.logical_or(self.train_mask, self.val_mask)
            case "test":
                mask = np.ones(self.mapping_df.shape[0]).astype(bool)

        G = nx.Graph()
        filtered_mapping_df = self.mapping_df[mask]
        coords = filtered_mapping_df[['longitude', 'latitude']].values

        ball_tree = NearestNeighbors(n_neighbors=self.knn + 1, algorithm='ball_tree').fit(coords)
        distances, indices = ball_tree.kneighbors(coords)

        for idx, row in filtered_mapping_df.iterrows():
            G.add_node(idx, lat=row['latitude'], lon=row['longitude'])

        for i, neighbors in enumerate(indices):
            for j, neighbor_idx in enumerate(neighbors[1:]):
                dist = distances[i][j + 1]
                G.add_edge(i, neighbor_idx, weight=dist)

        return G

    def initialise_masks(self):
        train_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)
        validation_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)
        test_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)

        train_mask[self.mapping_df['order'][self.mapping_df['id'].isin(self.train_gauges)].index.to_list()] = True
        validation_mask[self.mapping_df['order'][self.mapping_df['id'].isin(self.validation_gauges)].index.to_list()] = True
        test_mask[self.mapping_df['order'][self.mapping_df['id'].isin(self.test_gauges)].index.to_list()] = True
        return train_mask, validation_mask, test_mask

    def fill_heterodata(self, graph: str) -> HeteroData:
        heterodata = HeteroData()
        heterodata['raingauge'].x = torch.tensor(self.raingauge_df.values.T, dtype=torch.float32).unsqueeze(-1)
        heterodata['raingauge'].y = torch.tensor(self.raingauge_df.values.T, dtype=torch.float32).unsqueeze(-1)

        match graph:
            case "train":
                mask = torch.tensor(self.train_mask, dtype=bool)
                heterodata['raingauge'].mask = []
                edges = self.train_graph.edges(data=True)
            case "validation":
                mask = torch.tensor(np.logical_or(self.train_mask, self.val_mask), dtype=bool)
                val = self.mapping_df[self.mapping_df['id'].isin(self.validation_gauges) | self.mapping_df['id'].isin(self.train_gauges)]
                heterodata['raingauge'].mask = val['id'].isin(self.validation_gauges).to_numpy()
                edges = self.validation_graph.edges(data=True)
            case "test":
                mask = torch.tensor(np.ones(len(self.test_mask)), dtype=bool)
                heterodata['raingauge'].mask = self.test_mask
                edges = self.test_graph.edges(data=True)

        heterodata['raingauge'].x = heterodata['raingauge'].x[mask]
        heterodata['raingauge'].y = heterodata['raingauge'].y[mask]
        edge_index = []
        edge_attr = []
        for A, B, data in edges:
            edge_index.append([A, B])
            edge_attr.append(data['weight'])
        heterodata['raingauge', 'connects', 'raingauge'].edge_index = torch.tensor(edge_index, dtype=int).T
        heterodata['raingauge', 'connects', 'raingauge'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        heterodata['raingauge'].num_nodes = torch.tensor(heterodata['raingauge'].x.shape[0], dtype=torch.int32)
        return heterodata


class LegacyHeterogeneousWeatherGraphDatasetInductive(Dataset):
    """Vendored copy matching LegacyGaugeGraphNew's single-node-type output."""

    def __init__(self, heterodata):
        self.heterodata = heterodata
        self.num_timesteps = heterodata['raingauge'].x.shape[1]

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        x = self.heterodata['raingauge'].x[:, idx, :]
        y = self.heterodata['raingauge'].y[:, idx, :]

        data = HeteroData()
        data['raingauge'].x = x
        data['raingauge'].y = y
        for edge_type in self.heterodata.edge_types:
            data[edge_type].edge_index = self.heterodata[edge_type].edge_index
            data[edge_type].edge_attr = self.heterodata[edge_type].edge_attr
        data['raingauge'].mask = torch.tensor(self.heterodata['raingauge'].mask)
        data['raingauge'].num_nodes = self.heterodata['raingauge'].x.shape[0]
        return data


def run_inference(experiment_dir, config_path, rain_threshold, output_experiment_name, folds, knn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = read_config(config_path)

    batch_size = config['training_params']['batch_size']
    fold_count = config['training_params']['fold_count']

    uptime_threshold = config['filters']['uptime_threshold']
    start_year = config['dataset_parameters']['start_year']
    end_year = config['dataset_parameters']['end_year']

    raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(
        start=start_year, end=end_year, uptime_threshold=uptime_threshold
    )
    raingauge_df = raingauge_df.fillna(0)

    print(f"raingauge_df: {raingauge_df.shape}, stations: {raingauge_station_mappings_df.shape[0]}")

    split_info = stratified_spatial_kfold_dual(
        raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    hidden_channels = 8
    num_layers = 3
    out_channels = 1

    requested_folds = folds if folds is not None else list(range(fold_count))

    fold_metrics = []
    for i in requested_folds:
        print(f"\n=== Fold {i} ===")
        gauge_graph = LegacyGaugeGraphNew(
            raingauge_df, raingauge_station_mappings_df, split_info=split_info[i], knn=knn,
        )
        test_data = gauge_graph.get_test_heterodata()

        test_loader = GeometricDataLoader(
            LegacyHeterogeneousWeatherGraphDatasetInductive(test_data),
            batch_size=batch_size, shuffle=False,
        )

        model = GNNInductiveHetero(
            in_channels_dict={"raingauge": 1},
            hidden_channels=hidden_channels, out_channels=out_channels,
            num_layers=num_layers, edge_types=test_data.edge_types,
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
    parser.add_argument('--experiment-dir', default='experiments/homo_3')
    parser.add_argument('--config', default='config.yaml',
                         help='Config to source dataset filters (start/end year, uptime, fold_count) from')
    parser.add_argument('--rain-threshold', type=float, default=0.5)
    parser.add_argument('--output-suffix', default='reconstructed_inference',
                         help='Subfolder under the experiment dir to write outputs to, '
                              'so the original homo_3 plots/logs are left untouched')
    parser.add_argument('--folds', type=int, nargs='+', default=None)
    parser.add_argument('--knn', type=int, default=5,
                         help='Gauge-gauge KNN used by train.py at the time (hardcoded to 5)')
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.rstrip('/')
    base_name = experiment_dir[len('experiments/'):] if experiment_dir.startswith('experiments/') else experiment_dir
    output_experiment_name = f"{base_name}/{args.output_suffix}" if args.output_suffix else base_name

    run_inference(
        experiment_dir=experiment_dir,
        config_path=args.config,
        rain_threshold=args.rain_threshold,
        output_experiment_name=output_experiment_name,
        folds=args.folds,
        knn=args.knn,
    )


if __name__ == '__main__':
    main()
