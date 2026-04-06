from src.sampling.main import stratified_spatial_kfold_dual  # must be first

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_radar_dataset
from src.cml.utils import load_cml_dataset
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive
from src.graph.radargraph import RadarGraph
from src.graph.cmlgraph import CMLGraph
from src.visualization.error_analysis import (
    analyze_station_errors,
    plot_spatial_error_map,
    plot_error_ranking,
    plot_bias_map,
    plot_error_vs_isolation,
)
from src.visualization.grid_prediction import (
    predict_on_grid,
    plot_rainfall_grid,
    plot_rainfall_sequence,
)

# ── Config ────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "raingauge_radar_cml_5layer"
FOLD_IDX        = 0

config     = read_config("config.yaml")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOUNDS     = config["dataset_parameters"]["geography_bounds"]
FOLD_COUNT = config["training_params"]["fold_count"]
layer_cfg  = config["layer_connect"]

EXPERIMENT_DIR = f"experiments/{EXPERIMENT_NAME}"
WEIGHTS_PATH   = f"{EXPERIMENT_DIR}/weather_gnn_best_{FOLD_IDX}.pth"

# ── Load datasets ─────────────────────────────────────────────────────────────
raingauge_df, mapping_df = load_raingauge_dataset(
    start=config["dataset_parameters"]["start_year"],
    end=config["dataset_parameters"]["end_year"],
    uptime_threshold=config["filters"]["uptime_threshold"],
)
raingauge_df = raingauge_df.fillna(0)
radar_df     = load_radar_dataset(folder_name="database/sg_radar_data_cropped", cropped=True)
cml_df, cml_coordinates_df = load_cml_dataset(config["dataset_parameters"]["cml_folder"])

# Align timestamps across all three sources
merged_df    = radar_df.merge(raingauge_df, how="inner", on=["timestamp"]).merge(cml_df, how="inner", on=["timestamp"])
cml_df       = merged_df[cml_df.columns].drop_duplicates()
radar_df     = merged_df[radar_df.columns].drop_duplicates(subset=["timestamp"], keep="first")
raingauge_df = (
    pd.concat([merged_df["timestamp"], merged_df[raingauge_df.columns]], axis=1)
    .drop_duplicates().reset_index(drop=True).drop(columns=["timestamp"])
)

# ── Build graphs ──────────────────────────────────────────────────────────────
split_info = stratified_spatial_kfold_dual(
    mapping_df, seed=config["training_params"]["seed"], plot=False, n_splits=FOLD_COUNT
)

gauge_graph_arr = []
for i in range(FOLD_COUNT):
    g = GaugeGraphNew(raingauge_df, mapping_df, split_info=split_info[i], knn=layer_cfg["gauge_gauge"])
    radar_graph = RadarGraph(radar_df)
    cml_graph   = CMLGraph(cml_df, cml_coordinates_df)
    g.add_heterodata(heterodata_layer=radar_graph.get_radar_heterodata(), coords=radar_graph.grid_coords, layer_name="radar", knn=layer_cfg["radar_gauge"])
    g.add_heterodata(heterodata_layer=cml_graph.get_heterodata(),         coords=cml_coordinates_df,      layer_name="cml",   knn=layer_cfg["cml_gauge"])
    gauge_graph_arr.append(g)

# ── Load model ────────────────────────────────────────────────────────────────
def _infer_arch(weights_path):
    sd = torch.load(weights_path, map_location="cpu")
    conv_keys    = [k for k in sd if k.startswith("convs.")]
    num_layers   = max(int(k.split(".")[1]) for k in conv_keys) + 1
    hidden_channels = sd["lin.weight"].shape[1]
    return num_layers, hidden_channels

test_heterodata         = gauge_graph_arr[FOLD_IDX].get_test_heterodata()
num_layers, hidden_channels = _infer_arch(WEIGHTS_PATH)

model = GNNInductiveHetero(
    in_channels_dict={"raingauge": -1, "radar": -1, "cml": -1},
    hidden_channels=hidden_channels,
    out_channels=1,
    num_layers=num_layers,
    edge_types=test_heterodata.edge_types,
).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# ── Part 1: Error analysis ────────────────────────────────────────────────────
csv_paths  = sorted(glob.glob(f"{EXPERIMENT_DIR}/per_station_metrics_f*.csv"))
train_ids  = split_info[FOLD_IDX]["ml"]["train"]

station_df = analyze_station_errors(
    csv_paths=csv_paths,
    mapping_df=mapping_df,
    bounds=BOUNDS,
    train_station_ids=train_ids,
    output_dir=f"{EXPERIMENT_DIR}/analysis",
    top_n=10,
)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plot_spatial_error_map(station_df, BOUNDS, metric="mae",  ax=axes[0])
plot_spatial_error_map(station_df, BOUNDS, metric="rmse", ax=axes[1])
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
plot_bias_map(station_df, BOUNDS, top_n_labels=10, ax=ax)
plt.tight_layout(); plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plot_error_ranking(station_df, metric="mae", top_n=15, ax=ax1)
plot_error_ranking(station_df, metric="f1",  top_n=15, ax=ax2)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
plot_error_vs_isolation(station_df, train_ids, metric="mae", ax=ax)
plt.tight_layout(); plt.show()

# ── Part 2: Grid prediction ───────────────────────────────────────────────────
predictions, grid_coords, grid_shape = predict_on_grid(
    model=model,
    heterodata=test_heterodata,
    mapping_df=mapping_df,
    bounds=BOUNDS,
    resolution_km=1.0,
    knn_gauge=5,
    device=str(device),
    include_lpe=config["dataset_parameters"].get("include_lpe", True),
    lpe_k=4,
)

T = predictions.shape[0]

# Single timestep
fig, ax = plt.subplots(figsize=(10, 8))
plot_rainfall_grid(predictions, grid_shape, BOUNDS, timestamp_idx=0, mapping_df=mapping_df, ax=ax)
plt.tight_layout(); plt.show()

# Sequence of 6 evenly-spaced timesteps
fig = plot_rainfall_sequence(
    predictions, grid_shape, BOUNDS,
    timestep_indices=np.linspace(0, T - 1, min(6, T), dtype=int).tolist(),
    mapping_df=mapping_df,
    save_path=f"{EXPERIMENT_DIR}/analysis/grid_sequence.png",
)
plt.show()

# Mean over test period
mean_pred = predictions.mean(axis=0)[np.newaxis]
fig, ax = plt.subplots(figsize=(10, 8))
plot_rainfall_grid(mean_pred, grid_shape, BOUNDS, timestamp_idx=0, mapping_df=mapping_df,
                   title="Mean predicted rainfall over test period", ax=ax)
plt.tight_layout(); plt.show()
