from src.sampling.main import stratified_spatial_kfold_dual #Dont know why but this has to be initialised first else kernel crashes

import torch
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from torch_geometric.loader import DataLoader as GeometricDataLoader

from src.performance_logger import PerformanceLogger
from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.utils import (
  load_raingauge_dataset
)
from src.radar.utils import load_radar_dataset, load_processed_dataset
from src.cml.utils import load_cml_dataset
from training.logic_hetero import train_epoch, validate, test_model
from src.graph.cmlgraph import CMLGraph
from src.graph.radargraph import RadarGraph
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive


# 1. Load the config file
def train_fused(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['training_params']['batch_size']
    fold_count = config['training_params']['fold_count']
    datasources = config['datasources']


# 2. Set up experiment folder
    experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_new"
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    perf = PerformanceLogger(f"experiments/{experiment_name}/training_log.jsonl")


# 3. Load data
    uptime_threshold = config['filters']['uptime_threshold']
    start_year = config['dataset_parameters']['start_year']
    end_year = config['dataset_parameters']['end_year']
    raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(start = start_year, end = end_year, uptime_threshold=uptime_threshold)
    radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")
#radar_df = load_radar_dataset(folder_name='database/sg_radar_data_cropped', cropped=True)


# 4. INNER JOIN tables based on timestamp
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
    raingauge_df = pd.concat([merged_df['timestamp'], merged_df[raingauge_cols]], axis=1).drop_duplicates().reset_index(drop=True)
    radar_df = radar_df.drop_duplicates(subset=['timestamp'], keep='first')

    cml_df = cml_df.drop_duplicates()

    print(raingauge_df.shape)
    print(radar_df.shape)
    print(cml_df.shape)

# 5. Stratified sampling
    split_info = stratified_spatial_kfold_dual(
        raingauge_station_mappings_df, seed=123, plot=False, n_splits = fold_count
    )


# 6. Build graphs
    gauge_graph_arr = []
    for i in range(fold_count):
      gauge_graph = GaugeGraphNew(raingauge_df, raingauge_station_mappings_df, split_info = split_info[i], knn=config['layer_connect']['gauge_gauge'])
      if 'radar' in datasources:
          radar_graph = RadarGraph(radar_df)
          radar_heterodata = radar_graph.get_radar_heterodata()
          gauge_graph.add_heterodata(heterodata_layer=radar_heterodata, coords = radar_graph.grid_coords, layer_name='radar', knn=config['layer_connect']['radar_gauge'])
      if 'cml' in datasources:
        cml_graph = CMLGraph(cml_df, cml_coordinates_df)
        cml_heterodata = cml_graph.get_heterodata()
        gauge_graph.add_heterodata(heterodata_layer=cml_heterodata, coords=cml_coordinates_df, layer_name='cml', knn=config['layer_connect']['cml_gauge'])
      gauge_graph_arr.append(gauge_graph)


# 7. Initialise HGNN model
    if 'cml' in datasources:
        cml_features = gauge_graph_arr[0].get_train_heterodata()['cml'].x.shape[2]
    hidden_channels = config['model']['hidden_channels']
    out_channels = 1
    num_layers = config['model']['num_layers']
    model_arr = []
    raingauge_features = 6 if config['dataset_parameters']['include_lpe'] else 2
    for i in range(fold_count):
      if 'cml' in datasources:
          model_arr.append(
            GNNInductiveHetero(
              in_channels_dict = {
                "raingauge": raingauge_features,
                "radar": 1,
                "cml": cml_features
              },
              hidden_channels = hidden_channels,
              out_channels=out_channels,
              num_layers = num_layers,
              edge_types = gauge_graph_arr[i].get_train_heterodata().edge_types
            ).to(device=device)
          )
      else:
          model_arr.append(
            GNNInductiveHetero(
              in_channels_dict = {
                "raingauge": raingauge_features,
                "radar": 1,
              },
              hidden_channels = hidden_channels,
              out_channels=out_channels,
              num_layers = num_layers,
              edge_types = gauge_graph_arr[i].get_train_heterodata().edge_types
            ).to(device=device)
          )

#8. Batch the data
    def compute_norm_stats(heterodata):
        """Compute per-feature mean and std from one split's node features."""
        stats = {}
        for node_type in heterodata.node_types:
            x = heterodata[node_type].x  # [N, T, F]
            mean = x.mean(dim=(0, 1))
            std = x.std(dim=(0, 1)).clamp(min=1e-8)
            stats[node_type] = (mean, std)
        return stats

    def apply_norm(heterodata, stats):
        """Apply precomputed stats to a heterodata object. Only touches .x, never .y."""
        normed = heterodata.clone()
        for node_type in heterodata.node_types:
            if node_type in stats:
                mean, std = stats[node_type]
                normed[node_type].x = (heterodata[node_type].x - mean) / std
        return normed

    train_loader_arr = []
    val_loader_arr = []
    test_loader_arr = []
    for i in range(fold_count):
        train_data = gauge_graph_arr[i].get_train_heterodata()
        val_data   = gauge_graph_arr[i].get_validation_heterodata()
        test_data  = gauge_graph_arr[i].get_test_heterodata()

        # Compute stats from training split only, apply same stats to val and test
        stats = compute_norm_stats(train_data)

        train_loader = GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(apply_norm(train_data, stats)),
            batch_size=batch_size,
            shuffle=False,
        )
        val_loader = GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(apply_norm(val_data, stats)),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = GeometricDataLoader(
            HeterogeneousWeatherGraphDatasetInductive(apply_norm(test_data, stats)),
            batch_size=batch_size,
            shuffle=False,
        )

        train_loader_arr.append(train_loader)
        val_loader_arr.append(val_loader)
        test_loader_arr.append(test_loader)



    def train_fold(model, train_loader, val_loader, fold, device="cpu"):
        # CHECK 1: Print initial weights
        #torch.autograd.set_detect_anomaly(True)
        print("Training")
        print(f"Device type: {device}")
        first_param = next(model.parameters())
        print(f"Initial weight sample: {first_param.data.flatten()[:5]}")

        optimizer = torch.optim.Adam(model.parameters(), lr=(config['model']['learning_rate']), weight_decay=float(config['model']['weight_decay']))
        training_loss_arr = []
        validation_loss_arr = []
        early = 0
        mini = float('inf')
        stopping_condition = config['training_params']['early_stop']
        epochs = 0
        total_epochs = config['training_params']['epochs']
        print(f"-----FOLD: {fold}-----")
        training_start = time.time()
        for i in range(total_epochs):
            epoch_start = time.time()
            print(f"-----EPOCH: {i + 1}-----")

            # CHECK 2: Print weight before training
            #weight_before = first_param.data.clone()

            weighted_alpha = config['training_params'].get('weighted_loss_alpha', 0.0)
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                weighted_loss_alpha=weighted_alpha,
            )
            print(train_loss)

            validation_loss = validate(model, val_loader, device, weighted_loss_alpha=weighted_alpha)
            training_loss_arr.append(train_loss)
            validation_loss_arr.append(validation_loss)
            perf.log_epoch(i, train_loss, validation_loss)
            if mini >= validation_loss:
                mini = validation_loss
                early = 0
                torch.save(
                    model.state_dict(), f"experiments/{experiment_name}/weather_gnn_best_{fold}.pth"
                )
                print("✅ model weights saved to weather_gnn_best.pth")
            else:
                early += 1
            epochs += 1
            if early >= stopping_condition:
                print("Early stop loss")
                break

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")

            # CHECK 4: Print gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm**0.5
            print(f"Gradient norm: {total_norm:.6f}")
            epoch_end = time.time()
            print(f"epoch {i} took {epoch_end - epoch_start}")
        training_end = time.time()
        total_time = training_end - training_start
        perf.finalise(total_time)

        print(f"Training took {total_time} seconds over {epochs} epochs")
        plt.plot(training_loss_arr, label="training_loss", color="blue")
        plt.plot(validation_loss_arr, label="validation_loss", color="red")
        plt.legend()
        plt.savefig(f"experiments/{experiment_name}/train_loss_plot_{fold}.png", dpi=300)
        plt.close()


    fold_metrics = []
    for i in range(fold_count):
      train_fold(model_arr[i], train_loader=train_loader_arr[i], val_loader=val_loader_arr[i], fold = i, device=device)

      # load the best model (not the last model found)
      model_arr[i].load_state_dict(
        torch.load(f"experiments/{experiment_name}/weather_gnn_best_{i}.pth", map_location=device)
      )
      metrics_dict = test_model(model_arr[i], raingauge_station_mappings_df, test_loader_arr[i], device, fold = i, experiment_name=experiment_name)
      fold_metrics.append(metrics_dict)

    averaged_metrics_dict = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in ["rmse", "mae", "pearson_r", "timestep_rmse",
                    "precision", "recall", "f1"]
    }

    return averaged_metrics_dict


config = read_config("config.yaml")
train_fused(config=config)
