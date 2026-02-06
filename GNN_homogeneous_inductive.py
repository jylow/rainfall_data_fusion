from src.sampling.main import stratified_spatial_kfold_dual

import torch
from torch_geometric.data import Data
from src.raingauge.utils import (
    get_station_coordinate_mappings,
    load_raingauge_dataset,
    get_station_mapping_df,
    filter_uptime
)
import pandas as pd
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt
import time
import yaml
from scipy.stats import pearsonr
import matplotlib as mpl
from models.gnn import GNNInductive
from datetime import datetime
from src.performance_logger import PerformanceLogger
import os
from src.utils import (
    add_homogeneous_weather_station_data,
    add_homogeneous_mask_to_data,
    prepare_homogeneous_inductive_dataset,
)

from src.graph.gaugegraph import GaugeGraph
import torch.nn.functional as F

from training.logic import train_epoch, validate, test_model


# NOTE: Geographic extent of Singapore in longitude and latitude
bounds_singapore = {"left": 103.6, "right": 104.1, "top": 1.5, "bottom": 1.188}
bounds = [0.1, 0.2, 0.5, 1, 2, 4, 7, 10, 20]
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")

experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_new"
os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
perf = PerformanceLogger(f"experiments/{experiment_name}/training_log.jsonl")

#Read config file
config_file = 'config.yaml'
with open(config_file) as f:
    config = yaml.safe_load(f)


# # Preprocess station data.
#
# 1. Load weather station information
# 2. Load weather station mappings
# 3. Filter weather stations by uptime

# 1. Load weather station information
uptime_treshold = config['filter']['uptime_treshold']
start_year = config['dataset_parameters']['start_year']
end_year = config['dataset_parameters']['end_year']
weather_station_df = load_raingauge_dataset(f'database/{config['dataset_parameters']['raingauge_file']}')
weather_station_mappings_df = get_station_mapping_df(start_year, end_year)
weather_station_mappings = get_station_coordinate_mappings("", start = 2021, end = 2025)


#optional: get rid of excess no rain event
weather_station_filtered_cols = filter_uptime(weather_station_df).index
weather_station_filtered_df = weather_station_df[weather_station_filtered_cols]
# weather_station_filtered_df = weather_station_df.fillna(0)
# weather_station_filtered_df = (weather_station_filtered_df != 0).sum(axis=1)
# has_rain = weather_station_filtered_df > 0
# window_size = pd.Timedelta(hours=3)

# rain_indices = weather_station_df.index[has_rain]
# keep_mask = pd.Series(False, index=weather_station_df.index)

# for rain_time in rain_indices:
#     # Mark all timestamps within 3 hours before and after as "keep"
#     mask = (weather_station_df.index >= rain_time - window_size) & \
#            (weather_station_df.index <= rain_time + window_size)
#     keep_mask = keep_mask | mask

# weather_station_df = weather_station_df[keep_mask]

#_________

weather_station_mappings = {}


rainfall_stations = weather_station_df.columns
general_station = []

# 3.1 Also filter weather station mappings
weather_station_mappings = {k: v for k, v in weather_station_mappings.items() if k in rainfall_stations}
print("________")

weather_station_df = weather_station_df.resample("15min").first()
weather_station_df.fillna(0, inplace=True)

print("--- Station Data Stats ---")
print(weather_station_df.describe())




general_station_data = {}
rainfall_station_data = {}
dtype = torch.float32
fold_count = 5

# Prepare features in the correct order
general_station_features = []
rainfall_station_features = []
general_station_order = []
rainfall_station_order = [] # IMPORTANT TO KEEP TRACK OF ORDERING

for station in rainfall_stations:
    station_feat = weather_station_df[station]
    rainfall_station_features.append(station_feat)
    rainfall_station_order.append(station)



split_info = stratified_spatial_kfold_dual(
    weather_station_mappings, seed=123, plot=False, n_splits = fold_count
)
print(split_info)

rainfall_station_data_tensor = torch.tensor(
    np.stack([s.values for s in rainfall_station_features]),
    dtype=torch.float32
)
gauge_graph_arr = []
for i in range(fold_count):
  data = add_homogeneous_weather_station_data(
        Data(), # empty pygeometric Data type
        general_station_features = None,
        rainfall_station_features=rainfall_station_data_tensor,
        general_station_ids = None,
        rainfall_station_ids = rainfall_station_order,
        dtype=dtype,
    )

  data = add_homogeneous_mask_to_data(data, split_info[i], rainfall_stations)
  data.x = data.x.unsqueeze(-1)
  data.y = data.y.unsqueeze(-1)
  print(data)
  gauge_graph_arr.append(GaugeGraph(
    data = data,
    station_dict=weather_station_mappings,
    split_info=split_info[i],
    raingauge_station_order = rainfall_station_order,
    knn = 4
  ))


print("INITIALISING THE GRAPH")

hidden_channels = 8
in_channels = 1
out_channels = 1
num_layers = 5
model_arr = []
device = "cuda" if torch.cuda.is_available() else "cpu"

for i in range(fold_count):
    model_arr.append(
        GNNInductive(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        ).to(device=device)
    )



# set seeds

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
perf.log_model_config(model_arr[0].config)

batch_size = 512
train_loader_arr = []
val_loader_arr = []
for i in range(fold_count):
    gauge_graph = gauge_graph_arr[i]
    train_loader, val_loader = prepare_homogeneous_inductive_dataset(
        gauge_graph.get_train_graph(), gauge_graph.get_validation_graph(), batch_size=batch_size, mode="train"
    )
    train_loader_arr.append(train_loader)
    val_loader_arr.append(val_loader)


def train(model, train_loader, val_loader, fold, device="cpu"):
    # CHECK 1: Print initial weights
    print("Training")
    print(f"Device type: {device}")
    first_param = next(model.parameters())
    print(f"Initial weight sample: {first_param.data.flatten()[:5]}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    training_loss_arr = []
    validation_loss_arr = []
    early = 0
    mini = 1000
    stopping_condition = 5
    epochs = 0
    total_epochs = 30
    print(f"-----FOLD: {fold}-----")
    training_start = time.time()
    for i in range(total_epochs):
        epoch_start = time.time()
        print(f"-----EPOCH: {i + 1}-----")

        # CHECK 2: Print weight before training
        weight_before = first_param.data.clone()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            verbose=False,
            random_noise_masking=False,
        )
        print(train_loss)

        # CHECK 3: Print weight after training
        weight_after = first_param.data
        weight_change = (weight_after - weight_before).abs().mean().item()
        print(f"Weight change: {weight_change:.20f}")

        validation_loss = validate(model, val_loader, device)
        training_loss_arr.append(train_loss)
        validation_loss_arr.append(validation_loss)
        perf.log_epoch(i, train_loss, validation_loss)
        if mini >= validation_loss:
            mini = validation_loss
            early = 0
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

    torch.save(
        model.state_dict(), f"experiments/{experiment_name}/weather_gnn_best_{fold}.pth"
    )
    print("✅ model weights saved to weather_gnn_best.pth")

    perf.log_model_parameters(model)
    return model


for i in range(fold_count):
    model = train(model_arr[i], train_loader_arr[i], val_loader_arr[i], fold=i, device=device)



test_loader_arr = []
for i in range(fold_count):
    gauge_graph = gauge_graph_arr[i]
    test_loader = prepare_homogeneous_inductive_dataset(
        gauge_graph.get_train_graph(),
        gauge_graph.get_validation_graph(),
        gauge_graph.get_test_graph(),
        batch_size=batch_size,
        mode="test",
    )
    test_loader_arr.append(test_loader)

for i in range(fold_count):
    RMSE = test_model(model_arr[i], test_loader_arr[i], device, fold=i, experiment_name=experiment_name)
    print(f"TEST RMSE: {RMSE}")
