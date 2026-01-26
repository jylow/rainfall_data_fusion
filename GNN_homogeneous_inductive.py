from src.sampling.main import stratified_spatial_kfold_dual

import torch
from torch_geometric.data import Data
from src.raingauge.utils import (
    get_station_coordinate_mappings,
    load_raingauge_dataset,
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
uptime_treshold = 0.9
weather_station_df = load_raingauge_dataset(f'database/{config['dataset_parameters']['raingauge_file']}')


#______________
#optional: get rid of excess no rain event
weather_station_filtered_df = weather_station_df.fillna(0)
weather_station_filtered_df = (weather_station_filtered_df != 0).sum(axis=1)
has_rain = weather_station_filtered_df > 0
window_size = pd.Timedelta(hours=3)

rain_indices = weather_station_df.index[has_rain]
keep_mask = pd.Series(False, index=weather_station_df.index)

for rain_time in rain_indices:
    # Mark all timestamps within 3 hours before and after as "keep"
    mask = (weather_station_df.index >= rain_time - window_size) & \
           (weather_station_df.index <= rain_time + window_size)
    keep_mask = keep_mask | mask

weather_station_df = weather_station_df[keep_mask]

#_________

weather_station_mappings = {}

# 2. Load weather station mappings
for i in range(3):
  year = 2023 + i
  year_stations = get_station_coordinate_mappings(f"database/raingauge_nea_data/{year}/weather_stations_{year}.csv")
  print(year_stations)
  weather_station_mappings |= year_stations

rainfall_stations = weather_station_df.columns
general_station = []
print(rainfall_stations)

# 3. Filter for weather stations that have uptime of {Treshold} %
weather_station_df_uptime = weather_station_df.notna().sum() / len(weather_station_df)
filtered_stations_df = weather_station_df_uptime[weather_station_df_uptime >= uptime_treshold]
rainfall_stations = filtered_stations_df.keys()
print(len(rainfall_stations))

# 3.1 Also filter weather station mappings
weather_station_mappings = {k: v for k, v in weather_station_mappings.items() if k in rainfall_stations}
print("________")

weather_station_df = weather_station_df.resample("15min").first()
weather_station_df.fillna(0, inplace=True)

print("--- Station Data Stats ---")
print(weather_station_df.describe())


# ## Get list of station_ids

# In[4]:


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

print("SHAPE OF RAINGAUGE DATA")
print(np.array(rainfall_station_features).shape)


# # Stratified K Fold Spatial Sampling

# In[5]:


split_info = stratified_spatial_kfold_dual(
    weather_station_mappings, seed=123, plot=True, n_splits = fold_count
)
print(split_info)


# ## Add Station Features to HeteroData Class
# Note: Currently we are only using rainfall values so there is no need for general stations

# In[6]:


print(list(weather_station_mappings))


# In[7]:


rainfall_station_data_tensor = torch.tensor(rainfall_station_features)
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


# ##

# ## Graph Generation
# We will generate the following graphs.
# 1. Training graph: Train Nodes
# 2. Validation graph: Train + Val nodes
# 3. Testing graph: Train + Val + Test Nodes
# 
# The graphs will be generated independently. 
# Each graph will be connected with the K nearest neighbours where K = 4 such that the graph will be fully connected.
# 
# Note: Future graph creations with methods like epsilon ball radius and fully connected graphs can also be explored

# # Creating the GNN

# In[8]:


hidden_channels = 4
in_channels = 1
out_channels = 1
num_layers = 8
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


# In[9]:


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    verbose=False,
    log_file="training_gnn_new_debug.log",
    random_noise_masking=False,
    scheduler=None,
):
    """
    Corrected training loop with gradient debugging.
    """
    model.train()
    epoch_losses = []
    charge_bar = tqdm.tqdm(dataloader, desc="training")

    for batch_idx, batch in enumerate(charge_bar):

        optimizer.zero_grad()

        # PyG Batch object - move to device
        batch = batch.to(device)

        # Extract from PyG Batch format
        x = batch.x  # [B*N, F]
        y = batch.y  # [B*N, Tgt]
        #mask = batch.mask  # [N] - PROBLEM: single mask for one graph
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if batch.edge_attr is not None else None
        num_graphs = batch.num_graphs
        num_nodes = x.shape[0] // num_graphs

        batch_loss = torch.tensor(0.0, device=device)
        for node_pos in range(num_nodes):
            x_masked = x.clone()
            indices_to_mask = torch.arange(num_graphs, device=device) * x.shape[0] // num_graphs + node_pos
            x_masked[indices_to_mask] = 0.0
            out = model(x_masked, edge_index, edge_attributes=edge_attr)

            # Compute loss ONLY on trainable nodes
            loss = F.mse_loss(out[indices_to_mask], y[indices_to_mask])
            batch_loss += loss

            # Check gradients before backward
            #loss.backward()

            # Check if any gradients were computed
            # total_grad_norm = 0.0
            # num_params_with_grad = 0
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm().item()
            #         total_grad_norm += grad_norm**2
            #         num_params_with_grad += 1
            #         if verbose and grad_norm > 1e-6:
            #             print(f"  {name}: grad_norm={grad_norm:.2e}")
            #     elif verbose or batch_idx == 0:
            #         print(f"  {name}: NO GRADIENT")

            # total_grad_norm = np.sqrt(total_grad_norm)

            # if num_params_with_grad == 0:
            #     print(f"ERROR: No gradients computed in batch {batch_idx}!")
            #     return None

            # if total_grad_norm < 1e-8 and batch_idx % 20 == 0:
            #     print(
            #         f"WARNING: Very small gradient norm {total_grad_norm:.2e} in batch {batch_idx}"
            #     )
        batch_loss = batch_loss / num_nodes
        if scheduler is not None:
            scheduler.step()
        batch_loss.backward()


        epoch_losses.append(batch_loss.item())
        charge_bar.set_postfix(
            {
                "loss": batch_loss.item(),
                #"grad_norm": total_grad_norm,
            }
        )
        #Step only at the end of each batch
        optimizer.step()

    return float(np.mean(epoch_losses))


# In[10]:


def validate(
    model, dataloader, device, verbose=False, log_file="validation_gnn_new_debug.log"
):
    """
    Validation loop for PyG batched graph data (inductive setting).

    Key aspects:
    1. Data comes as PyG Batch objects
    2. Features are [B*N, F], already batched and flattened
    3. Mask is [N] - single mask for one graph, replicated across batch
    4. Computes metrics ONLY on validation nodes (where mask=True)
    5. No gradients computed - eval mode
    """
    model.eval()
    epoch_losses = []
    all_preds = []
    all_targets = []

    charge_bar = tqdm.tqdm(dataloader, desc="validation")

    with torch.no_grad():
        for batch in charge_bar:
            # PyG Batch object - move to device
            batch = batch.to(device)

            # Extract from PyG Batch format
            x = batch.x  # [B*N, F] - already batched and flattened
            y = batch.y  # [B*N, Tgt] - already batched and flattened
            val_mask = batch.mask.bool()  # [N] - single mask for one graph
            edge_index = batch.edge_index  # [2, E*B] - offset edge indices
            edge_attr = batch.edge_attr if batch.edge_attr is not None else None
            masked_x = x.clone()
            masked_x[val_mask] = 0.0

            # Forward pass
            out = model(x, edge_index, edge_attributes=edge_attr)  # [B*N, out_channels]

            # Compute loss ONLY on validation nodes
            val_mask = batch.mask # [B*N] boolean mask

            loss = F.mse_loss(out[val_mask], y[val_mask])
            epoch_losses.append(loss.item())

            # Store predictions and targets for metric computation
            all_preds.append(out[val_mask].detach().cpu())
            all_targets.append(y[val_mask].detach().cpu())

            charge_bar.set_postfix({"loss": loss.item()})

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)  # [Total_val_nodes, out_channels]
    all_targets = torch.cat(all_targets, dim=0)  # [Total_val_nodes, out_channels]

    # Compute metrics
    mean_loss = float(np.mean(epoch_losses))

    return mean_loss


# ## Training logic

# In[ ]:


# set seeds

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
perf.log_model_config(model_arr[0].config)

batch_size = 16
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
    first_param = next(model.parameters())
    print(f"Initial weight sample: {first_param.data.flatten()[:5]}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    training_loss_arr = []
    validation_loss_arr = []
    early = 0
    mini = 1000
    stopping_condition = 5
    epochs = 0
    total_epochs = 10
    print(f"-----FOLD: {fold}-----")
    training_start = time.time()
    for i in range(total_epochs):
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


# In[ ]:


sids = gauge_graph_arr[0].get_test_graph().station_id
print(sids)
masked_ids = sids[gauge_graph_arr[0].get_test_graph().test_mask]
masked_ids


# In[ ]:


def test_model(model, dataloader, device, fold=0, verbose=False):
    """
    Test loop following the SAME structure as validate():
      - PyG batch format
      - x, y shaped [B*N, F]
      - mask shaped [B*N]
      - station_id shaped [B*N]  (added)
      - Computes metrics ONLY on test nodes
    """

    model.eval()

    all_preds = []
    all_targets = []
    all_station_ids = []   # <-- FIXED: collect all station IDs here
    epoch_losses = []

    test_bar = tqdm.tqdm(dataloader, desc="Testing")

    with torch.no_grad():
        for batch in test_bar:
            batch = batch.to(device)

            # DEBUG: print available batch attributes once
            if not hasattr(test_model, "_printed_batch_info"):
                print("\n=== DEBUG: Batch Object Attributes ===")
                print(batch)
                print("Dir(batch):")
                print([attr for attr in dir(batch) if not attr.startswith("_")])

                # Check station ID fields
                print("\n=== DEBUG: Checking for station ID fields ===")
                if hasattr(batch, "station_id"):
                    print(f"FOUND: batch.station_id → shape {batch.station_id.shape}")
                else:
                    print("ERROR: batch.station_id not found!")

                # Print tensor attributes
                print("\n=== DEBUG: Tensor attributes found in batch.__dict__ ===")
                for k, v in batch.__dict__.items():
                    if torch.is_tensor(v):
                        print(f"{k}: shape = {tuple(v.shape)}")

                test_model._printed_batch_info = True

            # ----- Extract inputs from batch -----
            x = batch.x
            y = batch.y
            mask = batch.mask.bool()
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr if batch.edge_attr is not None else None
            station_id = batch.station_id               # <--- REQUIRED

            assert mask.shape[0] == x.shape[0], "Mask and x size mismatch"
            x_masked = x.clone()
            x_masked[mask] = 0.0

            # ----- Model forward -----
            out = model(x, edge_index, edge_attributes=edge_attr)

            # ----- Compute test loss -----
            loss = F.mse_loss(out[mask], y[mask])
            epoch_losses.append(loss.item())

            # ----- Collect outputs -----
            all_preds.append(out[mask].detach().cpu())
            all_targets.append(y[mask].detach().cpu())
            all_station_ids.append(station_id[mask].detach().cpu())   # <-- FIXED

            test_bar.set_postfix({"loss": loss.item()})

    # ============================================================
    # === CONCATENATE EVERYTHING
    # ============================================================
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_station_ids = torch.cat(all_station_ids, dim=0)   # <-- FIXED

    print("Final aggregated prediction shape:", all_preds.shape)
    print("Final aggregated target shape:", all_targets.shape)
    print("Final aggregated station_id shape:", all_station_ids.shape)

    unique_stations = all_station_ids.unique().tolist()
    print("Total stations in test set:", len(unique_stations))

    # ============================================================
    # === GLOBAL METRICS
    # ============================================================
    preds_np = all_preds.numpy().flatten()
    targets_np = all_targets.numpy().flatten()

    valid_mask = (~np.isnan(preds_np)) & (~np.isnan(targets_np))
    pearson_r, pearson_p = pearsonr(targets_np[valid_mask], preds_np[valid_mask])

    mse = ((all_preds - all_targets) ** 2).mean()
    rmse = torch.sqrt(mse).item()

    print(f"Pearson correlation (Test Nodes): {pearson_r}")
    print(f"Final Test RMSE: {rmse}")

    # ============================================================
    # === GLOBAL SCATTER
    # ============================================================
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_np, preds_np, alpha=0.5)
    max_v = max(np.nanmax(preds_np), np.nanmax(targets_np))
    plt.plot([0, max_v], [0, max_v], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Test Set Performance")
    plt.grid(True)
    text = f"Pearson r = {pearson_r:.3f}\nRMSE = {rmse:.3f}" 
    plt.text( 0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"), )
    plt.savefig(f"experiments/{experiment_name}/test_scatter_plot_{fold}.png", dpi=300)
    plt.close()

    # ============================================================
    # === PER-STATION PLOTS
    # ============================================================
    save_dir = f"experiments/{experiment_name}/per_station_plots_f{fold}"
    os.makedirs(save_dir, exist_ok=True)

    for sid in unique_stations:
        mask_sid = (all_station_ids == sid)

        preds_sid = all_preds[mask_sid].numpy().flatten()
        targets_sid = all_targets[mask_sid].numpy().flatten()

        if len(preds_sid) < 5:
            continue

        # ----- Scatter -----
        plt.figure(figsize=(7, 7))
        plt.scatter(targets_sid, preds_sid, alpha=0.6)
        max_val = max(preds_sid.max(), targets_sid.max())
        plt.plot([0, max_val], [0, max_val], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Station {sid} — Actual vs Predicted")
        plt.grid(True)
        plt.savefig(f"{save_dir}/station_{sid}_scatter.png", dpi=250)
        plt.close()

        # ----- Time series -----
        plt.figure(figsize=(15, 6))
        plt.plot(targets_sid, label="Actual")
        plt.plot(preds_sid, label="Predicted")
        plt.title(f"Station {sid} — Time Series")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/station_{sid}_timeseries.png", dpi=250)
        plt.close()

    print(f"Saved per-station plots in {save_dir}")

    return rmse



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
    RMSE = test_model(model_arr[i], test_loader_arr[i], device, fold=i)
    print(f"TEST RMSE: {RMSE}")
