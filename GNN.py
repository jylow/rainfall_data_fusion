#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

from src.sampling import stratified_spatial_sampling_dual
from src.utils import *
from dataset.weather_graph_dataset import WeatherGraphDataset

import torch
from torch_geometric.data import HeteroData


# In[ ]:


weather_station_data = load_weather_station_dataset("weather_station_data.csv")
weather_station_locations = get_station_coordinate_mappings()


# # Preprocess station data.
# Some stations only contain rainfall information but some stations contain both rainfall and other information.
# We will split these stations into weather station and general stations
#
# Additional info:
#   Windspeed
#   Wind Direction
#   Temperature
#   Relative Humidity

# In[ ]:


cols = list(weather_station_data.columns)
cols.remove("time_sgt")
cols.remove("gid")
weather_station_df_pivot = pd.pivot(
    data=weather_station_data, index="time_sgt", columns="gid", values=cols
)
weather_station_df_counts = weather_station_df_pivot.count().reset_index()

weather_station_info = pd.pivot(
    data=weather_station_df_counts, index="gid", columns="level_0"
)

pd.set_option("display.max_rows", None)

rainfall_station = [
    row[0] for row in weather_station_info.iterrows() if 0 in row[1].value_counts()
]
general_station = [s for s in weather_station_locations if s not in rainfall_station]

print(rainfall_station)
print(general_station)


# In[ ]:


pd.set_option("display.max_rows", 20)

general_station_data = {}
rainfall_station_data = {}
for station in weather_station_df_pivot.columns.get_level_values(1).unique():
    station_cols = (
        weather_station_df_pivot.xs(station, level=1, axis=1)
        .interpolate(method="linear")
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
    if station in general_station:
        general_station_data[station] = station_cols.values
    else:
        rainfall_station_data[station] = station_cols.values[:, 0:1]

print(rainfall_station_data)


# In[ ]:


data = HeteroData()

general_station_features = []
rainfall_station_features = []
for station in general_station:
    station_feat = general_station_data[station]
    general_station_features.append(station_feat)

for station in rainfall_station:
    station_feat = rainfall_station_data[station]
    rainfall_station_features.append(station_feat)

dtype = torch.float32
data["general_station"].x = torch.tensor(
    np.array(general_station_features).transpose(1, 0, 2), dtype=dtype
)
data["rainfall_station"].x = torch.tensor(
    np.array(rainfall_station_features).transpose(1, 0, 2), dtype=dtype
)

data["general_station"].y = torch.tensor(
    np.array(general_station_features)[:, :, 0:1].transpose(1, 0, 2), dtype=dtype
)
data["rainfall_station"].y = torch.tensor(
    np.array(rainfall_station_features).transpose(1, 0, 2), dtype=dtype
)

print(data)


# In[ ]:


split_info = stratified_spatial_sampling_dual(weather_station_locations)
print(split_info)

data["general_station"].train_mask = [
    1 if station in split_info["ml"]["train"] else 0 for station in general_station
]
data["general_station"].val_mask = [
    1 if station in split_info["ml"]["validation"] else 0 for station in general_station
]
data["general_station"].test_mask = [
    1 ^ (x | y)
    for x, y in zip(
        data["general_station"].train_mask, data["general_station"].val_mask
    )
]

data["rainfall_station"].train_mask = [
    1 if station in split_info["ml"]["train"] else 0 for station in rainfall_station
]
data["rainfall_station"].val_mask = [
    1 if station in split_info["ml"]["validation"] else 0
    for station in rainfall_station
]
data["rainfall_station"].test_mask = [
    1 ^ (x | y)
    for x, y in zip(
        data["rainfall_station"].train_mask, data["rainfall_station"].val_mask
    )
]

print(data)


# # Edge generation
# We consider the location of the stations when performing our edge generation.
# General station locations and rainfall station locations will be considered the same and we will make a connection across the nodes if required. This will ensure that we can connect both the layers together in the graph.

# In[ ]:


# variable to determine number of neighbours per node
import networkx as nx
from sklearn.neighbors import NearestNeighbors

K = 4

print(weather_station_locations)
ids = list(weather_station_locations.keys())
print(ids)
coords = np.array(list(weather_station_locations.values()))

knn = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree")
knn.fit(coords)

distances, indices = knn.kneighbors(coords)


G = nx.Graph()

edges = {
    "rainfall_to_rainfall": [],
    "rainfall_to_general": [],
    "general_to_rainfall": [],
    "general_to_general": [],
}

for idx, row in enumerate(indices):
    origin = row[0]
    for n in row[1:]:
        if ids[origin] in rainfall_station:
            start_id = rainfall_station.index(ids[origin])
            if ids[n] in rainfall_station:
                end_id = rainfall_station.index(ids[n])
                edges["rainfall_to_rainfall"].append([start_id, end_id])
            else:
                end_id = general_station.index(ids[n])
                edges["rainfall_to_general"].append([start_id, end_id])
        else:
            start_id = general_station.index(ids[origin])
            if ids[n] in rainfall_station:
                end_id = rainfall_station.index(ids[n])
                edges["general_to_rainfall"].append([start_id, end_id])
            else:
                end_id = general_station.index(ids[n])
                edges["general_to_general"].append([start_id, end_id])

G.add_edges_from(edges["rainfall_to_rainfall"])

for key, val in edges.items():
    xarr = []
    yarr = []
    for x, y in val:
        xarr.append(x)
        yarr.append(y)
    edges[key] = [xarr, yarr]


data["general_station", "gen_to_rain", "rainfall_station"].edge_index = torch.tensor(
    edges["general_to_rainfall"], dtype=torch.int64
)
data["rainfall_station", "rain_to_gen", "general_station"].edge_index = torch.tensor(
    edges["rainfall_to_general"], dtype=torch.int64
)
data["general_station", "gen_to_gen", "general_station"].edge_index = torch.tensor(
    edges["general_to_general"], dtype=torch.int64
)
data["rainfall_station", "rain_to_rain", "rainfall_station"].edge_index = torch.tensor(
    edges["rainfall_to_rainfall"], dtype=torch.int64
)

print(data)


# In[ ]:


# # Creating the GNN

# In[ ]:


from torch_geometric.nn import SAGEConv, HeteroConv, GCNConv, Linear


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("general_station", "gen_to_gen", "general_station"): GCNConv(
                        -1, hidden_channels
                    ),
                    ("general_station", "gen_to_rain", "rainfall_station"): SAGEConv(
                        (-1, -1), hidden_channels
                    ),
                    ("rainfall_station", "rain_to_gen", "general_station"): SAGEConv(
                        (-1, -1), hidden_channels
                    ),
                    ("rainfall_station", "rain_to_rain", "rainfall_station"): GCNConv(
                        -1, hidden_channels
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin_rainfall = Linear(hidden_channels, out_channels)
        self.lin_general = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        gen_out = self.lin_general(x_dict["general_station"])
        rain_out = self.lin_rainfall(x_dict["rainfall_station"])

        return {"general_station": gen_out, "rainfall_station": rain_out}


model = HeteroGNN(hidden_channels=16, out_channels=1, num_layers=3)

model.to(device="cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


import tqdm
import torch.functional as F


def collate_temporal_graphs(batch):
    gen_x = torch.stack([item["gen_x"] for item in batch])
    rain_x = torch.stack([item["rain_x"] for item in batch])
    gen_y = torch.stack([item["gen_y"] for item in batch])
    rain_y = torch.stack([item["rain_y"] for item in batch])

    return {"gen_x": gen_x, "rain_x": rain_x, "gen_y": gen_y, "rain_y": rain_y}


def train_epoch(model, data, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    gen_mask = torch.tensor(data["general_station"].train_mask, dtype=torch.bool).to(
        device
    )
    rain_mask = torch.tensor(data["rainfall_station"].train_mask, dtype=torch.bool).to(
        device
    )

    edge_index_dict = {key: val.to(device) for key, val in data.edge_index_dict.items()}
    for batch in tqdm.tqdm(dataloader, desc="training"):
        gen_x = batch["gen_x"].to(device)  # [batch_size, num_gen_nodes, gen_features]
        rain_x = batch["rain_x"].to(
            device
        )  # [batch_size, num_rain_nodes, rain_features]
        gen_y = batch["gen_y"].to(device)
        rain_y = batch["rain_y"].to(device)

        batch_size = gen_x.shape[0]

        batch_loss = 0
        for i in range(batch_size):
            x_dict = {"general_station": gen_x[i], "rainfall_station": rain_x[i]}
            optimizer.zero_grad()
            out = model(x_dict, edge_index_dict)

            gen_predictions = out["general_station"][gen_mask]
            rain_predictions = out["rainfall_station"][rain_mask]

            training_loss = F.mse_loss(
                gen_predictions, gen_y[i][gen_mask]
            ) + F.mse_loss(rain_predictions, rain_y[i][rain_mask])
            training_loss.backward()
            optimizer.step()
            batch_loss += training_loss.item()
        total_loss += batch_loss

    return total_loss / len(dataloader.dataset)


# In[ ]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

batch_size = 128
train_dataset = WeatherGraphDataset(data, mode="train")
val_dataset = WeatherGraphDataset(data, mode="val")
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_temporal_graphs,
)
test_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_temporal_graphs,
)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
training_loss_arr = []
test_loss_arr = []

for i in range(10):
    train_loss = train_epoch(model, data, train_loader, optimizer, device)
    training_loss_arr.append(train_loss)

    print(f"Train Loss: {train_loss:.4f}")

# print(training_loss_arr)
# plt.plot(training_loss_arr, label='validation_loss', color='blue')
# plt.plot(test_loss_arr, label='test_loss', color='red')


# In[ ]:


print(model)


# In[ ]:


print(test_loss_arr[-1])


# # Visualisation of output
# Test event will be 02-05-2025 0415 to 0615
#

# In[ ]:


test_event_data = weather_station_df_pivot.iloc[1772:1798].resample("15min").first()
test_data = data.clone()

test_general_station_data = {}
test_rainfall_station_data = {}

for station in test_event_data.columns.get_level_values(1).unique():
    station_cols = (
        test_event_data.xs(station, level=1, axis=1)
        .interpolate(method="linear")
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
    if station in general_station:
        test_general_station_data[station] = station_cols.values
    else:
        test_rainfall_station_data[station] = station_cols.values[:, 0:1]
# print(test_general_station_data)
# print(test_rainfall_station_data)

test_general_station_features = []
test_rainfall_station_features = []

for station in general_station:
    if station in test_general_station_data:
        station_feat = test_general_station_data[station]
        test_general_station_features.append(station_feat)

for station in rainfall_station:
    if station in test_rainfall_station_data:
        station_feat = test_rainfall_station_data[station]
        test_rainfall_station_features.append(station_feat)

# print(test_general_station_features)
# print(test_rainfall_station_features)

test_data["general_station"].x = torch.tensor(
    np.array(test_general_station_features).transpose(1, 0, 2), dtype=torch.float
)
test_data["general_station"].y = torch.tensor(
    np.array(test_general_station_features)[:, :, 0:1].transpose(1, 0, 2),
    dtype=torch.float,
)
test_data["rainfall_station"].x = torch.tensor(
    np.array(test_rainfall_station_features).transpose(1, 0, 2), dtype=torch.float
)
test_data["rainfall_station"].y = torch.tensor(
    np.array(test_rainfall_station_features).transpose(1, 0, 2), dtype=torch.float
)

out = model(test_data.x_dict, test_data.edge_index_dict)
print(out.detach().numpy()[0])


# # Visualise rain on radar grid
# Hard coded to plot only consequitive 9 timestamps

# In[ ]:


print(out.detach().numpy()[0])


# In[ ]:


from src.utils import *
from src.visualisation import *

radar_df = load_radar_dataset("radar_vis")

fig, ax = plt.subplots(
    3, 3, figsize=(15, 12), subplot_kw={"projection": ccrs.PlateCarree()}
)

bounds_singapore = {"left": 103.6, "right": 104.1, "top": 1.5, "bottom": 1.188}
bounds = [0.1, 0.2, 0.5, 1, 2, 4, 7, 10, 20]
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")

for idx, timestamp in enumerate(out):
    output = {}
    count = 0

    for stn in general_station:
        output[stn] = float(timestamp[count])
        count += 1
    for stn in rainfall_station:
        output[stn] = float(timestamp[count])
        count += 1
    axi = ax[idx // 3][idx % 3]
    node_df = pd.Series(output)
    node_df = pandas_to_geodataframe(node_df)

    visualise_gauge_grid(node_df=node_df, ax=axi)
    improved_visualise_radar_grid(
        radar_df.iloc[idx], ax=axi, zoom=bounds_singapore, norm=norm
    )
    visualise_singapore_outline(ax=axi)


# In[ ]:


original_rainfall_rates = (
    weather_station_df_pivot.iloc[1773:1797].resample("15min").first()["rain_rate"]
)


print(original_rainfall_rates)


# In[ ]:


print(out)


# In[ ]:


actual_arr = []
pred_arr = []

for idx, timestamp in enumerate(out):
    output = {}
    count = 0
    a_arr = []
    p_arr = []

    for stn in general_station:
        output[stn] = float(timestamp[count])
        count += 1
    for stn in rainfall_station:
        output[stn] = float(timestamp[count])
        count += 1

    for key, value in output.items():
        a_arr.append(original_rainfall_rates.iloc[idx][key])
        p_arr.append(output[key])
    a_arr = list(map(lambda x: float(x), a_arr))
    actual_arr.append(a_arr)
    pred_arr.append(p_arr)

actual_arr = np.array(actual_arr)
pred_arr = np.array(pred_arr)

print(actual_arr)
print(pred_arr)
error = []
for i in range(len(actual_arr)):
    error.append(np.nanmean(actual_arr - pred_arr) ** 2)

MSE = np.mean(np.array(error))
print(MSE)


# In[ ]:


print(original_rainfall_rates.iloc[0])
