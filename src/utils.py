from dataset.weather_graph_dataset import WeatherGraphDataset, WeatherGraphDatasetNew
from src.miscellaneous import get_straight_distance
import xarray as xr
import rasterio
import yaml
import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader


def read_config(config_file):
    """
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    """

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    return cfg


def read_tif_file(tif_path: str):
    """
    Reads data from .tif files
    """

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds
        crs = src.crs
        transform = src.transform

    return data, bounds, crs, transform


def read_nc_file(filepath: str):
    """
    Reads data from .nc files
    """

    data = xr.open_dataset(filepath)

    return data


def add_weather_station_data(
    data,
    general_station_features,
    rainfall_station_features,
    dtype=torch.float32,
):
    data["general_station"].x = torch.tensor(
        np.array(general_station_features).transpose(1, 0, 2), dtype=dtype
    )
    data["rainfall_station"].x = torch.tensor(
        np.array(rainfall_station_features).transpose(1, 0, 2), dtype=dtype
    )

    # Add station targets
    data["general_station"].y = torch.tensor(
        np.array(general_station_features)[:, :, 0:1].transpose(1, 0, 2), dtype=dtype
    )
    data["rainfall_station"].y = torch.tensor(
        np.array(rainfall_station_features).transpose(1, 0, 2), dtype=dtype
    )

    print(data)
    print("\n=== Station Features Added ===")
    print(f"General station features shape: {data['general_station'].x.shape}")
    print(f"Rainfall station features shape: {data['rainfall_station'].x.shape}")

    return data


def add_mask_to_data(data, split_info, general_station, rainfall_station):
    data["general_station"].train_mask = [
        1 if station in split_info["ml"]["train"] else 0 for station in general_station
    ]
    data["general_station"].val_mask = [
        1 if station in split_info["ml"]["validation"] else 0
        for station in general_station
    ]
    data["general_station"].test_mask = [
        1 if (x == 0 and y == 0) else 0
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
        1 if (x == 0 and y == 0) else 0
        for x, y in zip(
            data["rainfall_station"].train_mask, data["rainfall_station"].val_mask
        )
    ]
    return data


def generate_edges(
    weather_station_locations,
    general_station,
    rainfall_station,
    K=4,
):
    ids = general_station + rainfall_station
    print(f"\nTotal stations for KNN: {len(ids)}")
    print(ids)

    coordinates = []
    for id in ids:
        coordinates.append(weather_station_locations[id])
    coords = np.array(coordinates)
    print(coords)

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

    edge_attributes = {
        "rainfall_to_rainfall": [],
        "rainfall_to_general": [],
        "general_to_rainfall": [],
        "general_to_general": [],
    }

    # Add station coordinates for nx plotting
    for idx, station in enumerate(general_station + rainfall_station):
        G.add_node(
            idx,
            pos=(
                weather_station_locations[station][1],
                weather_station_locations[station][0],
            ),
        )

    color_map = ["green" for i in range(len(general_station))] + [
        "red" for i in range(len(rainfall_station))
    ]

    # Build edges
    for idx, row in enumerate(indices):
        origin = row[0]

        for n in row[1:]:
            G.add_edge(origin, n)
            if ids[origin] in rainfall_station:
                start_id = rainfall_station.index(ids[origin])
                if ids[n] in rainfall_station:
                    end_id = rainfall_station.index(ids[n])
                    edges["rainfall_to_rainfall"].append([start_id, end_id])
                    edge_attributes["rainfall_to_rainfall"].append(
                        [
                            get_straight_distance(
                                weather_station_locations[ids[origin]],
                                weather_station_locations[ids[n]],
                            )
                        ]
                    )
                else:
                    end_id = general_station.index(ids[n])
                    edges["rainfall_to_general"].append([start_id, end_id])
                    edge_attributes["rainfall_to_general"].append(
                        [
                            get_straight_distance(
                                weather_station_locations[ids[origin]],
                                weather_station_locations[ids[n]],
                            )
                        ]
                    )
            else:
                start_id = general_station.index(ids[origin])
                if ids[n] in rainfall_station:
                    end_id = rainfall_station.index(ids[n])
                    edges["general_to_rainfall"].append([start_id, end_id])
                    edge_attributes["general_to_rainfall"].append(
                        [
                            get_straight_distance(
                                weather_station_locations[ids[origin]],
                                weather_station_locations[ids[n]],
                            )
                        ]
                    )
                else:
                    end_id = general_station.index(ids[n])
                    edges["general_to_general"].append([start_id, end_id])
                    edge_attributes["general_to_general"].append(
                        [
                            get_straight_distance(
                                weather_station_locations[ids[origin]],
                                weather_station_locations[ids[n]],
                            )
                        ]
                    )

    print(f"\nGraph info: {G}")
    print(f"Connected components: {len(list(nx.connected_components(G)))}")
    nx.draw(G, nx.get_node_attributes(G, 'pos'), node_color = color_map, with_labels=True, font_weight='bold')

    # Convert edge lists to proper format
    for key, val in edges.items():
        xarr = []
        yarr = []
        for x, y in val:
            xarr.append(x)
            yarr.append(y)
        edges[key] = [xarr, yarr]

    return edges, edge_attributes


def add_edge_attributes_to_data(
    data,
    edges,
    edge_attributes,
    dtype=torch.float32,
):
    # Add station-to-station edges
    data[
        "general_station", "gen_to_rain", "rainfall_station"
    ].edge_index = torch.tensor(edges["general_to_rainfall"], dtype=torch.long)
    data[
        "rainfall_station", "rain_to_gen", "general_station"
    ].edge_index = torch.tensor(edges["rainfall_to_general"], dtype=torch.long)
    data["general_station", "gen_to_gen", "general_station"].edge_index = torch.tensor(
        edges["general_to_general"], dtype=torch.long
    )
    data[
        "rainfall_station", "rain_to_rain", "rainfall_station"
    ].edge_index = torch.tensor(edges["rainfall_to_rainfall"], dtype=torch.long)

    # Add edge attributes
    data["general_station", "gen_to_rain", "rainfall_station"].edge_attr = torch.tensor(
        edge_attributes["general_to_rainfall"], dtype=dtype
    )
    data["rainfall_station", "rain_to_gen", "general_station"].edge_attr = torch.tensor(
        edge_attributes["rainfall_to_general"], dtype=dtype
    )
    data["general_station", "gen_to_gen", "general_station"].edge_attr = torch.tensor(
        edge_attributes["general_to_general"], dtype=dtype
    )
    data[
        "rainfall_station", "rain_to_rain", "rainfall_station"
    ].edge_attr = torch.tensor(edge_attributes["rainfall_to_rainfall"], dtype=dtype)

    print("\n=== Station-to-Station Edges Added ===")
    return data


def print_data_structure(data):
    print("\n" + "=" * 60)
    print("FINAL HETERODATA STRUCTURE")
    print("=" * 60)
    print(data)
    print("\nNode types:", data.node_types)
    print("Edge types:", data.edge_types)

    print("\n--- Feature Shapes ---")
    print(f"General stations: {data['general_station'].x.shape}")
    print(f"Rainfall stations: {data['rainfall_station'].x.shape}")

    print("\n--- Edge Counts ---")
    for edge_type in data.edge_types:
        edge_count = data[edge_type].edge_index.shape[1]
        print(f"{edge_type}: {edge_count} edges")

    print("\n--- Mask Counts ---")
    print(f"General train: {sum(data['general_station'].train_mask)}")
    print(f"General val: {sum(data['general_station'].val_mask)}")
    print(f"General test: {sum(data['general_station'].test_mask)}")
    print(f"Rainfall train: {sum(data['rainfall_station'].train_mask)}")
    print(f"Rainfall val: {sum(data['rainfall_station'].val_mask)}")
    print(f"Rainfall test: {sum(data['rainfall_station'].test_mask)}")
    print("=" * 60)


def collate_temporal_graphs(batch):
    gen_x = torch.stack([item["gen_x"] for item in batch])
    rain_x = torch.stack([item["rain_x"] for item in batch])
    gen_y = torch.stack([item["gen_y"] for item in batch])
    rain_y = torch.stack([item["rain_y"] for item in batch])

    return {
        "gen_x": gen_x,
        "rain_x": rain_x,
        "gen_y": gen_y,
        "rain_y": rain_y,
    }

def prepare_dataset(data, batch_size=16):
    train_dataset = WeatherGraphDataset(data, mode="train")
    val_dataset = WeatherGraphDataset(data, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_graphs,
    )
    return train_loader, val_loader

def collate_temporal_graphs_new(batch):
  gen_x = torch.stack([item['gen_x'] for item in batch])
  rain_x = torch.stack([item['rain_x'] for item in batch])
  gen_y = torch.stack([item['gen_y'] for item in batch])
  rain_y = torch.stack([item['rain_y'] for item in batch])

  metastation_mask = batch[0]['metastation_mask']
  rainfallstation_mask = batch[0]['rainfallstation_mask']
  edge_index_dict = batch[0]['edge_index_dict']
  edge_attribute_dict = batch[0]['edge_attr_dict']

  return {
      'gen_x': gen_x,
      'rain_x': rain_x,
      'gen_y': gen_y,
      'rain_y': rain_y,
      'metastation_mask': metastation_mask,
      'rainfallstation_mask': rainfallstation_mask,
      'edge_index_dict': edge_index_dict,
      'edge_attr_dict': edge_attribute_dict
  }


def prepare_dataset_new(data, batch_size=16):
    train_dataset = WeatherGraphDatasetNew(data, mode="train")
    val_dataset = WeatherGraphDatasetNew(data, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_graphs_new,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_graphs_new,
    )
    return train_loader, val_loader
