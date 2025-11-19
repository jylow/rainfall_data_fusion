from dataset.weather_graph_dataset import (
    HomogeneousWeatherGraphDatasetInductive,
    WeatherGraphDataset,
    WeatherGraphDatasetNew,
    HomogeneousWeatherGraphDatasetNew,
)
from src.miscellaneous import get_straight_distance
import xarray as xr
import rasterio
import yaml
import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph


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


def add_homogeneous_weather_station_data(
    data,
    general_station_features,
    rainfall_station_features,
    general_station_ids=None,
    rainfall_station_ids=None,
    dtype=torch.float32,
):
    general_station_data_tensor = torch.tensor(
        np.array(general_station_features)[:, :, 0:1].transpose(1, 0, 2), dtype=dtype
    )
    rainfall_station_data_tensor = torch.tensor(
        np.array(rainfall_station_features).transpose(1, 0, 2), dtype=dtype
    )

    # Add station targets
    general_station_target_tensor = torch.tensor(
        np.array(general_station_features)[:, :, 0:1].transpose(1, 0, 2), dtype=dtype
    )
    rainfall_station_target_tensor = torch.tensor(
        np.array(rainfall_station_features).transpose(1, 0, 2), dtype=dtype
    )

    # --- Add station IDs ---
    # Number of general + rainfall stations
    N_gen = np.array(general_station_features).shape[0]
    N_rain = np.array(rainfall_station_features).shape[0]

    # --- Assign General Station IDs ---
    if general_station_ids is not None and isinstance(general_station_ids, int):
        gen_ids = torch.tensor(np.array(general_station_ids), dtype=torch.long)
    else:
        gen_ids = torch.tensor(np.arange(N_gen), dtype=torch.long)

    # --- Assign Rainfall Station IDs with OFFSET ---
    if rainfall_station_ids is not None and isinstance(rainfall_station_ids, int):
        rain_ids = torch.tensor(np.array(rainfall_station_ids), dtype=torch.long)
    else:
        rain_ids = torch.tensor(np.arange(N_gen, N_gen + N_rain), dtype=torch.long)

    # print("\n=== Station ID Mapping ===")
    # print("\nGeneral Stations:")
    # for name, sid in zip(general_station_features, gen_ids):
    #     print(f"  {name}  →  {sid}")

    # print("\nRainfall Stations (Offset IDs):")
    # for name, sid in zip(rainfall_station_features, rain_ids):
    #     print(f"  {name}  →  {sid}")

    # print("\nTotal Stations:", len(gen_ids) + len(rain_ids))

    station_data_tensor = torch.concat(
        [general_station_data_tensor, rainfall_station_data_tensor], dim=1
    )
    station_target_tensor = torch.concat(
        [general_station_target_tensor, rainfall_station_target_tensor], dim=1
    )
    station_id_tensor = torch.concat([gen_ids, rain_ids], dim=0)
    data.x = station_data_tensor
    data.y = station_target_tensor
    data.station_id = station_id_tensor

    print(data)
    print("\n=== Station Features Added ===")
    print(f"Station features shape: {data.x.shape}")

    return data


def add_weather_station_data(
    data,
    general_station_features,
    rainfall_station_features,
    general_station_ids=None,
    rainfall_station_ids=None,
    dtype=torch.float32,
    include_metastation_info=True,
):
    if include_metastation_info:
        data["general_station"].x = torch.tensor(
            np.array(general_station_features).transpose(1, 0, 2), dtype=dtype
        )
    else:
        data["general_station"].x = torch.tensor(
            np.array(general_station_features)[:, :, 0:1].transpose(1, 0, 2),
            dtype=dtype,
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

    # --- Add station IDs ---
    if general_station_ids is not None:
        data["general_station"].station_ids = np.array(general_station_ids)
    else:
        data["general_station"].station_ids = np.arange(
            np.array(general_station_features).shape[0]
        )

    if rainfall_station_ids is not None:
        data["rainfall_station"].station_ids = np.array(rainfall_station_ids)
    else:
        data["rainfall_station"].station_ids = np.arange(
            np.array(rainfall_station_features).shape[0]
        )

    print(data)
    print("\n=== Station Features Added ===")
    print(f"General station features shape: {data['general_station'].x.shape}")
    print(f"Rainfall station features shape: {data['rainfall_station'].x.shape}")
    print(f"General station IDs: {data['general_station'].station_ids[:5]}")
    print(f"Rainfall station IDs: {data['rainfall_station'].station_ids[:5]}")

    return data


def build_train_and_full_graph_homogeneous(
    data,
    split_info,
    stations_ids,
):
    """
    Builds for a HOMOGENEOUS graph:
      1. full_graph (all nodes, with masks only)
      2. train_graph (train + val nodes only, test nodes removed)

    Assumes data.x and data.y are (T, N, F)
    """

    # 1. Build full_graph (all nodes) and add masks
    # -------------------------------------------------
    # print("before cloning, edge_index:", data.edge_index.shape)
    # print("before cloning, edge_attr:", data.edge_attr.shape)

    full_graph = data.clone()
    all_station_ids_list = list(stations_ids)

    # Create sets for fast O(1) lookups
    train_set = set(split_info["ml"]["train"])
    val_set = set(split_info["ml"]["validation"])

    # Build boolean masks that align with the node order (0...N_total-1)
    train_mask = torch.tensor(
        [sid in train_set for sid in all_station_ids_list], dtype=torch.bool
    )
    val_mask = torch.tensor(
        [sid in val_set for sid in all_station_ids_list], dtype=torch.bool
    )
    # Test mask is everything not in train or val
    test_mask = ~(train_mask | val_mask)

    # Assign masks to the homogeneous graph
    full_graph.train_mask = train_mask
    full_graph.val_mask = val_mask
    full_graph.test_mask = test_mask

    # 2. Build train_graph (subgraph)
    # -------------------------------------------------
    train_graph = data.clone()

    # Get the mask for all nodes to KEEP (train + val)
    # keep_nodes_mask = full_graph.train_mask | full_graph.val_mask
    keep_nodes_mask = full_graph.train_mask
    num_total_nodes = len(keep_nodes_mask)

    # --- Filter Node Features (Spatio-Temporal Aware) ---
    # We must filter on dimension 1 (the Node dimension)
    # (T, N_total, F) -> (T, N_sub, F)
    train_graph.x = train_graph.x[:, keep_nodes_mask, :]
    train_graph.y = train_graph.y[:, keep_nodes_mask, :]

    # --- Filter other node-level attributes ---
    # We also filter the masks themselves so they align with the new graph
    train_graph.train_mask = full_graph.train_mask[keep_nodes_mask]
    train_graph.val_mask = full_graph.val_mask[keep_nodes_mask]
    train_graph.test_mask = full_graph.test_mask[keep_nodes_mask]  # (will be all False)

    # --- Save mapping from new index (0..N_sub-1) to old index (0..N_total-1) ---
    # This is critical for mapping predictions back to the full graph
    train_graph.orig_id = torch.arange(num_total_nodes)[keep_nodes_mask]

    # --- Filter Edges (The PyG Way) ---
    # Use the 'subgraph' utility to:
    # 1. Select only edges where both nodes are in `keep_nodes_mask`
    # 2. Relabel node indices in edge_index from (0..N_total-1) to (0..N_sub-1)

    new_edge_index, new_edge_attr, edge_mask = subgraph(
        subset=keep_nodes_mask,
        edge_index=train_graph.edge_index,
        relabel_nodes=True,
        num_nodes=num_total_nodes,
        return_edge_mask=True,
    )

    train_graph.edge_index = new_edge_index

    # Also filter edge attributes if they exist
    if hasattr(train_graph, "edge_attr") and train_graph.edge_attr is not None:
        train_graph.edge_attr = train_graph.edge_attr[:, edge_mask]

    # 3. Build validation_graph (subgraph)
    # -------------------------------------------------
    validation_graph = data.clone()

    # Get the mask for all nodes to KEEP (train + val)
    keep_nodes_mask = full_graph.train_mask | full_graph.val_mask
    num_total_nodes = len(keep_nodes_mask)

    # --- Filter Node Features (Spatio-Temporal Aware) ---
    # We must filter on dimension 1 (the Node dimension)
    # (T, N_total, F) -> (T, N_sub, F)
    validation_graph.x = validation_graph.x[:, keep_nodes_mask, :]
    validation_graph.y = validation_graph.y[:, keep_nodes_mask, :]

    # --- Filter other node-level attributes ---
    # We also filter the masks themselves so they align with the new graph
    validation_graph.train_mask = full_graph.train_mask[keep_nodes_mask]
    validation_graph.val_mask = full_graph.val_mask[keep_nodes_mask]
    validation_graph.test_mask = full_graph.test_mask[keep_nodes_mask]  # (will be all False)

    # --- Save mapping from new index (0..N_sub-1) to old index (0..N_total-1) ---
    # This is critical for mapping predictions back to the full graph
    validation_graph.orig_id = torch.arange(num_total_nodes)[keep_nodes_mask]

    # --- Filter Edges (The PyG Way) ---
    # Use the 'subgraph' utility to:
    # 1. Select only edges where both nodes are in `keep_nodes_mask`
    # 2. Relabel node indices in edge_index from (0..N_total-1) to (0..N_sub-1)

    new_edge_index, new_edge_attr, edge_mask = subgraph(
        subset=keep_nodes_mask,
        edge_index=validation_graph.edge_index,
        relabel_nodes=True,
        num_nodes=num_total_nodes,
        return_edge_mask=True,
    )

    validation_graph.edge_index = new_edge_index

    # Also filter edge attributes if they exist
    if hasattr(validation_graph, "edge_attr") and validation_graph.edge_attr is not None:
        validation_graph.edge_attr = validation_graph.edge_attr[:, edge_mask]

    # print("after processing, train edge_index:", train_graph.edge_index.shape)
    # print("after processing, train edge_attr:", train_graph.edge_attr.shape)
    # print("after processing, full edge_index:", full_graph.edge_index.shape)
    # print("after processing, full edge_attr:", full_graph.edge_attr.shape)

    return train_graph, validation_graph, full_graph


def build_train_and_full_graph(
    data, split_info, general_station_ids, rainfall_station_ids
):
    """
    Builds:
      1. full_graph (all nodes, with masks only)
      2. train_graph (train + val nodes only, test nodes removed)
    """

    # Clone original graph
    full_graph = data.clone()

    # Masks --------------------------
    full_graph["general_station"].train_mask = torch.tensor(
        [1 if sid in split_info["ml"]["train"] else 0 for sid in general_station_ids]
    )
    full_graph["general_station"].val_mask = torch.tensor(
        [
            1 if sid in split_info["ml"]["validation"] else 0
            for sid in general_station_ids
        ]
    )
    full_graph["general_station"].test_mask = 1 - (
        full_graph["general_station"].train_mask
        | full_graph["general_station"].val_mask
    )

    full_graph["rainfall_station"].train_mask = torch.tensor(
        [1 if sid in split_info["ml"]["train"] else 0 for sid in rainfall_station_ids]
    )
    full_graph["rainfall_station"].val_mask = torch.tensor(
        [
            1 if sid in split_info["ml"]["validation"] else 0
            for sid in rainfall_station_ids
        ]
    )
    full_graph["rainfall_station"].test_mask = 1 - (
        full_graph["rainfall_station"].train_mask
        | full_graph["rainfall_station"].val_mask
    )

    # --------------------------------------------
    # Build train_graph by KEEPING train+val nodes
    # --------------------------------------------
    train_graph = data.clone()

    keep_gen = (
        full_graph["general_station"].train_mask
        | full_graph["general_station"].val_mask
    ).bool()
    keep_rain = (
        full_graph["rainfall_station"].train_mask
        | full_graph["rainfall_station"].val_mask
    ).bool()

    # Filter node features
    train_graph["general_station"].x = train_graph["general_station"].x[keep_gen]
    train_graph["general_station"].y = train_graph["general_station"].y[keep_gen]

    train_graph["rainfall_station"].x = train_graph["rainfall_station"].x[keep_rain]
    train_graph["rainfall_station"].y = train_graph["rainfall_station"].y[keep_rain]

    # Save mapping (very important)
    train_graph["general_station"].orig_id = torch.arange(len(keep_gen))[keep_gen]
    train_graph["rainfall_station"].orig_id = torch.arange(len(keep_rain))[keep_rain]

    # Filter edges based on retained nodes
    new_edge_index_dict = {}
    for (src, rel, dst), edge_index in train_graph.edge_index_dict.items():
        if src == "general_station":
            src_mask = keep_gen
        elif src == "rainfall_station":
            src_mask = keep_rain
        else:
            src_mask = torch.ones(train_graph[src].num_nodes, dtype=bool)

        if dst == "general_station":
            dst_mask = keep_gen
        elif dst == "rainfall_station":
            dst_mask = keep_rain
        else:
            dst_mask = torch.ones(train_graph[dst].num_nodes, dtype=bool)

        # Keep edges where both endpoints survive
        keep_edges = src_mask[edge_index[0]] & dst_mask[edge_index[1]]
        new_edge_index = edge_index[:, keep_edges]

        new_edge_index_dict[(src, rel, dst)] = new_edge_index

    train_graph.edge_index_dict = new_edge_index_dict

    return train_graph, full_graph


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


def add_homogeneous_mask_to_data(data, split_info, stations):
    data.train_mask = [
        1 if station in split_info["ml"]["train"] else 0 for station in stations
    ]
    data.val_mask = [
        1 if station in split_info["ml"]["validation"] else 0 for station in stations
    ]
    data.test_mask = [
        1 if station in split_info["ml"]["test"] else 0 for station in stations
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
        sid = int(station[1:])
        G.add_node(
            idx,
            pos=(
                weather_station_locations[station][1],
                weather_station_locations[station][0],
            ),
            label=station,
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
    labels = nx.get_node_attributes(G, "label")
    nx.draw(
        G,
        nx.get_node_attributes(G, "pos"),
        node_color=color_map,
        with_labels=True,
        font_weight="bold",
        labels=labels,
    )

    # Convert edge lists to proper format
    for key, val in edges.items():
        xarr = []
        yarr = []
        for x, y in val:
            xarr.append(x)
            yarr.append(y)
        edges[key] = [xarr, yarr]

    return edges, edge_attributes


def generate_homogeneous_edges(
    weather_station_locations,
    stations,
    K=4,
):
    ids = stations
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

    edges = []

    edge_attributes = []
    # Add station coordinates for nx plotting
    for idx, station in enumerate(stations):
        sid = int(station[1:])
        G.add_node(
            idx,
            pos=(
                weather_station_locations[station][1],
                weather_station_locations[station][0],
            ),
            label=station,
        )

    color_map = ["green" for i in range(len(stations))]

    # Build edges
    for idx, row in enumerate(indices):
        origin = row[0]

        for n in row[1:]:
            G.add_edge(origin, n)

            edges.append([origin, n])
            edge_attributes.append(
                [
                    get_straight_distance(
                        weather_station_locations[ids[origin]],
                        weather_station_locations[ids[n]],
                    )
                ]
            )

    print(f"\nGraph info: {G}")
    print(f"Connected components: {len(list(nx.connected_components(G)))}")
    labels = nx.get_node_attributes(G, "label")
    nx.draw(
        G,
        nx.get_node_attributes(G, "pos"),
        node_color=color_map,
        with_labels=True,
        font_weight="bold",
        labels=labels,
    )

    # Convert edge lists to proper format
    # for key, val in edges.items():
    #     xarr = []
    #     yarr = []
    #     for x, y in val:
    #         xarr.append(x)
    #         yarr.append(y)
    #     edges[key] = [xarr, yarr]

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


def add_homogeneous_edge_attributes_to_data(
    data,
    edges,
    edge_attributes,
    dtype=torch.float32,
):
    # Add station-to-station edges
    data.edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1)

    # Add edge attributes
    data.edge_attr = torch.tensor(edge_attributes, dtype=dtype).transpose(0, 1)
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
    gen_x = torch.stack([item["gen_x"] for item in batch])
    rain_x = torch.stack([item["rain_x"] for item in batch])
    gen_y = torch.stack([item["gen_y"] for item in batch])
    rain_y = torch.stack([item["rain_y"] for item in batch])

    metastation_mask = batch[0]["metastation_mask"]
    rainfallstation_mask = batch[0]["rainfallstation_mask"]
    edge_index_dict = batch[0]["edge_index_dict"]
    edge_attribute_dict = batch[0]["edge_attr_dict"]

    return {
        "gen_x": gen_x,
        "rain_x": rain_x,
        "gen_y": gen_y,
        "rain_y": rain_y,
        "metastation_mask": metastation_mask,
        "rainfallstation_mask": rainfallstation_mask,
        "edge_index_dict": edge_index_dict,
        "edge_attr_dict": edge_attribute_dict,
    }


def collate_homogeneous_graphs_new(batch):
    x = torch.stack([item["x"] for item in batch])
    y = torch.stack([item["y"] for item in batch])

    mask = batch[0]["mask"]
    edge_index = batch[0]["edge_index"]
    edge_attribute = batch[0]["edge_attr"]

    return {
        "x": x,
        "y": y,
        "mask": mask,
        "edge_index": edge_index,
        "edge_attr": edge_attribute,
    }

def inductive_collate_fn(batch):
    """
    Collate function for inductive weather graph dataset.
    Expects each Data object in the batch to contain ._dataset_mask,
    which points to the dataset-level node mask.
    """

    # PyG batch
    batched_data = Batch.from_data_list(batch)

    return batched_data


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


def prepare_homogeneous_dataset(data, batch_size=16):
    train_dataset = HomogeneousWeatherGraphDatasetNew(data, mode="train")
    val_dataset = HomogeneousWeatherGraphDatasetNew(data, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_homogeneous_graphs_new,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_homogeneous_graphs_new,
    )
    return train_loader, val_loader


def filter_edges_for_inductive(graph):
    """
    Remove edges involving test/val nodes for the *training graph*.
    Also filter edge_attr accordingly.
    """
    train_val_mask = graph.train_mask | graph.val_mask
    node_is_trainval = train_val_mask[graph.edge_index]
    keep = node_is_trainval[0] & node_is_trainval[1]

    graph.edge_index = graph.edge_index[:, keep]
    if graph.edge_attr is not None:
        graph.edge_attr = graph.edge_attr[:, keep]

    return graph


def prepare_homogeneous_inductive_dataset(
    train_graph, validation_graph, full_graph=None, batch_size=16, mode="train"
):
    """
    Prepare dataloaders using train_graph for training/validation
    and full_graph for testing (optional)

    Args:
        train_graph: Graph with only train nodes (for training)
        validation_graph: Graph with only train+val nodes (for validation)
        full_graph: Graph with all nodes including test (for testing, optional)
        batch_size: Batch size
        mode: 'train' or 'test'

    Returns:
        train_loader, val_loader (if mode='train')
        test_loader (if mode='test' and full_graph provided)
    """

    if mode == "train":
        # Use TRAIN GRAPH for both training and validation
        print(f"\n{'=' * 60}")
        print("PREPARING TRAIN/VAL DATALOADERS (using train_graph)")
        print(f"{'=' * 60}")

        train_graph = filter_edges_for_inductive(train_graph)
        train_dataset = HomogeneousWeatherGraphDatasetInductive(
            train_graph, mode="train"
        )
        val_dataset = HomogeneousWeatherGraphDatasetInductive(validation_graph, mode="val")

        # Inspect one sample
        sample = train_dataset[0]
        print(f"Sample node features shape: {sample.x.shape}")  # [N, F]
        print(f"Sample labels shape: {sample.y.shape}")  # [N, Tgt]
        print(f"Train mask shape: {train_dataset.mask.shape}")  # [N]
        print(f"Train mask sum (trainable nodes): {train_dataset.mask.sum()}")

        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for training
            collate_fn=inductive_collate_fn,
        )
        val_loader = GeometricDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for validation
            collate_fn=inductive_collate_fn,
        )

        print(f"✅ Train loader: {len(train_loader)} batches")
        print(f"✅ Val loader: {len(val_loader)} batches")

        # Verify batch shapes
        sample_batch = next(iter(train_loader))
        print(f"\n{'='*60}")
        print("BATCH VERIFICATION")
        print(f"{'='*60}")
        print(f"Batch content: {sample_batch}")
        print(f"Batched x shape: {sample_batch.x.shape}")  # [B*N, F]
        print(f"Batched y shape: {sample_batch.y.shape}")  # [B*N, Tgt]
        print(f"Mask shape: {sample_batch.mask.shape}")  # [N] - SINGLE mask!
        print(f"Edge index shape: {sample_batch.edge_index.shape}")
        print(f"Batch vector shape: {sample_batch.batch.shape}")
        print(f"Num graphs in batch: {sample_batch.num_graphs}")
        
        # Calculate expected single graph node count
        nodes_per_graph = sample_batch.x.shape[0] // sample_batch.num_graphs
        masks_per_graph = sample_batch.mask.shape[0] // sample_batch.num_graphs
        print(f"\nNodes per graph: {nodes_per_graph}")
        print(f"Mask size per graph: {masks_per_graph}")
        print("✅ CORRECT!" if nodes_per_graph == masks_per_graph else "❌ MISMATCH!")

        return train_loader, val_loader

    elif mode == "test":
        # Use FULL GRAPH for testing
        if full_graph is None:
            raise ValueError("full_graph must be provided for test mode")

        print(f"\n{'=' * 60}")
        print("PREPARING TEST DATALOADER (using full_graph)")
        print(f"{'=' * 60}")

        test_dataset = HomogeneousWeatherGraphDatasetInductive(full_graph, mode="test")

        test_loader = GeometricDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=inductive_collate_fn,
        )

        print(f"✅ Test loader: {len(test_loader)} batches")

        return test_loader

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'test'")

def debug_dataloader(dataloader, num_batches=1):
    """
    Print shapes of a few batches to debug.
    """
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx}")
        print(f"{'='*60}")
        print(f"batch.x shape: {batch.x.shape}")  # Should be [B*N, F]
        print(f"batch.y shape: {batch.y.shape}")  # Should be [B*N, Tgt]
        print(f"batch.mask shape: {batch.mask.shape}")  # Should be [N] NOT [B*N]!
        print(f"batch.edge_index shape: {batch.edge_index.shape}")  # [2, E*B]
        print(f"batch.batch shape: {batch.batch.shape}")  # [B*N]
        print(f"batch.num_graphs: {batch.num_graphs}")  # Should be B (batch size)
        print(f"\nExpected single node count: {batch.x.shape[0] // batch.num_graphs}")
        print(f"Actual mask size: {batch.mask.shape[0]}")
        print(f"Match: {batch.mask.shape[0] == batch.x.shape[0] // batch.num_graphs}")
