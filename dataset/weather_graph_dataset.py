from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Literal


class WeatherGraphDataset(Dataset):
    def __init__(self, data, mode=Literal["train", "test", "val"]):
        assert mode in ["train", "test", "val"], (
            "Invalid mode: must be either 'train' or 'test'."
        )

        self.data = data
        self.mode = mode
        self.num_timesteps = data["general_station"].x.shape[0]

        if mode == "train":
            self.gen_mask = data["general_station"].train_mask
            self.rain_mask = data["rainfall_station"].train_mask

        elif mode == "val":
            self.gen_mask = data["general_station"].val_mask
            self.rain_mask = data["rainfall_station"].val_mask

        else:  # test
            self.gen_mask = data["general_station"].test_mask
            self.rain_mask = data["rainfall_station"].test_mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        return {
            "gen_x": self.data["general_station"].x[idx],
            "rain_x": self.data["rainfall_station"].x[idx],
            "gen_y": self.data["general_station"].y[idx],
            "rain_y": self.data["rainfall_station"].y[idx],
        }


class WeatherGraphDatasetNew(Dataset):
    def __init__(self, data, mode=Literal["train", "test", "val"], device="cpu"):
        assert mode in ["train", "test", "val"], (
            "Invalid mode: must be either 'train' or 'test'."
        )

        self.data = data.to(device)
        self.mode = mode
        self.device = device
        self.num_timesteps = data["general_station"].x.shape[0]
        self.edge_index_dict = {
            key: val for key, val in self.data.edge_index_dict.items()
        }

        self.edge_attribute_dict = {
            key: val.to(device) for key, val in data.edge_attr_dict.items()
        }

        if mode == "train":
            self.metastation_mask = data["general_station"].train_mask
            self.rainfallstation_mask = data["rainfall_station"].train_mask

        elif mode == "val":
            self.metastation_mask = data["general_station"].val_mask
            self.rainfallstation_mask = data["rainfall_station"].val_mask

        else:  # test
            self.metastation_mask = data["general_station"].test_mask
            self.rainfallstation_mask = data["rainfall_station"].test_mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        return {
            "gen_x": self.data["general_station"].x[idx],
            "rain_x": self.data["rainfall_station"].x[idx],
            "gen_y": self.data["general_station"].y[idx],
            "rain_y": self.data["rainfall_station"].y[idx],
            "metastation_mask": self.metastation_mask,
            "rainfallstation_mask": self.rainfallstation_mask,
            "edge_index_dict": self.edge_index_dict,
            "edge_attr_dict": self.edge_attribute_dict,
        }


class WeatherGraphDatasetNew(Dataset):
    def __init__(self, data, mode=Literal["train", "test", "val"], device="cpu"):
        assert mode in ["train", "test", "val"], (
            "Invalid mode: must be either 'train' or 'test'."
        )

        self.data = data.to(device)
        self.mode = mode
        self.device = device
        self.num_timesteps = data["general_station"].x.shape[0]
        self.edge_index_dict = {
            key: val for key, val in self.data.edge_index_dict.items()
        }

        self.edge_attribute_dict = {
            key: val.to(device) for key, val in data.edge_attr_dict.items()
        }

        if mode == "train":
            self.metastation_mask = data["general_station"].train_mask
            self.rainfallstation_mask = data["rainfall_station"].train_mask

        elif mode == "val":
            self.metastation_mask = data["general_station"].val_mask
            self.rainfallstation_mask = data["rainfall_station"].val_mask

        else:  # test
            self.metastation_mask = data["general_station"].test_mask
            self.rainfallstation_mask = data["rainfall_station"].test_mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        return {
            "x": self.data["general_station"].x[idx],
            "rain_x": self.data["rainfall_station"].x[idx],
            "gen_y": self.data["general_station"].y[idx],
            "rain_y": self.data["rainfall_station"].y[idx],
            "metastation_mask": self.metastation_mask,
            "rainfallstation_mask": self.rainfallstation_mask,
            "edge_index_dict": self.edge_index_dict,
            "edge_attr_dict": self.edge_attribute_dict,
        }


class HomogeneousWeatherGraphDatasetNew(Dataset):
    def __init__(self, data, mode=Literal["train", "test", "val"], device="cpu"):
        assert mode in ["train", "test", "val"], (
            "Invalid mode: must be either 'train' or 'test'."
        )

        self.data = data.to(device)
        self.mode = mode
        self.device = device
        self.num_timesteps = data.x.shape[0]

        if mode == "train":
            self.mask = data.train_mask

        elif mode == "val":
            self.mask = data.val_mask

        else:  # test
            self.mask = data.test_mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        return {
            "x": self.data.x[idx],
            "y": self.data.y[idx],
            "mask": self.mask,
            "edge_index": self.data.edge_index,
            "edge_attr": self.data.edge_attr,
        }


class HomogeneousWeatherGraphDatasetInductive(Dataset):
    def __init__(self, graph, mode=Literal["train", "test", "val"], device="cpu"):
        assert mode in ["train", "test", "val"], (
            "Invalid mode: must be either 'train' or 'test'."
        )

        self.graph = graph
        self.mode = mode
        self.device = device

        # Graph has shape [T, N, F]
        self.num_timesteps = graph.x.shape[0]

        if mode == "train":
            self.mask = graph.train_mask
        elif mode == "val":
            self.mask = graph.val_mask
        else:
            self.mask = graph.test_mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        """Return a PyG Data object for timestep idx"""
        # Node features at this timestep: shape [N, F]
        x = self.graph.x[idx]
        # Labels at this timestep: shape [N, ...]
        y = self.graph.y[idx]
        
        # Create a PyG Data object
        data = Data(
            x=x,
            y=y,
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr if hasattr(self.graph, 'edge_attr') else None,
            mask=self.mask,  # Nodes to train on
            train_mask=self.graph.train_mask,  # For masking features
        )
        
        return data


class WeatherGraphDatasetWithRadar(Dataset):
    def __init__(self, data, mode=Literal["train", "test", "val"]):
        assert mode in ["train", "test", "val"], (
            "Invalid mode: must be either 'train' or 'test'."
        )

        self.data = data
        self.mode = mode
        self.num_timesteps = data["general_station"].x.shape[0]

        if mode == "train":
            self.gen_mask = data["general_station"].train_mask
            self.rain_mask = data["rainfall_station"].train_mask

        elif mode == "val":
            self.gen_mask = data["general_station"].val_mask
            self.rain_mask = data["rainfall_station"].val_mask

        else:  # test
            self.gen_mask = data["general_station"].test_mask
            self.rain_mask = data["rainfall_station"].test_mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        return {
            "gen_x": self.data["general_station"].x[idx],
            "rain_x": self.data["rainfall_station"].x[idx],
            "radar_x": self.data["radar_grid"].x[idx],  # Include radar
            "gen_y": self.data["general_station"].y[idx],
            "rain_y": self.data["rainfall_station"].y[idx],
        }


class WeatherGraphDatasetWithRadarNew(Dataset):
    def __init__(self, data, mode=Literal["train", "test", "val"], device="cpu"):
        assert mode in ["train", "test", "val"], (
            "Invalid mode: must be either 'train' or 'test'."
        )

        self.data = data.to(device)
        self.mode = mode
        self.device = device
        self.num_timesteps = data["general_station"].x.shape[0]
        self.edge_index_dict = {
            key: val for key, val in self.data.edge_index_dict.items()
        }

        self.edge_attribute_dict = {
            key: val.to(device) for key, val in data.edge_attr_dict.items()
        }

        if mode == "train":
            self.metastation_mask = data["general_station"].train_mask
            self.rainfallstation_mask = data["rainfall_station"].train_mask

        elif mode == "val":
            self.metastation_mask = data["general_station"].val_mask
            self.rainfallstation_mask = data["rainfall_station"].val_mask

        else:  # test
            self.metastation_mask = data["general_station"].test_mask
            self.rainfallstation_mask = data["rainfall_station"].test_mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        return {
            "gen_x": self.data["general_station"].x[idx],
            "rain_x": self.data["rainfall_station"].x[idx],
            "radar_x": self.data["radar_grid"].x[idx],  # Include radar
            "gen_y": self.data["general_station"].y[idx],
            "rain_y": self.data["rainfall_station"].y[idx],
            "metastation_mask": self.metastation_mask,
            "rainfallstation_mask": self.rainfallstation_mask,
            "edge_index_dict": self.edge_index_dict,
            "edge_attr_dict": self.edge_attribute_dict,
        }
