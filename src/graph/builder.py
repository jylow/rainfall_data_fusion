from torch_geometric.data import HeteroData
import torch
import numpy as np
import math

"""
Class for graph builder
Helps to take the input data and produce graphs based on training, testing and validation splits based on one input dataset
"""


class GraphBuilder:
    def __init__(self, node_feature_dict: dict, station_lists=dict):
        """
        node_feature_dict: contains information on heterogeneous node
        station_lists: reference to maintain mapping of stations to node orderings
        """
        self.hetero_data = HeteroData()
        self.dtype = torch.float32
        self.station_lists = station_lists
        self.masks = {}

        for nodetype, nodedata in node_feature_dict.items():
            self.hetero_data[nodetype].x = torch.tensor(
                np.array(nodedata).transpose(1, 0, 2), dtype=self.dtype
            )
            self.hetero_data[nodetype].y = torch.tensor(
                np.array(nodedata)[:, :, 0:1].transpose(1, 0, 2), dtype=self.dtype
            )

    def get_networkX_graph(self):
        pass

    def set_split(self, split_info: dict):
        self.split_info = split_info

    def get_training_graph(self) -> HeteroData:
        self.training_graph = HeteroData()

        for nodetype, nodedata in self.hetero_data.node_items():
            stations = self.station_lists[nodetype]
            training_stations = self.split_info["train"]
            training_mask = [
                True if stn in training_stations else False for stn in stations
            ]
            training_mask_tensor = torch.tensor(training_mask)

            self.training_graph[nodetype].x = nodedata.x[:, training_mask_tensor, :]

        return self.training_graph

    def get_validation_graph(self):
        pass

    def get_testing_graph(self):
        pass

    def get_original_graph_data(self) -> HeteroData:
        return self.hetero_data

    def get_distance(a, b) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def build_graph(node_df: HeteroData):
        """
        Function to build graph for training/test/validation/whole graph
        Does not return anything. Graph is modified in place to add edges.
        """

        edges = {
            "rainfall_to_rainfall": [],
            "rainfall_to_general": [],
            "meta_to_meta": [],
            "meta_to_meta": [],
        }

        edge_attributes = {
            "rainfall_to_rainfall": [],
            "rainfall_to_general": [],
            "meta_to_meta": [],
            "meta_to_meta": [],
        }
