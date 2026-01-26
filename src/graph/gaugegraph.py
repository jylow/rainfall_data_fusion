import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data


from src.utils import generate_homogeneous_edges, add_homogeneous_edge_attributes_to_data

class GaugeGraph():

    def __init__(self, data: Data, station_dict: dict, split_info: dict, raingauge_station_order, knn: int):
        """
        node_feature_dict: contains information on heterogeneous node
        station_lists: reference to maintain mapping of stations to node orderings
        """
        self.dtype = torch.float32
        self.station_dict = station_dict
        self.data = data
        self.split_info = split_info
        self.raingauge_order = raingauge_station_order
        self.knn = knn

        train_stations = np.array([x for x in self.raingauge_order if x in split_info['ml']['train']])
        validation_stations = np.array([x for x in self.raingauge_order if x in split_info['ml']['train'] or x in split_info['ml']['validation']])

        self.train_graph = self.build_graph("train", train_stations)
        self.validation_graph = self.build_graph("validation", validation_stations)
        self.test_graph = self.build_graph("test", self.raingauge_order)

    def get_train_graph(self):
        return self.train_graph

    def get_validation_graph(self):
        self.validation_graph.validation_mask = self.get_validation_graph_mask()
        return self.validation_graph

    def get_validation_graph_mask(self):
        train_indices = torch.nonzero(self.validation_graph.train_mask)
        val_indices = torch.nonzero(self.validation_graph.val_mask)

        combined_indices = torch.cat([train_indices, val_indices], dim=0)
        combined_indices_sorted, _ = torch.sort(combined_indices.squeeze())

        num_train = train_indices.size(0)
        num_val = val_indices.size(0)

        validation_graph_mask = torch.isin(combined_indices_sorted, val_indices.flatten())
        return validation_graph_mask

    def get_test_graph(self):
        return self.test_graph

    def build_graph(self, split: str, stations) -> Data:
        '''
        BUILDS A GRAPH FOR TRAIN/VALIDATION/TEST
        '''
        match split:
            case "train":
                mask = self.data.train_mask
            case "validation":
                mask = torch.logical_or(self.data.train_mask, self.data.val_mask)
            case "test":
                mask = torch.ones(self.data.train_mask.shape[0])
            case _: #should not reach here
                print("ERROR CODE SHOULD NOT REACH HERE PLEASE LOOK AT BUILD_GRAPH FUNCTION")
                mask = [1 for _ in range(len(self.data.test_mask))]

        mask = torch.tensor(mask, dtype=torch.bool)

        split_data = self.data.clone()
        print(split_data)
        print("SPLIT DATA")
        split_data.x = split_data.x[:, mask, :]
        split_data.y = split_data.y[:, mask, :]

        edges, edge_attributes = generate_homogeneous_edges(
            self.station_dict,
            stations = stations,
            K = self.knn
        )

        split_graph = add_homogeneous_edge_attributes_to_data(
                split_data,
                edges,
                edge_attributes,
                dtype=self.dtype
        )

        return split_graph


    def visualise_graph_split(self):

        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        #1. Build the training graph
        train_nx_graph = nx.Graph()
        train_indices = torch.nonzero(self.train_graph.train_mask)
        train_nx_graph.add_nodes_from(range(train_indices.shape[0]))
        train_nx_graph.add_edges_from(self.train_graph.edge_index.numpy().T) # Get the edge indices from the graph

        train_station_ids = [self.raingauge_order[x.item()] for x in train_indices]
        train_station_locations = []
        for id in train_station_ids:
            lat, lon = self.station_dict[id]
            train_station_locations.append((lon, lat))
        print(train_station_ids)


        #2. Build the validation graph
        validation_nx_graph = nx.Graph()
        # concatenate the validation and train masks
        validation_indices = torch.concat([torch.nonzero(self.validation_graph.val_mask), torch.nonzero(self.validation_graph.train_mask)]).flatten()
        validation_indices, _= torch.sort(validation_indices)
        validation_nx_graph.add_nodes_from(range(validation_indices.shape[0]))
        validation_nx_graph.add_edges_from(self.validation_graph.edge_index.numpy().T)

        val_station_ids = [self.raingauge_order[x] for x in validation_indices]
        val_station_locations = []
        print(val_station_ids)
        for id in val_station_ids:
            lat, lon = self.station_dict[id]
            val_station_locations.append((lon, lat))
        val_station_colors = ["blue" for _ in range(len(val_station_locations))]

        # Set validation stations to green
        for i in range(len(val_station_colors)):
            if val_station_ids[i] not in train_station_ids:
                val_station_colors[i] = "green"


        #3. Build the test graph
        test_nx_graph = nx.Graph()
        test_nx_graph.add_nodes_from(range(validation_indices.shape[0]))
        test_nx_graph.add_edges_from(self.test_graph.edge_index.numpy().T)

        test_station_ids = self.raingauge_order
        test_station_locations = []
        for id in test_station_ids:
            lat, lon = self.station_dict[id]
            test_station_locations.append((lon, lat))
        print(test_station_ids)
        test_station_colors = ["blue" for _ in range(len(test_station_locations))]

        #Set validation stations to green and test stations to red
        for i in range(len(test_station_colors)):
            if test_station_ids[i] not in val_station_ids:
                test_station_colors[i]= "red"
            elif test_station_ids[i] not in train_station_ids:
                test_station_colors[i] = "green"

        #4. Plotting
        nx.draw(
            train_nx_graph,
            train_station_locations,
            ax=ax[0]
        )
        nx.draw(
            validation_nx_graph,
            val_station_locations,
            node_color=val_station_colors,
            ax=ax[1]
        )
        nx.draw(
            test_nx_graph,
            test_station_locations,
            node_color=test_station_colors,
            ax=ax[2]
        )

        fig.show()




