import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from typing import Literal


from src.utils import generate_homogeneous_edges, add_homogeneous_edge_attributes_to_data

class GaugeGraphNew():

    def __init__(self, data_df: pd.DataFrame, mapping_df: pd.DataFrame, split_info: dict, knn: int):
        """
        node_feature_dict: contains information on heterogeneous node
        station_lists: reference to maintain mapping of stations to node orderings
        """
        self.dtype = torch.float32
        self.raingauge_df = data_df[mapping_df['id'].values.tolist()]
        self.mapping_df = mapping_df
        self.split_info = split_info
        self.knn = knn
        self.train_gauges = split_info["ml"]['train']
        self.validation_gauges = split_info['ml']['validation']
        self.test_gauges = split_info['ml']['test']
        self.heterodata = HeteroData()

        self.train_mask, self.val_mask, self.test_mask = self.initialise_masks()

        self.train_graph = self.build_graph("train")
        self.validation_graph = self.build_graph("validation")
        self.test_graph = self.build_graph("test")

        self.train_heterodata = self.fill_heterodata("train")
        self.validation_heterodata = self.fill_heterodata("validation")
        self.test_heterodata = self.fill_heterodata("test")

    def get_train_graph(self):
        return self.train_graph

    def get_validation_graph(self):
        self.validation_graph.validation_mask = self.get_validation_graph_mask()
        return self.validation_graph

    def get_validation_graph_mask(self):
        return np.logical_or([self.train_mask, self.val_mask])

    def get_test_graph(self):
        return self.test_graph
    
    def get_train_heterodata(self):
        return self.train_heterodata

    def get_validation_heterodata(self):
        return self.validation_heterodata

    def get_test_heterodata(self):
        return self.test_heterodata

    def build_graph(self, split: str):
        '''
        BUILDS A GRAPH FOR TRAIN/VALIDATION/TEST
        returns an nx graph
        '''
        match split:
            case "train":
                mask = self.train_mask
            case "validation":
                mask = np.logical_or(self.train_mask, self.val_mask)
            case "test":
                mask = np.ones(self.mapping_df.shape[0]).astype(bool)
            case _: #should not reach here
                print("ERROR CODE SHOULD NOT REACH HERE PLEASE LOOK AT BUILD_GRAPH FUNCTION")

        #Build the graph
        G = nx.Graph()
        filtered_mapping_df = self.mapping_df[mask]
        coords = filtered_mapping_df[['longitude', 'latitude']].values

        ball_tree = NearestNeighbors(n_neighbors=self.knn+1, algorithm='ball_tree').fit(coords)

        distances, indices = ball_tree.kneighbors(coords)

        for idx, row in filtered_mapping_df.iterrows():
            G.add_node(row['id'], lat=row['latitude'], lon=row['longitude'])
          
        for i, neighbors in enumerate(indices):
            node_id = filtered_mapping_df.iloc[i]['id']
              
            for j, neighbor_idx in enumerate(neighbors[1:]):
              neighbor_id = filtered_mapping_df.iloc[neighbor_idx]['id']
              dist = distances[i][j + 1]

              G.add_edge(node_id, neighbor_id, weight=dist)


        return G

    def initialise_masks(self):
        '''
        Returns mask tensors
        
        :param self: Description
        '''
        train_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)
        validation_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)
        test_mask = np.zeros(self.mapping_df.shape[0], dtype=bool)

        train_mask[self.mapping_df['order'][self.mapping_df['id'].isin(self.train_gauges)].index.to_list()] = True
        validation_mask[self.mapping_df['order'][self.mapping_df['id'].isin(self.validation_gauges)].index.to_list()] = True
        test_mask[self.mapping_df['order'][self.mapping_df['id'].isin(self.test_gauges)].index.to_list()] = True
        return train_mask, validation_mask, test_mask

    def fill_heterodata(self, graph: str) -> HeteroData:

        self.heterodata['raingauge'].x = torch.tensor(self.raingauge_df.values.T, dtype=torch.float32).unsqueeze(-1)
        self.heterodata['raingauge'].y = torch.tensor(self.raingauge_df.values.T, dtype=torch.float32).unsqueeze(-1)

        match graph:
            case "train":
              self.heterodata['raingauge'].mask = torch.tensor(self.train_mask, dtype=bool)
              edges = self.train_graph.edges()
            case "validation":
              self.heterodata['raingauge'].mask = torch.tensor(self.val_mask, dtype=bool)
              edges = self.validation_graph.edges()
            case "test":
              self.heterodata['raingauge'].mask = torch.tensor(self.test_mask, dtype=bool)
              edges = self.test_graph.edges()

        
        edge_index = []
        mapping_df_indexed = self.mapping_df.set_index('id')
        for A, B in edges:
            edge_index.append([
                mapping_df_indexed.loc[A]['order'],
                mapping_df_indexed.loc[B]['order']
            ])
        self.heterodata['raingauge'].edge_index = torch.tensor(edge_index, dtype=int).T
        self.heterodata['raingauge'].num_nodes = torch.tensor(self.heterodata['raingauge'].x.shape[0], dtype=torch.int32)
        return self.heterodata


















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




class HeterogeneousWeatherGraphDatasetInductive(Dataset):
    def __init__(self, heterodata, device="cpu"):

        self.heterodata = heterodata
        self.device = device

        # Graph has shape [N, T, F]
        self.num_timesteps = heterodata['raingauge'].x.shape[1]

        self.mask = heterodata['raingauge'].mask

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        """Return a PyG Data object for timestep idx"""
        # Node features at this timestep: shape [N, F]
        x = self.heterodata['raingauge'].x[:, idx, :]
        # Labels at this timestep: shape [N, ...]
        y = self.heterodata['raingauge'].y[:, idx, :]
        
        # Create a PyG Data object
        data = HeteroData()
        data['raingauge'].x = x
        data['raingauge'].y = y
        data['raingauge', 'connects', 'raingauge'].edge_index = self.heterodata['raingauge'].edge_index
        data['raingauge'].edge_attr = self.heterodata['raingauge'].edge_attr if hasattr(self.heterodata, 'edge_attr') else None
        data['raingauge'].mask = self.heterodata['raingauge'].mask
        data['raingauge'].num_nodes = self.heterodata['raingauge'].x.shape[0]
        return data
