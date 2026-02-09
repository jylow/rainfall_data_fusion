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
        if self.fused_train_heterodata:
            return self.fused_train_heterodata
        else:
            return self.train_heterodata

    def get_validation_heterodata(self):
        if self.fused_validation_heterodata:
            return self.fused_validation_heterodata
        else:
            return self.validation_heterodata

    def get_test_heterodata(self):
        if self.fused_test_heterodata:
            return self.fused_test_heterodata
        else:
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
              edges = self.train_graph.edges(data=True)
            case "validation":
              self.heterodata['raingauge'].mask = torch.tensor(self.val_mask, dtype=bool)
              edges = self.validation_graph.edges(data=True)
            case "test":
              self.heterodata['raingauge'].mask = torch.tensor(self.test_mask, dtype=bool)
              edges = self.test_graph.edges(data=True)

        
        edge_index = []
        edge_attr = []
        mapping_df_indexed = self.mapping_df.set_index('id')
        for A, B, data in edges:
            edge_index.append([
                mapping_df_indexed.loc[A]['order'],
                mapping_df_indexed.loc[B]['order']
            ])
            weight = data['weight']
            edge_attr.append(weight)
        self.heterodata['raingauge', 'connects', 'raingauge'].edge_index = torch.tensor(edge_index, dtype=int).T
        self.heterodata['raingauge', 'connects', 'raingauge'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        self.heterodata['raingauge'].num_nodes = torch.tensor(self.heterodata['raingauge'].x.shape[0], dtype=torch.int32)
        return self.heterodata


    def add_heterodata(self, radar_heterodata: HeteroData, coords, knn=4):
        self.fused_train_heterodata = self.train_heterodata.clone()
        self.fused_validation_heterodata = self.validation_heterodata.clone()
        self.fused_test_heterodata = self.test_heterodata.clone()

        for node_type in radar_heterodata.node_types:
            self.fused_train_heterodata[node_type].x = radar_heterodata[node_type].x
            self.fused_validation_heterodata[node_type].x = radar_heterodata[node_type].x
            self.fused_test_heterodata[node_type].x = radar_heterodata[node_type].x

        for edge_type in radar_heterodata.edge_types:
            self.fused_train_heterodata[edge_type].edge_index = radar_heterodata[edge_type].edge_index
            self.fused_validation_heterodata[edge_type].edge_index = radar_heterodata[edge_type].edge_index
            self.fused_test_heterodata[edge_type].edge_index = radar_heterodata[edge_type].edge_index

        #Connect the raingauge and the radar
        raingauge_coords = list(zip(self.mapping_df['longitude'], self.mapping_df['latitude']))

        connecting_edges = self.connect_graphs(raingauge_coords, coords)
        connecting_edges_tensor = torch.tensor(connecting_edges).T
        self.fused_train_heterodata['raingauge', 'connects', 'radar'].edge_index = connecting_edges_tensor
        self.fused_validation_heterodata['raingauge', 'connects', 'radar'].edge_index = connecting_edges_tensor
        self.fused_test_heterodata['raingauge', 'connects', 'radar'].edge_index = connecting_edges_tensor

        self.fused_train_heterodata['radar', 'rev_connects', 'raingauge'].edge_index = connecting_edges_tensor.flip(0)
        self.fused_validation_heterodata['radar', 'rev_connects', 'raingauge'].edge_index = connecting_edges_tensor.flip(0)
        self.fused_test_heterodata['radar', 'rev_connects', 'raingauge'].edge_index = connecting_edges_tensor.flip(0)

        return self.fused_train_heterodata, self.fused_validation_heterodata, self.fused_test_heterodata

    def connect_graphs(self, raingauge_coords, other_coords, knn=4):
        edges = []
        A_coords = np.radians(np.array(raingauge_coords))
        B_coords = np.radians(np.array(other_coords))

        
        # Use haversine metric
        nearestNeighbors = NearestNeighbors(n_neighbors=knn, metric='haversine')
        nearestNeighbors.fit(B_coords)
        
        distances, indices = nearestNeighbors.kneighbors(A_coords)

        # Create edge list
        edge_list = []
        for i in range(len(raingauge_coords)):
            for j in range(knn):
                edge_list.append((i, indices[i, j]))
        return edge_list

    def get_fused_heterodata(self):
        return self.fused_train_heterodata, self.fused_validation_heterodata, self.fused_test_heterodata


    def visualise_graph_split(self):

        fig, ax = plt.subplots(1, 3, figsize=(30, 10))
        mappings = self.mapping_df.set_index('id')
        train_df = mappings.loc[list(self.train_graph.nodes)]
        val_df = mappings.loc[list(self.validation_graph.nodes)]
        test_df = mappings.loc[list(self.test_graph.nodes)]
        print(train_df.iloc[0])

        train_pos = {node: (row['longitude'], row['latitude']) 
                 for node, row in train_df.iterrows()}
        validation_pos = {node: (row['longitude'], row['latitude']) 
                for node, row in val_df.iterrows()}
        test_pos = {node: (row['longitude'], row['latitude']) 
                    for node, row in test_df.iterrows()}
        
        train_stations = train_df.index
        val_stations = val_df.index
        val_colors = []
        test_colors = []
        for node in val_df.index:
            if node in train_stations:
                val_colors.append("blue")
            else:
                val_colors.append("green")

        for node in test_df.index:
            if node in train_stations:
                test_colors.append('blue')
            elif node in val_stations:
                test_colors.append('green')
            else:
                test_colors.append('red')
    

        #4. Plotting
        nx.draw(
            self.train_graph,
            pos=train_pos,
            with_labels=True,
            ax=ax[0]
        )
        nx.draw(
            self.validation_graph,
            pos=validation_pos,
            node_color = val_colors,
            with_labels=True,
            ax=ax[1]
        )
        nx.draw(
            self.test_graph,
            pos=test_pos,
            node_color=test_colors,
            with_labels=True,
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
        for edge_type in self.heterodata.edge_types:
            data[edge_type].edge_index = self.heterodata[edge_type].edge_index
            if edge_type == ('raingauge', 'connects', 'raingauge'):
                data[edge_type].edge_attr = self.heterodata[edge_type].edge_attr
        data['raingauge'].mask = self.heterodata['raingauge'].mask
        data['raingauge'].num_nodes = self.heterodata['raingauge'].x.shape[0]
        for node_type in self.heterodata.node_types:
            if node_type == 'raingauge':
                continue
            data[node_type].x = self.heterodata[node_type].x[:, idx, :]
        return data
