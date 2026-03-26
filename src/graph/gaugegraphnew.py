import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch

import geopandas as gpd
from shapely import STRtree, LineString, Point
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import ToUndirected, AddLaplacianEigenvectorPE
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from typing import Literal

from src.utils import generate_homogeneous_edges, add_homogeneous_edge_attributes_to_data, read_config

config = read_config("config.yaml")

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
        self.fused_test_heterodata=None
        self.fused_train_heterodata=None
        self.fused_validation_heterodata=None
        self.inverse_weighted=True

        self.train_mask, self.val_mask, self.test_mask = self.initialise_masks()

        self.train_graph = self.build_graph("train")
        self.validation_graph = self.build_graph("validation")
        self.test_graph = self.build_graph("test")


        self.train_heterodata = self.fill_heterodata("train")
        self.validation_heterodata = self.fill_heterodata("validation")
        self.test_heterodata = self.fill_heterodata("test")

        self.train_heterodata = ToUndirected()(self.train_heterodata)
        self.validation_heterodata = ToUndirected()(self.validation_heterodata)
        self.test_heterodata = ToUndirected()(self.test_heterodata)

    def get_train_graph(self):
        return self.train_graph

    def get_validation_graph(self):
        self.validation_graph.validation_mask = self.get_validation_graph_mask()
        return self.validation_graph

    def get_validation_graph_mask(self):
        return np.logical_or(self.train_mask, self.val_mask)

    def get_test_graph(self):
        return self.test_graph

    def get_train_heterodata(self, normalize = True):
        if self.fused_train_heterodata:
            return self.fused_train_heterodata
        else:
            return self.train_heterodata

    def get_validation_heterodata(self, normalize = True):
        if self.fused_validation_heterodata:
            return self.fused_validation_heterodata
        else:
            return self.validation_heterodata

    def get_test_heterodata(self, normalize = True):
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
        filtered_mapping_df = self.mapping_df[mask].reset_index()
        coords = filtered_mapping_df[['longitude', 'latitude']].values

        ball_tree = NearestNeighbors(n_neighbors=self.knn+1, algorithm='ball_tree').fit(coords)

        distances, indices = ball_tree.kneighbors(coords)

        for idx, row in filtered_mapping_df.iterrows():
            G.add_node(idx, lat=row['latitude'], lon=row['longitude'])

        for i, neighbors in enumerate(indices):
            for j, neighbor_idx in enumerate(neighbors[1:]):
              dist = distances[i][j + 1]

              if self.inverse_weighted:
                  G.add_edge(i, neighbor_idx, weight=1/dist)
              else:
                  G.add_edge(i, neighbor_idx, weight=dist)

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
        heterodata = HeteroData()
        rainfall_values = torch.tensor(self.raingauge_df.fillna(0).values.T, dtype=torch.float32)
        rainfall_validity = torch.tensor(self.raingauge_df.notna().astype(int).values.T, dtype=torch.int16)
        rainfall_features = torch.stack([rainfall_values, rainfall_validity], dim=2)
        heterodata['raingauge'].x = rainfall_features
        heterodata['raingauge'].y = rainfall_values.unsqueeze(-1)

        match graph:
            case "train":
              mask = torch.tensor(self.train_mask, dtype=bool)
              heterodata['raingauge'].mask = []
              edges = self.train_graph.edges(data=True)
            case "validation":
              mask = torch.tensor(np.logical_or(self.train_mask, self.val_mask), dtype=bool)
              val = self.mapping_df[self.mapping_df['id'].isin(self.validation_gauges) | self.mapping_df['id'].isin(self.train_gauges)]
              heterodata['raingauge'].mask = val['id'].isin(self.validation_gauges).to_numpy()
              edges = self.validation_graph.edges(data=True)
            case "test":
              mask = torch.tensor(np.ones(len(self.test_mask)), dtype=bool)
              heterodata['raingauge'].mask = self.test_mask
              edges = self.test_graph.edges(data=True)

        heterodata['raingauge'].x = heterodata['raingauge'].x[mask]
        heterodata['raingauge'].y = heterodata['raingauge'].y[mask]
        edge_index = []
        edge_attr = []
        for A, B, data in edges:
            edge_index.append([
                A,
                B
            ])
            weight = data['weight']
            edge_attr.append(weight)
        max_attr = max(edge_attr)
        edge_attr = [x / max_attr for x in edge_attr]
        heterodata['raingauge', 'connects', 'raingauge'].edge_index = torch.tensor(edge_index, dtype=int).T
        heterodata['raingauge', 'connects', 'raingauge'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        heterodata['raingauge'].num_nodes = torch.tensor(heterodata['raingauge'].x.shape[0], dtype=torch.int32)

        #Fill laplacian
        if config['dataset_parameters']['include_lpe']:
            temp_graph = Data(
                x=torch.zeros(heterodata['raingauge'].x.shape[0], 1),
                edge_index=heterodata['raingauge', 'connects', 'raingauge'].edge_index,
                num_nodes=heterodata['raingauge'].x.shape[0],
            )
            lpe_transform = AddLaplacianEigenvectorPE(k=4, attr_name='laplacian_pe')
            temp_graph = lpe_transform(temp_graph)
            lpe = temp_graph.laplacian_pe

            timestamps = heterodata['raingauge'].x.shape[1]
            lpe_expanded = lpe.unsqueeze(1).expand(-1, timestamps, -1)
            heterodata['raingauge'].x = torch.cat([heterodata['raingauge'].x, lpe_expanded], dim=2)

        return heterodata


    def add_heterodata(self, heterodata_layer: HeteroData, coords:pd.DataFrame, layer_name: str,knn=4) -> tuple[HeteroData, HeteroData, HeteroData]:
        '''
        Adds layer to the heterodata.
        '''
        if not self.fused_train_heterodata:
            self.fused_train_heterodata = self.train_heterodata.clone()
            self.fused_validation_heterodata = self.validation_heterodata.clone()
            self.fused_test_heterodata = self.test_heterodata.clone()

        for node_type in heterodata_layer.node_types:
            self.fused_train_heterodata[node_type].x = heterodata_layer[node_type].x
            self.fused_validation_heterodata[node_type].x = heterodata_layer[node_type].x
            self.fused_test_heterodata[node_type].x = heterodata_layer[node_type].x

        for edge_type in heterodata_layer.edge_types:
            self.fused_train_heterodata[edge_type].edge_index = heterodata_layer[edge_type].edge_index
            self.fused_validation_heterodata[edge_type].edge_index = heterodata_layer[edge_type].edge_index
            self.fused_test_heterodata[edge_type].edge_index = heterodata_layer[edge_type].edge_index
            self.fused_train_heterodata[edge_type].edge_attr = heterodata_layer[edge_type].edge_attr
            self.fused_validation_heterodata[edge_type].edge_attr = heterodata_layer[edge_type].edge_attr
            self.fused_test_heterodata[edge_type].edge_attr = heterodata_layer[edge_type].edge_attr



        #Connect the raingauge and the radar
        train_raingauge_coords = list(zip(self.mapping_df[self.train_mask]['longitude'],
                                          self.mapping_df[self.train_mask]['latitude']))
        val_raingauge_coords = list(zip(self.mapping_df[np.logical_or(self.train_mask, self.val_mask)]['longitude'],
                                       self.mapping_df[np.logical_or(self.train_mask, self.val_mask)]['latitude']))
        test_raingauge_coords = list(zip(self.mapping_df['longitude'],
                                         self.mapping_df['latitude']))

        if layer_name == "cml":
            train_connecting_edges, train_connecting_edge_weight = self.connect_cml_graph(train_raingauge_coords, coords, knn)
            val_connecting_edges, val_connecting_edge_weight = self.connect_cml_graph(val_raingauge_coords, coords, knn)
            test_connecting_edges, test_connecting_edge_weight = self.connect_cml_graph(test_raingauge_coords, coords, knn)
        else:
            train_connecting_edges, train_connecting_edge_weight = self.connect_graphs(train_raingauge_coords, coords, knn)
            val_connecting_edges, val_connecting_edge_weight = self.connect_graphs(val_raingauge_coords, coords, knn)
            test_connecting_edges, test_connecting_edge_weight = self.connect_graphs(test_raingauge_coords, coords, knn)



        self.fused_train_heterodata['raingauge', 'connects', f'{layer_name}'].edge_index = torch.tensor(train_connecting_edges,dtype=torch.long).T
        self.fused_validation_heterodata['raingauge', 'connects', f'{layer_name}'].edge_index = torch.tensor(val_connecting_edges, dtype=torch.long).T
        self.fused_test_heterodata['raingauge', 'connects', f'{layer_name}'].edge_index = torch.tensor(test_connecting_edges, dtype=torch.long).T

        self.fused_train_heterodata['raingauge', 'connects', f'{layer_name}'].edge_attr = torch.tensor(train_connecting_edge_weight, dtype=torch.float32).T
        self.fused_validation_heterodata['raingauge', 'connects', f'{layer_name}'].edge_attr = torch.tensor(val_connecting_edge_weight, dtype=torch.float32).T
        self.fused_test_heterodata['raingauge', 'connects', f'{layer_name}'].edge_attr = torch.tensor(test_connecting_edge_weight, dtype=torch.float32).T

        if not config['layer_connect']['is_directed']:
            self.fused_train_heterodata[f'{layer_name}', 'rev_connects', 'raingauge'].edge_index = torch.tensor(train_connecting_edges, dtype=torch.long).T.flip(0)
            self.fused_validation_heterodata[f'{layer_name}', 'rev_connects', 'raingauge'].edge_index = torch.tensor(val_connecting_edges, dtype=torch.long).T.flip(0)
            self.fused_test_heterodata[f'{layer_name}', 'rev_connects', 'raingauge'].edge_index = torch.tensor(test_connecting_edges, dtype=torch.long).T.flip(0)

            self.fused_train_heterodata[f'{layer_name}', 'rev_connects', 'raingauge'].edge_attr = torch.tensor(train_connecting_edge_weight, dtype=torch.float32)
            self.fused_validation_heterodata[f'{layer_name}', 'rev_connects', 'raingauge'].edge_attr = torch.tensor(val_connecting_edge_weight, dtype=torch.float32)
            self.fused_test_heterodata[f'{layer_name}', 'rev_connects', 'raingauge'].edge_attr = torch.tensor(test_connecting_edge_weight, dtype=torch.float32)

        return self.fused_train_heterodata, self.fused_validation_heterodata, self.fused_test_heterodata

    def connect_cml_graph(self, raingauge_coords, cml_coords: pd.DataFrame, knn:int) -> tuple[list,list]:
        gauge_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat
             in raingauge_coords],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")

        cml_gdf = gpd.GeoDataFrame(
            cml_coords,
            geometry=[
                LineString([(row['site_a_longitude'], row['site_a_latitude']),
                            (row['site_b_longitude'], row['site_b_latitude'])])
                for _, row in cml_coords.iterrows()
            ],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")

        node_a_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(cml_coords['site_a_longitude'], cml_coords['site_a_latitude']),
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")

        node_b_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(cml_coords['site_b_longitude'], cml_coords['site_b_latitude']),
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")


        tree = STRtree(cml_gdf.geometry)
        K = knn
        edge_list = []
        weight_list = []

        for gauge_idx, gauge_row in gauge_gdf.iterrows():
            gauge_pt = gauge_row.geometry

            distances = cml_gdf.geometry.distance(gauge_pt)
            top_k = distances.nsmallest(K)

            for cml_idx, dist in top_k.items():
                if cml_idx % 2 == 1:
                    cml_idx -= 1
                row = cml_coords.iloc[cml_idx]
                edge_list.append((gauge_idx, cml_idx))
                edge_list.append((gauge_idx, cml_idx + 1))
                station_A = node_a_gdf.iloc[cml_idx]
                station_B = node_b_gdf.iloc[cml_idx]
                weight_A = gauge_pt.distance(station_A) / 1000 #downscale
                weight_B  = gauge_pt.distance(station_B) / 1000 #downscale
                if self.inverse_weighted:
                    weight_A = 1/weight_A
                    weight_B = 1/weight_B
                weight_list.append(float(weight_A.item()))
                weight_list.append(float(weight_B.item()))

        max_weight = max(weight_list)
        weight_list = [x / max_weight for x in weight_list]

        return edge_list, weight_list

    def connect_graphs(self, raingauge_coords, other_coords: pd.DataFrame, knn=16) -> tuple[list, list]:
        edges = []
        A_coords = np.radians(np.array(raingauge_coords))
        B_coords = np.radians(np.array(list(zip(other_coords['latitude'], other_coords['longitude']))))


        # Use haversine metric
        nearestNeighbors = NearestNeighbors(n_neighbors=knn, metric='haversine')
        nearestNeighbors.fit(B_coords)

        distances, indices = nearestNeighbors.kneighbors(A_coords)

        # Create edge list
        edge_list = []
        weight_list = []
        for i in range(len(raingauge_coords)):
            for j in range(knn):
                edge_list.append((i, indices[i, j]))
                if self.inverse_weighted:
                    weight_list.append(1/distances[i, j])
                else:
                    weight_list.append(distances[i, j])

        max_weight = max(weight_list)
        weight_list = [x/max_weight for x in weight_list]

        return edge_list, weight_list

    def get_fused_heterodata(self):
        return self.fused_train_heterodata, self.fused_validation_heterodata, self.fused_test_heterodata


    def visualise_graph_split(self):

        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        # Build DataFrames from masks with reset_index so positions match
        # graph node labels (which are 0-based integers from enumerate in build_graph)
        train_df = self.mapping_df[self.train_mask].reset_index(drop=True)
        val_df = self.mapping_df[np.logical_or(self.train_mask, self.val_mask)].reset_index(drop=True)
        test_df = self.mapping_df.reset_index(drop=True)

        train_pos = {i: (row['longitude'], row['latitude']) for i, row in train_df.iterrows()}
        validation_pos = {i: (row['longitude'], row['latitude']) for i, row in val_df.iterrows()}
        test_pos = {i: (row['longitude'], row['latitude']) for i, row in test_df.iterrows()}

        train_ids = set(self.train_gauges)
        val_ids = set(self.validation_gauges)

        val_colors = [
            'blue' if row['id'] in train_ids else 'green'
            for _, row in val_df.iterrows()
        ]
        test_colors = []
        for _, row in test_df.iterrows():
            if row['id'] in train_ids:
                test_colors.append('blue')
            elif row['id'] in val_ids:
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
            data[edge_type].edge_attr = self.heterodata[edge_type].edge_attr
        data['raingauge'].mask = torch.tensor(self.heterodata['raingauge'].mask)
        data['raingauge'].num_nodes = self.heterodata['raingauge'].x.shape[0]
        for node_type in self.heterodata.node_types:
            if node_type == 'raingauge':
                continue
            data[node_type].x = self.heterodata[node_type].x[:, idx, :]
        return data
