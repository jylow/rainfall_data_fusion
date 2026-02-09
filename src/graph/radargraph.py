
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class RadarGraph():

    def __init__(self, df: pd.DataFrame):
        """
        node_feature_dict: contains information on heterogeneous node
        station_lists: reference to maintain mapping of stations to node orderings

        NOTE: Its a little hardcoded
        """
        self.dtype = torch.float32
        self.data = df['data'] #(rows * cols)
        self.bounds = df.iloc[0]['bounds']
        self.x_coords = np.arange(self.bounds.left + 0.005, self.bounds.right, 0.01) # cols
        self.y_coords = np.arange(self.bounds.top - 0.005, self.bounds.bottom, -0.01) # rows

        x_grid, y_grid = np.meshgrid(self.x_coords, self.y_coords)
        self.grid_coords = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        self.graph = self.build_graph()
        self.generate_heterodata()


    def flattened_id(self, row, col):
        return row * len(self.x_coords) + col

    def build_graph(self):
        '''
        BUILDS A GRAPH FOR TRAIN/VALIDATION/TEST
        '''

        G = nx.Graph()

        for row in range(len(self.y_coords)):
            for col in range(len(self.x_coords)):
                node_id = self.flattened_id(row, col)
                G.add_node(node_id, pos = (self.y_coords[row], self.x_coords[col]))

        for row in range(len(self.y_coords)):
            for col in range(len(self.x_coords)):
                node_id = self.flattened_id(row, col)
                neighbors = [
                    (row-1, col-1), (row-1, col), (row-1, row+1),  # top row
                    (row, col-1),             (row, col+1),    # left and right
                    (row+1, col-1), (row+1, col), (row+1, col+1)   # bottom row
                ]

                for nrow, ncol in neighbors:
                    if 0 <= nrow < len(self.y_coords) and 0 <= ncol < len(self.x_coords):
                        neighbor_id = self.flattened_id(nrow, ncol)
                        G.add_edge(node_id, neighbor_id)

        return G
    
    def generate_heterodata(self):
        #Convert data in dataframe to tensor
        stack_data = np.stack(self.data.tolist())
        datatensor = torch.tensor(stack_data.T.reshape(stack_data.shape[1] * stack_data.shape[2], -1))


        self.heterodata = HeteroData()
        self.heterodata['radar'].x = datatensor.unsqueeze(-1)
        self.heterodata['radar', 'connect', 'radar'].edge_index = torch.tensor(list(self.graph.edges())).T
        self.heterodata['radar', 'connect', 'radar'].edge_attr = torch.ones(self.heterodata['radar', 'connect', 'radar'].edge_index.shape[1])
        self.heterodata = ToUndirected()(self.heterodata)
        return self.heterodata

    def get_radar_heterodata(self) -> HeteroData:
        return self.heterodata
