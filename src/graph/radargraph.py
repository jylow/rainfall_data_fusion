
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data


from src.utils import generate_homogeneous_edges, add_homogeneous_edge_attributes_to_data


class RadarGraph():

    def __init__(self, df: pd.DataFrame, knn: int):
        """
        node_feature_dict: contains information on heterogeneous node
        station_lists: reference to maintain mapping of stations to node orderings

        NOTE: Its a little hardcoded
        """
        self.dtype = torch.float32
        self.data = df['data'] #(rows * cols)
        self.bounds = df.iloc[0]['bounds']
        self.x_coords = np.arange(self.bounds.left + 0.005, self.bounds.right, 0.01)
        self.y_coords = np.arange(self.bounds.top - 0.005, self.bounds.bottom, -0.01)
      


        self.knn = knn

    def build_graph(self, split: str, stations) -> Data:
        '''
        BUILDS A GRAPH FOR TRAIN/VALIDATION/TEST
        '''

        G = nx.graph()

        for row in range(len(self.y_coords)):
            for col in range(len(self.x_coords)):
                node_id = (self.x_coords[row], self.y_coords[col])
                G.add_node(node_id, pos = tuple(self.data[row][col]))

        for row in range(len(self.y_coords)):
            for col in range(len(self.x_coords)):
                neighbors = [
                    (row-1, col-1), (row-1, col), (row-1, row+1),  # top row
                    (i, col-1),             (row, col+1),    # left and right
                    (row+1, col-1), (row+1, col), (row+1, col+1)   # bottom row
                ]

        return split_graph

