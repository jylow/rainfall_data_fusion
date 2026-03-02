import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class CMLGraph():

    def __init__(self, df: pd.DataFrame, coordinates_df: pd.DataFrame):
        """
        node_feature_dict: contains information on heterogeneous node
        station_lists: reference to maintain mapping of stations to node orderings

        NOTE: Its a little hardcoded
        """
        self.df = df
        self.coordinates_df = coordinates_df.iloc[::2]

        self.graph = self.build_graph()
        self.datatensor = self.prepare_tensor()
        self.heterodata = self.generate_heterodata()


    def build_graph(self):
        '''
        BUILDS A GRAPH FOR TRAIN/VALIDATION/TEST
        '''

        CML_G = nx.Graph()
        for idx, row in self.coordinates_df.iterrows():
          CML_G.add_node(idx, pos=(row['site_a_longitude'], row['site_a_latitude']))
          CML_G.add_node(idx + 1, pos = (row['site_b_longitude'], row['site_b_latitude']))
          CML_G.add_edge(idx, idx + 1, weight=row['length'])

        return CML_G
    
    def prepare_tensor(self):
        '''
        Docstring for prepare_tensor

        Returns tensor of shape [nodes, timestamps, features]
        
        :param self: Description
        '''
        encoded_df = pd.get_dummies(self.df, columns=['polarization'])
        columns_to_remove = ['site_a_latitude', 'site_a_longitude', 'site_b_latitude', 'site_b_longitude', 'source_site_id', 'sink_site_id', 'resource_name']
        columns_to_keep = list(set(encoded_df.columns) - set(columns_to_remove))

        formatted_cml_df = encoded_df[columns_to_keep].pivot_table(
            index="timestamp", columns=["link_id","station"] 
        )
        formatted_cml_df = formatted_cml_df.reorder_levels(["link_id", "station", None], axis=1).sort_index(axis=1)
        formatted_cml_df = formatted_cml_df[self.coordinates_df['link_id']]
        station_count = self.df['link_id'].unique().shape[0] * 2
        formatted_cml_tensor = torch.tensor(formatted_cml_df.values, dtype=torch.float32)
        formatted_cml_tensor = formatted_cml_tensor.reshape(formatted_cml_df.shape[0], station_count, -1)
        formatted_cml_tensor = formatted_cml_tensor.permute(1, 0, 2)
        self.datatensor = formatted_cml_tensor
        
        return formatted_cml_tensor


    def get_heterodata(self) -> HeteroData:
        return self.heterodata
    
    def generate_heterodata(self):
        #Convert data in dataframe to tensor
        datatensor = self.datatensor

        edges = self.graph.edges(data=True)
        edge_index = []
        edge_attr = []
        for A, B, data in edges:
            edge_index.append([
                A, B
            ])
            weight = data['weight']
            edge_attr.append(weight)

        self.heterodata = HeteroData()
        self.heterodata['cml'].x = datatensor

        self.heterodata['cml', 'connect', 'cml'].edge_index = torch.tensor(edge_index,dtype=torch.long).T
        self.heterodata['cml', 'connect', 'cml'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        self.heterodata = ToUndirected()(self.heterodata)
        return self.heterodata