import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch_geometric.nn import ChebConv, TAGConv, GATConv
from torch import Tensor
from torch.linalg import vector_norm
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GINEConv, to_hetero, HeteroConv, GCNConv, GATConv, Linear, GraphConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # store constructor arguments
        self.config = dict(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )


        for _ in range(num_layers):
            conv = HeteroConv({
                ('general_station', 'gen_to_gen', 'general_station'): 
                    GraphConv((-1, -1), hidden_channels),
                ('general_station', 'gen_to_rain', 'rainfall_station'): 
                    GraphConv((-1,-1), hidden_channels),
                ('rainfall_station', 'rain_to_gen', 'general_station'): 
                    GraphConv((-1,-1), hidden_channels),
                ('rainfall_station', 'rain_to_rain', 'rainfall_station'): 
                    GraphConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        
        self.lin_rainfall = Linear(hidden_channels, out_channels)
        self.lin_general = Linear(hidden_channels, out_channels)

        # Add layer normalization
        self.norm_general = torch.nn.LayerNorm(hidden_channels)
        self.norm_rainfall = torch.nn.LayerNorm(hidden_channels)

        # torch.nn.init.xavier_uniform_(self.lin_rainfall.weight)
        # torch.nn.init.xavier_uniform_(self.lin_general.weight)
        # torch.nn.init.constant_(self.lin_rainfall.bias, 1.0)
        # torch.nn.init.constant_(self.lin_general.bias, 1.0)
    
    def forward(self, x_dict, edge_index_dict, edge_attributes_dict):
        # Initialize with zeros so first layer only gets neighbor info
        h_dict = {key: torch.zeros_like(x) for key, x in x_dict.items()}
        
        # First layer: aggregate from original features
        for conv in self.convs:
            h_dict = conv({key: x for key, x in x_dict.items()}, 
                            edge_index_dict, edge_attributes_dict)
            # # Use Leaky ReLU in hidden layers
            # h_dict = {key: F.leaky_relu(x, negative_slope=0.01) for key, x in h_dict.items()}
            h_dict = {key: x.relu() for key, x in h_dict.items()}
        
        # # Normalize before output layer
        # h_gen_norm = self.norm_general(h_dict['general_station'])
        # h_rain_norm = self.norm_rainfall(h_dict['rainfall_station'])

        # return {
        #     'general_station': F.softplus(self.lin_general(h_gen_norm)),
        #     'rainfall_station': F.softplus(self.lin_rainfall(h_rain_norm))
        # }
        
        return {
            'general_station': F.relu(self.lin_general(h_dict['general_station'])),
            'rainfall_station': F.relu(self.lin_rainfall(h_dict['rainfall_station']))
        }
    
class HeteroGNN2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # store constructor arguments
        self.config = dict(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )

        # for _ in range(num_layers - 1):
        #     conv = HeteroConv({
        #         ('general_station', 'gen_to_gen', 'general_station'): 
        #             GraphConv((-1, -1), hidden_channels),
        #         ('general_station', 'gen_to_rain', 'rainfall_station'): 
        #             GraphConv((-1, -1), hidden_channels),
        #         ('rainfall_station', 'rain_to_gen', 'general_station'): 
        #             GraphConv((-1, -1), hidden_channels),
        #         ('rainfall_station', 'rain_to_rain', 'rainfall_station'): 
        #             GraphConv((-1, -1), hidden_channels),
        #     }, aggr='mean')
        #     self.convs.append(conv)
        for _ in range(num_layers):
            conv = HeteroConv({
                    ('general_station', 'gen_to_gen', 'general_station'): 
                        GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('general_station', 'gen_to_rain', 'rainfall_station'): 
                        GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('rainfall_station', 'rain_to_gen', 'general_station'): 
                        GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('rainfall_station', 'rain_to_rain', 'rainfall_station'): 
                        GATConv((-1, -1), hidden_channels, add_self_loops=False),
                }, aggr='mean')
            self.convs.append(conv)

            self.lin_rainfall = Linear(hidden_channels, out_channels)
            self.lin_general = Linear(hidden_channels, out_channels)

    
    def forward(self, x_dict, edge_index_dict, edge_attributes_dict):
        # First layer: aggregate from original features
        for conv in self.convs:
            x_dict = conv({key: x for key, x in x_dict.items()}, 
                            edge_index_dict, edge_attributes_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        
        return {
            'general_station': self.lin_general(x_dict['general_station']),
            'rainfall_station': self.lin_rainfall(x_dict['rainfall_station'])
        }

class HeteroGCNGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                    ('general_station', 'gen_to_gen', 'general_station'): 
                        GCNConv(-1, hidden_channels, add_self_loops=True),
                    ('general_station', 'gen_to_rain', 'rainfall_station'): 
                        GCNConv(-1, hidden_channels, add_self_loops=False),
                    ('rainfall_station', 'rain_to_gen', 'general_station'): 
                        GCNConv(-1, hidden_channels, add_self_loops=False),
                    ('rainfall_station', 'rain_to_rain', 'rainfall_station'): 
                        GCNConv(-1, hidden_channels, add_self_loops=True),
                }, aggr='mean')
            self.convs.append(conv)

            self.lin_rainfall = Linear(hidden_channels, out_channels)
            self.lin_general = Linear(hidden_channels, out_channels)

    
    def forward(self, x_dict, edge_index_dict, edge_weights_dict):
        # First layer: aggregate from original features
        for conv in self.convs:
            x_dict = conv({key: x for key, x in x_dict.items()}, 
                            edge_index_dict, edge_weights_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        
        return {
            'general_station': self.lin_general(x_dict['general_station']),
            'rainfall_station': self.lin_rainfall(x_dict['rainfall_station'])
        }
    
    

class HeteroSAGEGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                    ('general_station', 'gen_to_gen', 'general_station'): 
                        SAGEConv(-1, hidden_channels),
                    ('general_station', 'gen_to_rain', 'rainfall_station'): 
                        SAGEConv(-1, hidden_channels),
                    ('rainfall_station', 'rain_to_gen', 'general_station'): 
                        SAGEConv(-1, hidden_channels),
                    ('rainfall_station', 'rain_to_rain', 'rainfall_station'): 
                        SAGEConv(-1, hidden_channels),
                }, aggr='mean')
            self.convs.append(conv)

            self.lin_rainfall = Linear(hidden_channels, out_channels)
            self.lin_general = Linear(hidden_channels, out_channels)

    
    def forward(self, x_dict, edge_index_dict, edge_attributes_dict):
        # First layer: aggregate from original features
        for conv in self.convs:
            x_dict = conv({key: x for key, x in x_dict.items()}, 
                            edge_index_dict, edge_attributes_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        
        return {
            'general_station': self.lin_general(x_dict['general_station']),
            'rainfall_station': self.lin_rainfall(x_dict['rainfall_station'])
        }
    

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # store constructor arguments
        self.config = dict(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )


        for _ in range(num_layers):
            conv = GraphConv((-1, -1), hidden_channels)
            self.convs.append(conv)
        
        self.lin_rainfall = Linear(hidden_channels, out_channels)
        self.lin_general = Linear(hidden_channels, out_channels)

        # Add layer normalization
        self.norm_general = torch.nn.LayerNorm(hidden_channels)
        self.norm_rainfall = torch.nn.LayerNorm(hidden_channels)

        # torch.nn.init.xavier_uniform_(self.lin_rainfall.weight)
        # torch.nn.init.xavier_uniform_(self.lin_general.weight)
        # torch.nn.init.constant_(self.lin_rainfall.bias, 1.0)
        # torch.nn.init.constant_(self.lin_general.bias, 1.0)

    def forward(self, x, edge_index, edge_attributes):
        # First layer: aggregate from original features
        for conv in self.convs:
            x = conv(x, edge_index, edge_attributes)
            x = x.relu()

        
        # # Normalize before output layer
        # h_gen_norm = self.norm_general(h_dict['general_station'])
        # h_rain_norm = self.norm_rainfall(h_dict['rainfall_station'])
        return F.relu(self.lin_general(x))
            