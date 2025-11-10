import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch_geometric.nn import ChebConv, TAGConv, GATConv
from torch import Tensor
from torch.linalg import vector_norm

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GINEConv, to_hetero, HeteroConv, GCNConv, GATConv, Linear, GraphConv, GATv2Conv

class HeteroGNN_WithRadar(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('general_station', 'gen_to_gen', 'general_station'): 
                    GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=1),
                ('general_station', 'gen_to_rain', 'rainfall_station'): 
                    GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=1),
                ('rainfall_station', 'rain_to_gen', 'general_station'): 
                    GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=1),
                ('rainfall_station', 'rain_to_rain', 'rainfall_station'): 
                    GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=1),
                
                # Radar grid connections with edge attributes
                ('radar_grid', 'radar_to_gen', 'general_station'): 
                    GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=1),
                ('radar_grid', 'radar_to_rain', 'rainfall_station'): 
                    GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=1),
                ('general_station', 'gen_to_radar', 'radar_grid'): 
                    GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=1),
                ('rainfall_station', 'rain_to_radar', 'radar_grid'): 
                    GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=1),
            }, aggr='mean')
            self.convs.append(conv)
        
        self.lin_rainfall = Linear(hidden_channels, out_channels)
        self.lin_general = Linear(hidden_channels, out_channels)
        self.lin_radar = Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        for conv in self.convs:
            x_dict = conv(
                x_dict, 
                edge_index_dict, 
                edge_attr_dict  # Pass edge attributes
            )
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        return {
            'general_station': self.lin_general(x_dict['general_station']),
            'rainfall_station': self.lin_rainfall(x_dict['rainfall_station']),
            'radar_grid': self.lin_radar(x_dict['radar_grid'])
        }