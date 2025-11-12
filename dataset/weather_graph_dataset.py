import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from typing import Literal

class WeatherGraphDataset(Dataset):
  def __init__(self, data, mode = Literal['train', 'test', 'val']):
    assert mode in ['train', 'test', 'val'], f"Invalid mode: must be either 'train' or 'test'."

    self.data = data
    self.mode = mode
    self.num_timesteps = data['general_station'].x.shape[0]

    if mode == 'train':
      self.gen_mask = data['general_station'].train_mask
      self.rain_mask = data['rainfall_station'].train_mask

    elif mode == 'val':
      self.gen_mask = data['general_station'].val_mask
      self.rain_mask = data['rainfall_station'].val_mask

    else: #test
      self.gen_mask = data['general_station'].test_mask
      self.rain_mask = data['rainfall_station'].test_mask

  def __len__(self):
    return self.num_timesteps
  
  def __getitem__(self, idx):
    return{
      'gen_x': self.data['general_station'].x[idx],
      'rain_x': self.data['rainfall_station'].x[idx],
      'gen_y': self.data['general_station'].y[idx],
      'rain_y': self.data['rainfall_station'].y[idx],
    }
class WeatherGraphDatasetNew(Dataset):
  def __init__(self, data, mode = Literal['train', 'test', 'val'], device='cpu'):
    assert mode in ['train', 'test', 'val'], f"Invalid mode: must be either 'train' or 'test'."

    self.data = data.to(device)
    self.mode = mode
    self.device = device
    self.num_timesteps = data['general_station'].x.shape[0]
    self.edge_index_dict = {
        key:val for key, val in self.data.edge_index_dict.items()
    }

    self.edge_attribute_dict = {
      key:val.to(device) for key, val in data.edge_attr_dict.items()
    }


    if mode == 'train':
      self.metastation_mask = data['general_station'].train_mask
      self.rainfallstation_mask = data['rainfall_station'].train_mask

    elif mode == 'val':
      self.metastation_mask = data['general_station'].val_mask
      self.rainfallstation_mask = data['rainfall_station'].val_mask

    else: #test
      self.metastation_mask = data['general_station'].test_mask
      self.rainfallstation_mask = data['rainfall_station'].test_mask

  def __len__(self):
    return self.num_timesteps

  def __getitem__(self, idx):
    return{
      'gen_x': self.data['general_station'].x[idx],
      'rain_x': self.data['rainfall_station'].x[idx],
      'gen_y': self.data['general_station'].y[idx],
      'rain_y': self.data['rainfall_station'].y[idx],
      'metastation_mask': self.metastation_mask,
      'rainfallstation_mask': self.rainfallstation_mask,
      'edge_index_dict': self.edge_index_dict,
      'edge_attr_dict': self.edge_attribute_dict
    }
  
class WeatherGraphDatasetWithRadar(Dataset):
  def __init__(self, data, mode = Literal['train', 'test', 'val']):
    assert mode in ['train', 'test', 'val'], f"Invalid mode: must be either 'train' or 'test'."

    self.data = data
    self.mode = mode
    self.num_timesteps = data['general_station'].x.shape[0]

    if mode == 'train':
      self.gen_mask = data['general_station'].train_mask
      self.rain_mask = data['rainfall_station'].train_mask

    elif mode == 'val':
      self.gen_mask = data['general_station'].val_mask
      self.rain_mask = data['rainfall_station'].val_mask

    else: #test
      self.gen_mask = data['general_station'].test_mask
      self.rain_mask = data['rainfall_station'].test_mask

  def __len__(self):
    return self.num_timesteps
  
  def __getitem__(self, idx):
    return{
      'gen_x': self.data['general_station'].x[idx],
      'rain_x': self.data['rainfall_station'].x[idx],
      "radar_x": self.data['radar_grid'].x[idx],  # Include radar
      'gen_y': self.data['general_station'].y[idx],
      'rain_y': self.data['rainfall_station'].y[idx],
    }
  
class WeatherGraphDatasetWithRadarNew(Dataset):
  def __init__(self, data, mode = Literal['train', 'test', 'val'], device='cpu'):
    assert mode in ['train', 'test', 'val'], f"Invalid mode: must be either 'train' or 'test'."

    self.data = data.to(device)
    self.mode = mode
    self.device = device
    self.num_timesteps = data['general_station'].x.shape[0]
    self.edge_index_dict = {
        key:val for key, val in self.data.edge_index_dict.items()
    }

    self.edge_attribute_dict = {
      key:val.to(device) for key, val in data.edge_attr_dict.items()
    }

    if mode == 'train':
      self.metastation_mask = data['general_station'].train_mask
      self.rainfallstation_mask = data['rainfall_station'].train_mask

    elif mode == 'val':
      self.metastation_mask = data['general_station'].val_mask
      self.rainfallstation_mask = data['rainfall_station'].val_mask

    else: #test
      self.metastation_mask = data['general_station'].test_mask
      self.rainfallstation_mask = data['rainfall_station'].test_mask

  def __len__(self):
    return self.num_timesteps
  
  def __getitem__(self, idx):
    return{
      'gen_x': self.data['general_station'].x[idx],
      'rain_x': self.data['rainfall_station'].x[idx],
      "radar_x": self.data['radar_grid'].x[idx],  # Include radar
      'gen_y': self.data['general_station'].y[idx],
      'rain_y': self.data['rainfall_station'].y[idx],
      'metastation_mask': self.metastation_mask,
      'rainfallstation_mask': self.rainfallstation_mask,
      'edge_index_dict': self.edge_index_dict,
      'edge_attr_dict': self.edge_attribute_dict
    }