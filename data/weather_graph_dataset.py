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