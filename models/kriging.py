from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
import pandas as pd
from utils import *
import numpy as np

def kriging_external_drift(df: pd.DataFrame, station_names: list, station_dict: dict, method='KED', variogram_model='linear'):
  '''
  Performs Kriging with external frist on the data. 
  TODO: Make the kriging generalised and not fixed based on bounds
  '''
  row_data = df.dropna()
  data = []

  for s in station_names[1:]:
    lat, long = station_dict[s]
    data.append([long, lat, row_data[s]])

  gauge_data = np.array(data)


  #NOTE: GRID RANGES ARE FIXED
  gridx = np.arange(103.605, 104.05, 0.01)
  gridy = np.arange(1.145, 1.51, 0.01)

  #RADAR FOR USE IN EXTERNAL DRIFT

  radar_grid = row_data['data']
  bounds = row_data['bounds']
  transform = row_data['transform']
  x_min = bounds.left
  y_max = bounds.top
  pixel_width = transform[0]
  pixel_height = -transform[4]

  e_dx = []
  e_dy = []

  for row in range(radar_grid.shape[0]): 
      y = y_max - (row * pixel_height) + pixel_height / 2
      e_dy.append(y)

  for col in range(radar_grid.shape[0]):

      # Calculate middle of cell
      x = x_min + (col * pixel_width) + pixel_width / 2
      e_dx.append(x)

  if np.count_nonzero(gauge_data[:, 2]) < 1:
     return None, None

  e_dx = np.array(e_dx)
  e_dy = np.array(e_dy)
  

  if method == 'KED':
    model = UniversalKriging(
        x=gauge_data[:, 0],
        y=gauge_data[:, 1],
        z=gauge_data[:, 2],
        variogram_model=variogram_model,
        drift_terms=["external_Z"],
        external_drift=radar_grid,
        external_drift_x=e_dx,
        external_drift_y=e_dy,
        pseudo_inv=True
    )

  elif method == 'universal': # Defaults to ordinary universal kriging if all sensors dont collect rain
    model = UniversalKriging(
        gauge_data[:, 0],
        gauge_data[:, 1],
        gauge_data[:, 2],
        variogram_model=variogram_model,
        drift_terms=["regional_linear"],
        pseudo_inv=True
    )

  else: #Defaults to ordinary kriging
     model = OrdinaryKriging(
        x=gauge_data[:, 0],
        y=gauge_data[:, 1],
        z=gauge_data[:, 2],
        variogram_model=variogram_model,
        pseudo_inv=True
     )



  z,ss = model.execute("grid", gridx, gridy)

  return z, ss
