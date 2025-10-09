import math
import numpy as np
import pandas as pd
import random
import time
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils.load import load_raingauge_dataset, get_gauge_coordinate_mappings
from utils.visualisation import visualise_gauge_grid, visualise_gauge_split, visualise_with_basemap


def run_IDW_benchmark(raingauge_data: pd.DataFrame, coordinates: dict, training_stations: list, validation_stations: list, power=1, loss_hist=False, x_grid=None, y_grid=None, plot_time_start=None, ax=None, axis_rows=0, axis_cols=0, n_nearest=None):

  '''
  Runs IDW benchmark. A grid will be generated based on the given x and y coordinate values and will be compared with the training
  data.
  ------
  returns: 2D array containing the grid that was interpolated for use in interpolation if necessary
  '''   

  start_time = time.time()

  #Variables for plotting
  axr = axis_rows
  axc = axis_cols
  axcount = 0 #use this variable to keep track of the times variable is plotted
  axtotal = axr * axc
  if axtotal > 0:
     assert(start_time != None)

  #loss histogram display
  loss_data = []

  print(f"training_stations {training_stations}")
  print(f'validation_stations {validation_stations}')
  total_RMSE_loss = 0.0
  total_RMSE_loss_pointwise = 0.0
  instance_count = 0

  for row in tqdm.tqdm(raingauge_data.iterrows()):
    timestamp = row[0]
    row = row[1].fillna(0) ##hacky
    known_x = []
    known_y = []
    known_values = []
    predicted_values = []

    for station in training_stations:
      lat, lon = coordinates[station]
      known_x.append(lon)
      known_y.append(lat)
      known_values.append(row[station])

    predicted_values = idw_interpolation_gridded(x_grid=x_grid,
                                         y_grid=y_grid,
                                         gauge_x=known_x,
                                         gauge_y=known_y,
                                         gauge_z=known_values,
                                         power=power,
                                         n_nearest=n_nearest)
    
    #plotting function to visualise the data. In the function, we are assuming that the timestamps are contiguous
    if plot_time_start != None and timestamp >= plot_time_start and axcount < axtotal:
       axi = ax[axcount//axis_rows][axcount%axis_cols]
       cmap=plt.get_cmap('turbo').copy()
       cmap.set_under('w')
       pc = axi.pcolormesh(
          x_grid, 
          y_grid,
          predicted_values,
          shading='nearest',
          cmap= cmap,
          norm= mpl.colors.BoundaryNorm(boundaries=[0.1, 0.2, 0.5, 1, 2, 4, 7, 10, 20], ncolors=256, extend='both'),
          alpha=0.5,
       )
       axi.set_title(str(timestamp))
       visualise_with_basemap(axi)
       axcount += 1


    RMSE = 0.0
    loss = []
    for station in validation_stations:
       lat, lon = coordinates[station]
       val = row[station]
       resolution = x_grid[1] - x_grid[0]
       r = math.floor((lon - x_grid[0]) / resolution)
       c = math.floor((lat - y_grid[0]) / resolution)
       loss.append(((val - predicted_values[c][r]) ** 2))

       #compare against the grid
      #  result = idw_interpolation(known_points=zip(known_x, known_y), known_values=known_values, target_points=[(lat, lon)])
      #  if loss > 1:
      #   print(f"Predicated: {predicted_values[c][r]}, actual: {val}")
    row_RMSE = np.sqrt(np.mean(np.array(loss)))
    total_RMSE_loss += row_RMSE
    loss_data.append(loss)
    instance_count+=1

  average_RMSE_loss = total_RMSE_loss / instance_count
  end_time = time.time()

  time_taken = end_time - start_time

  print(f"The average RMSE loss was {average_RMSE_loss} mm/hr")
  print(f"The time taken was {time_taken} seconds")

  if loss_hist:
    plt.figure(figsize=(15,8))
    plt.title("Loss histogram (mm/hr)")
    plt.hist(loss_data, bins=30, log=True) #plot on a log scale
    plt.show()

  return average_RMSE_loss

def idw_interpolation_gridded(x_grid, y_grid, gauge_x, gauge_y, gauge_z, power=2, smoothing=0, n_nearest=None):
    """
    Perform Inverse Distance Weighting (IDW) interpolation on a 2D grid.
    
    Parameters:
    -----------
    x_grid : array-like
        1D array of equally spaced x-coordinates for the grid
    y_grid : array-like
        1D array of equally spaced y-coordinates for the grid
    gauge_x : array-like
        1D array of x-coordinates for gauge locations
    gauge_y : array-like
        1D array of y-coordinates for gauge locations
    gauge_z : array-like
        1D array of z-values at gauge locations
    power : float, optional (default=2)
        Power parameter for IDW (higher values give more weight to closer points)
    smoothing : float, optional (default=0)
        Smoothing parameter added to distances to avoid division by zero
    n_nearest : int, optional (default=None)
        Number of nearest gauges to use for interpolation. If None, uses all gauges.
    Returns:
    --------
    z_interpolated : 2D numpy array
        Interpolated z-values on the grid with shape (len(y_grid), len(x_grid))
    """
    # Convert inputs to numpy arrays
    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)
    gauge_x = np.asarray(gauge_x)
    gauge_y = np.asarray(gauge_y)
    gauge_z = np.asarray(gauge_z)
    
    # Create meshgrid for the output grid
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Initialize output array
    z_interpolated = np.zeros_like(X)

    

    # Iterate over each grid point
    for i in range(len(y_grid)):
        for j in range(len(x_grid)):
            # Calculate distances from current grid point to all gauges
            distances = np.sqrt((X[i, j] - gauge_x)**2 + (Y[i, j] - gauge_y)**2)

            # Select only the nearest N gauges if n_nearest is specified
            if n_nearest is not None and n_nearest < len(gauge_x):
                # Get indices of the n_nearest closest gauges
                nearest_indices = np.argpartition(distances, n_nearest)[:n_nearest]
                distances = distances[nearest_indices]
                local_gauge_z = gauge_z[nearest_indices]
            else:
                local_gauge_z = gauge_z
            
            # Add smoothing parameter to avoid division by zero
            distances = distances + smoothing
            
            # Check if any gauge is exactly at this grid point
            if np.any(distances == smoothing):
                # Use the value of the closest gauge
                idx = np.argmin(distances)
                z_interpolated[i, j] = local_gauge_z[idx]
            else:
                # Calculate weights (inverse distance raised to power)
                weights = 1.0 / (distances ** power)
                
                # Normalize weights
                weights = weights / np.sum(weights)
                
                # Calculate interpolated value
                z_interpolated[i, j] = np.sum(weights * local_gauge_z)

    return z_interpolated


def idw_interpolation(known_points, known_values, target_points, power=2, epsilon=1e-12):
    """
    Perform Inverse Distance Weighted (IDW) interpolation.
    
    Parameters:
    -----------
    known_points : array-like, shape (n_known, n_dims)
        Coordinates of known data points
    known_values : array-like, shape (n_known,)
        Values at the known data points
    target_points : array-like, shape (n_target, n_dims)
        Coordinates where interpolation is desired
    power : float, default=2
        Power parameter for inverse distance weighting (higher = more local influence)
    epsilon : float, default=1e-12
        Small value to avoid division by zero when target point coincides with known point
    
    Returns:
    --------
    interpolated_values : ndarray, shape (n_target,)
        Interpolated values at target points
    """
    known_points = np.array(known_points)
    known_values = np.array(known_values)
    target_points = np.array(target_points)
    
    # Ensure arrays have correct dimensions
    if known_points.ndim == 1:
        known_points = known_points.reshape(-1, 1)
    if target_points.ndim == 1:
        target_points = target_points.reshape(-1, 1)
    
    n_target = target_points.shape[0]
    n_known = known_points.shape[0]
    
    interpolated_values = np.zeros(n_target)
    
    for i, target in enumerate(target_points):
        # Calculate distances from target point to all known points
        distances = np.sqrt(np.sum((known_points - target) ** 2, axis=1))
        
        # Handle case where target point coincides with a known point
        if np.any(distances < epsilon):
            # If target point is very close to a known point, use that value
            closest_idx = np.argmin(distances)
            interpolated_values[i] = known_values[closest_idx]
        else:
            # Calculate weights (inverse distance with power)
            weights = 1 / (distances ** power)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Calculate weighted average
            interpolated_values[i] = np.sum(weights * known_values)
    
    return interpolated_values
