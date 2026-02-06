import math
import numpy as np
import pandas as pd
import random
import time
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr



def run_IDW_benchmark(raingauge_data: pd.DataFrame, 
                            coordinates: dict, 
                            training_stations: list, 
                            test_stations: list, 
                            power=2, 
                            n_nearest=15,
                            regression_plot=False):
    '''
    Runs IDW benchmark with exact point interpolation.
    For each test station, interpolates using n_nearest training stations.
    
    Parameters:
    -----------
    raingauge_data : pd.DataFrame
        DataFrame with timestamps as index and station IDs as columns
    coordinates : dict
        Dictionary mapping station IDs to (lat, lon) tuples
    training_stations : list
        List of station IDs to use for training
    test_stations : list
        List of station IDs to evaluate
    power : float, optional (default=2)
        Power parameter for IDW
    n_nearest : int, optional (default=15)
        Number of nearest training stations to use for each interpolation
    regression_plot : bool, optional (default=False)
        Whether to show regression plot
        
    Returns:
    --------
    average_RMSE_loss : float
        Root mean squared error across all timestamps and test stations
    '''
    
    start_time = time.time()
    
    actual_values_list = []
    predicted_values_list = []
    
    print(f"Training stations: {training_stations}")
    print(f"Test stations: {test_stations}")
    print(f"Using {n_nearest} nearest neighbors for interpolation")
    
    # Iterate over each timestamp
    for timestamp, row in tqdm.tqdm(raingauge_data.iterrows(), total=len(raingauge_data)):
        timestep_actual_values_list = []
        timestep_predicted_values_list = []
        #Handle missing values
        row = row.dropna()
        #row = row.fillna(0) 
        
        # Get training data for this timestamp
        training_coords = []
        training_values = []
        
        for station in training_stations:
            if station in row.index:
                lat, lon = coordinates[station]
                training_coords.append((lat, lon))
                training_values.append(row[station])
        # Skip if insufficient training data
        if len(training_coords) < n_nearest:
            continue
        
        training_coords = np.array(training_coords)
        training_values = np.array(training_values)
        
        # Interpolate for each test station
        for station in test_stations:
            if station not in row.index:
                continue
                
            test_lat, test_lon = coordinates[station]
            actual_value = row[station]
            
            # Calculate distances from test station to all training stations
            distances = np.sqrt(
                (training_coords[:, 0] - test_lat)**2 + 
                (training_coords[:, 1] - test_lon)**2
            )
            
            # Get indices of n_nearest closest stations
            nearest_indices = np.argpartition(distances, min(n_nearest, len(distances)-1))[:n_nearest]
            nearest_distances = distances[nearest_indices]
            nearest_values = training_values[nearest_indices]
            
            # Perform IDW interpolation
            if np.any(nearest_distances == 0):
                # Test station coincides with a training station
                predicted_value = nearest_values[np.argmin(nearest_distances)]
            else:
                weights = 1.0 / (nearest_distances ** power)
                weights = weights / np.sum(weights)
                predicted_value = np.sum(weights * nearest_values)
            
            timestep_actual_values_list.append(actual_value)
            timestep_predicted_values_list.append(predicted_value)
        actual_values_list.append(np.array(timestep_actual_values_list))
        predicted_values_list.append(np.array(timestep_predicted_values_list))

    timestep_MSE_arr = []
    for i in range(len(actual_values_list)):
        timestep_MSE = np.nanmean((actual_values_list[i] - predicted_values_list[i]) ** 2)
        timestep_MSE_arr.append(timestep_MSE)
    timestep_RMSE_arr = np.sqrt(np.array(timestep_MSE_arr))
    average_timestep_RMSE = np.mean(timestep_RMSE_arr)

    actual_values_arr = np.concat(actual_values_list) #This is a 2d array of no. timestamps * 
    predicted_values_arr = np.concat(predicted_values_list)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual_values_arr) | np.isnan(predicted_values_arr))
    actual_values_arr = actual_values_arr[mask]
    predicted_values_arr = predicted_values_arr[mask]

    
    
    # Calculate MSE and RMSE
    squared_errors = (actual_values_arr - predicted_values_arr) ** 2
    average_MSE_loss = np.mean(squared_errors)
    average_RMSE_loss = np.sqrt(average_MSE_loss)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Average RMSE loss: {average_RMSE_loss:.4f} mm/hr")
    print(f"Average RMSE per timestep: {average_timestep_RMSE:.4f} mm/hr")
    print(f"Average MSE loss: {average_MSE_loss:.4f} mm²/hr²")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Number of predictions: {len(actual_values_arr)}")
    
    # Regression plot
    if regression_plot:
        plt.figure(figsize=(10, 10))
        plt.scatter(actual_values_arr, predicted_values_arr, alpha=0.5)
        
        plot_bound = max(np.nanmax(actual_values_arr), np.nanmax(predicted_values_arr))
        plt.plot([0, plot_bound], [0, plot_bound], 'r--', label='Perfect prediction')
        
        plt.xlabel('Actual values (mm/hr)')
        plt.ylabel('Predicted values (mm/hr)')
        plt.title(f'IDW Point Interpolation (n_nearest={n_nearest})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Calculate Pearson correlation
        pearson_r, pearson_p = pearsonr(actual_values_arr, predicted_values_arr)
        print(f"Pearson correlation: {pearson_r:.4f} (p-value: {pearson_p:.4e})")
    
    return average_RMSE_loss

'''
def run_IDW_benchmark(raingauge_df: pd.DataFrame, coordinates: dict, training_stations: list, test_stations: list, power=1, loss_hist=False, x_grid=None, y_grid=None, plot_time_start=None, ax=None, axis_rows=0, axis_cols=0, n_nearest=None, regression_plot=False):



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
  actual_values_arr = np.zeros(shape=[raingauge_df.shape[0], len(test_stations)])
  predicted_values_arr = np.zeros(shape=[raingauge_df.shape[0], len(test_stations)])

  print(f"training_stations {training_stations}")
  print(f'validation_stations {test_stations}')

  for idx, row in tqdm.tqdm(raingauge_df.iterrows()):
    known_x = []
    known_y = []
    known_values = []
    predicted_values = []

    for station in training_stations:
      if station in row.index:
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
    
    row_predicted_arr = []
    row_actual_arr = []
    for station in test_stations:
      if station in row.index:
        lat, lon = coordinates[station]
        val = row[station]
        resolution = x_grid[1] - x_grid[0]
        r = math.floor((y_grid[0] - lat) / resolution)
        c = math.floor((lon - x_grid[0]) / resolution)
        row_actual_arr.append(val)
        row_predicted_arr.append(predicted_values[r][c])
      else:
        row_actual_arr.append(np.nan)
        row_predicted_arr.append(np.nan)
    
    actual_values_arr[idx] = np.array(row_actual_arr)
    predicted_values_arr[idx] = np.array(row_predicted_arr)


  #Calculate loss
  print("CALCULATING LOS")
  MSE_arr = []
  print(len(actual_values_arr))
  for i in range(len(actual_values_arr)):
     actual = np.array(actual_values_arr[i])
     predicted = np.array(predicted_values_arr[i])
     mask = ~np.isnan(actual)
     timestamp_loss = np.mean((actual[mask] - predicted[mask]) ** 2) #calculating the MSE
     MSE_arr.append(timestamp_loss)

  average_MSE_loss = np.nansum(np.array(MSE_arr)) / raingauge_data.shape[0]
  average_RMSE_loss = np.nansum(np.sqrt(np.array(MSE_arr))) / raingauge_data.shape[0]
  end_time = time.time()

  time_taken = end_time - start_time

  print(f"The average RMSE loss was {average_RMSE_loss} mm/hr")
  print(f"The time taken was {time_taken} seconds")

  if loss_hist:
    plt.figure(figsize=(15,8))
    plt.title("Loss histogram (mm/hr)")
    plt.hist(loss_data, bins=30, log=True) #plot on a log scale
    plt.show()

  if regression_plot:
    plt.figure(figsize=(10,10))
    actual = np.array(actual_values_arr).flatten()
    predicted = np.array(predicted_values_arr).flatten()
    plt.scatter(x=actual, y=predicted)
    plot_bound = max(np.nanmax(actual).astype(int),np.nanmax(predicted).astype(int))
    plt.plot(np.linspace(0,plot_bound,100),
            np.linspace(0,plot_bound,100))
    plt.xlabel('actual_values')
    plt.ylabel('predicted_values')
    plt.show()

    mask = ~np.isnan(actual)
    pearson_r_global, pearson_p_global = pearsonr(actual[mask], predicted[mask])
    print(f"Pearson correlation: {pearson_r_global}")

  return average_MSE_loss

def idw_interpolation():
   pass



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
'''