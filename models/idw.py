import math
import numpy as np
import pandas as pd
import random
import time
import tqdm

from utils.load import load_raingauge_dataset, get_gauge_coordinate_mappings


def run_IDW_benchmark(raingauge_data: pd.DataFrame, coordinates: dict, training_split=0.7, seed=42, power=1):

  start_time = time.time()
  random.seed(seed)
  validation_split=1-training_split
  cols = raingauge_data.columns
  station_names = [i for i in cols if i in coordinates.keys()]

  #Perform splitting of stations
  training_stations = random.sample(station_names, int(len(station_names) * training_split))
  validation_stations = [s for s in station_names if s not in training_stations]  
  
  print(f"training_stations {training_stations}")
  print(f'validation_stations {validation_stations}')
  total_RMSE_loss = 0.0
  instance_count = 0

  for row in tqdm.tqdm(raingauge_data.iterrows()):
    row = row[1].fillna(0) ##hacky
    known_points = []
    known_values = []
    target_points = []
    validation_values = []
    predicted_values = []

    for station in training_stations:
      lat, lon = coordinates[station]
      known_points.append([lat, lon])
      known_values.append(row[station])

    for station in validation_stations:
      lat, lon = coordinates[station]
      target_points.append([lat, lon])
      validation_values.append(row[station])

    predicted_values = idw_interpolation(known_points=np.array(known_points),
                                        known_values=np.array(known_values),
                                        target_points=np.array(target_points),
                                        power=power)


    predicted_values = np.array(predicted_values)
    validation_values = np.array(validation_values)
    loss = np.mean(np.sqrt((predicted_values - validation_values) ** 2))
    print(f"batchloss = {loss}")
    total_RMSE_loss += loss
    instance_count+=1

  average_RMSE_loss = total_RMSE_loss / instance_count
  end_time = time.time()

  time_taken = end_time - start_time

  print(f"The average RMSE loss was {average_RMSE_loss} mm/hr")
  print(f"The time taken was {time_taken} seconds")

  return average_RMSE_loss

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



def get_benchmarks(benchmarks_to_run: list):

  for benchmark in benchmarks_to_run:
    if benchmark == 'IDW':

      rg_data = load_raingauge_dataset('rainfall_data.csv')
      rg_stations = get_gauge_coordinate_mappings()

      run_IDW_benchmark(rg_data, rg_stations, )
    else:
      print(f"Benchmark requested: {benchmark} is not available")

  