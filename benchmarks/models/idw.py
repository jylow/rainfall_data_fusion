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
                            fold=0,
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

    print(f"Predictions evaluated = {len(actual_values_arr)}")

    pearson_r, pearson_p = pearsonr(actual_values_arr, predicted_values_arr)

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

        text = f"Pearson r = {pearson_r:.3f}\nRMSE = {average_RMSE_loss:.3f} \n TimestepRMSE = {average_timestep_RMSE:.3f}"
        plt.text( 0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"), )

        plot_bound = max(np.nanmax(actual_values_arr), np.nanmax(predicted_values_arr))
        plt.plot([0, plot_bound], [0, plot_bound], 'r--', label='Perfect prediction')

        plt.xlabel('Actual values (mm/hr)')
        plt.ylabel('Predicted values (mm/hr)')
        plt.title(f'IDW Point Interpolation (n_nearest={n_nearest})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'idw_results/plot{fold}.png')


    return average_RMSE_loss
