from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
import pandas as pd
from src import *
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, confusion_matrix


def kriging_external_drift(
    df: pd.DataFrame,
    station_names: list,
    station_dict: dict,
    method="KED",
    variogram_model="linear",
):
    """
    Performs Kriging with external drift on the data.
    TODO: Make the kriging generalised and not fixed based on bounds
    """
    row_data = df.dropna()
    data = []

    # if rain gauge value is nan, we do not consider it for kriging
    for s in station_names:
        if s in row_data.index:
            lat, long = station_dict[s]
            data.append([long, lat, row_data[s]])

    gauge_data = np.array(data)

    # NOTE: GRID RANGES ARE FIXED
    gridx = np.arange(103.605, 104.1, 0.01)
    gridy = np.arange(1.145, 1.51, 0.01)
    gridy = gridy[::-1]  # flip along the y

    # RADAR FOR USE IN EXTERNAL DRIFT
    if method == "KED":
        radar_grid = row_data["data"]
        bounds = row_data["bounds"]
        transform = row_data["transform"]
        x_min = bounds.left
        y_max = bounds.top
        pixel_width = transform[0]
        pixel_height = -transform[4]

        e_dx = np.arange(bounds.left + 0.005, bounds.right - 0.005, 0.01)
        e_dy = np.arange(round(bounds.top, 2) - 0.005, round(bounds.bottom, 2), -0.01)

    # Kriging does not work when the gauge data has values that are all 0
    if gauge_data.shape[0] == 0 or np.count_nonzero(gauge_data[:, 2]) < 1:
        return None, None

    if method == "KED":
        model = UniversalKriging(
            x=gauge_data[:, 0],
            y=gauge_data[:, 1],
            z=gauge_data[:, 2],
            variogram_model=variogram_model,
            drift_terms=["external_Z"],
            external_drift=radar_grid,
            external_drift_x=e_dx,
            external_drift_y=e_dy,
            pseudo_inv=True,
        )

    elif (
        method == "universal"
    ):  # Defaults to ordinary universal kriging if all sensors dont collect rain
        model = UniversalKriging(
            gauge_data[:, 0],
            gauge_data[:, 1],
            gauge_data[:, 2],
            variogram_model=variogram_model,
            drift_terms=["regional_linear"],
            pseudo_inv=True,
        )

    else:  # Defaults to ordinary kriging
        model = OrdinaryKriging(
            x=gauge_data[:, 0],
            y=gauge_data[:, 1],
            z=gauge_data[:, 2],
            variogram_model=variogram_model,
            pseudo_inv=True,
        )

    z, ss = model.execute("grid", gridx, gridy)

    return z, ss


def _sample_radar_at_points(data, bounds, transform, lons, lats):
    """Sample a radar raster at given (lon, lat) points (nearest pixel).

    `bounds` / `transform` follow the rasterio convention produced by
    `src.radar.utils.load_radar_dataset` (BoundingBox with .left/.top,
    Affine transform where transform[0] = pixel width, transform[4] =
    -pixel height).

    Returns NaN for points that fall outside the raster extent.
    """
    data = np.asarray(data)
    pixel_width = transform[0]
    pixel_height = -transform[4]

    rows = np.floor((bounds.top - np.asarray(lats)) / pixel_height).astype(int)
    cols = np.floor((np.asarray(lons) - bounds.left) / pixel_width).astype(int)

    n_rows, n_cols = data.shape
    valid = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)

    values = np.full(len(lons), np.nan, dtype=np.float64)
    values[valid] = data[rows[valid], cols[valid]]
    return values


def run_kriging_benchmark(raingauge_data: pd.DataFrame,
                           coordinates: dict,
                           training_stations: list,
                           test_stations: list,
                           radar_df: pd.DataFrame = None,
                           method: str = "ordinary",
                           variogram_model: str = "spherical",
                           fold: int = 0,
                           min_training_stations: int = 4,
                           rain_threshold: float = 0.5,
                           regression_plot: bool = False,
                           return_detailed: bool = False):
    '''
    Runs a Kriging benchmark with exact point interpolation, holding out
    `test_stations` and predicting their values from `training_stations` at
    every timestamp. Mirrors the evaluation protocol of `run_IDW_benchmark`
    so results are directly comparable across models.

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
    radar_df : pd.DataFrame, optional
        Radar data required for method="ked" (external-drift kriging).
        Must have the SAME INDEX as `raingauge_data` (i.e. row i of both
        DataFrames correspond to the same timestamp) with columns 'data'
        (2D raster), 'bounds' (rasterio BoundingBox) and 'transform'
        (rasterio/affine transform) — the schema returned by
        `src.radar.utils.load_radar_dataset`. At each timestamp the radar
        value at each station's location is sampled and used as the
        external-drift covariate. Unused for "ordinary" / "universal".
    method : str, optional (default="ordinary")
        "ordinary" | "universal" (regional_linear drift) | "ked" (external drift)
    variogram_model : str, optional (default="spherical")
        pykrige variogram model, e.g. "linear", "power", "gaussian",
        "spherical", "exponential"
    fold : int, optional (default=0)
        Fold index for saving regression plot
    min_training_stations : int, optional (default=4)
        Minimum number of reporting training stations required to fit a
        variogram at a given timestamp; timestamps with fewer are skipped
    rain_threshold : float, optional (default=0.5)
        Rainfall threshold (mm/hr) for binary rain/no-rain F1 classification
    regression_plot : bool, optional (default=False)
        Whether to save a regression plot
    return_detailed : bool, optional (default=False)
        If True, also returns per-station breakdowns and predictions/actuals
        DataFrames for downstream visualisation (see benchmarks/visualization.py)

    Returns:
    --------
    dict with keys:
        average_RMSE_loss, average_MAE_loss, f1
        and, if return_detailed=True:
        per_station_data, predictions_df, actuals_df, actual_values,
        predicted_values, metrics
    '''
    if method == "ked" and radar_df is None:
        raise ValueError("method='ked' requires radar_df (external drift covariate)")

    start_time = time.time()

    test_coords = np.array([coordinates[s] for s in test_stations])  # (lat, lon)
    test_lats = test_coords[:, 0]
    test_lons = test_coords[:, 1]

    actual_full_rows = []
    predicted_full_rows = []
    valid_timestamps = []
    invalid_count = 0

    print(f"Training stations: {training_stations}")
    print(f"Test stations: {test_stations}")
    print(f"Kriging method: {method} ({variogram_model})")

    for timestamp, row in tqdm.tqdm(raingauge_data.iterrows(), total=len(raingauge_data)):
        row = row.dropna()

        train_lats, train_lons, train_values = [], [], []
        for station in training_stations:
            if station in row.index:
                lat, lon = coordinates[station]
                train_lats.append(lat)
                train_lons.append(lon)
                train_values.append(row[station])

        if len(train_values) < min_training_stations:
            invalid_count += 1
            continue

        present_mask = np.array([s in row.index for s in test_stations])
        if not present_mask.any():
            continue

        train_lats = np.array(train_lats)
        train_lons = np.array(train_lons)
        train_values = np.array(train_values, dtype=float)

        if method == "ked":
            if timestamp not in radar_df.index:
                invalid_count += 1
                continue
            radar_row = radar_df.loc[timestamp]
            radar_data = radar_row["data"]
            if radar_data is None or (np.isscalar(radar_data) and pd.isna(radar_data)):
                invalid_count += 1
                continue

            train_radar = _sample_radar_at_points(
                radar_data, radar_row["bounds"], radar_row["transform"],
                train_lons, train_lats,
            )
            test_radar = _sample_radar_at_points(
                radar_data, radar_row["bounds"], radar_row["transform"],
                test_lons, test_lats,
            )

            # Drop training stations that fall outside the radar extent —
            # they have no drift covariate to krige with.
            radar_valid = ~np.isnan(train_radar)
            if radar_valid.sum() < min_training_stations:
                invalid_count += 1
                continue
            train_lats = train_lats[radar_valid]
            train_lons = train_lons[radar_valid]
            train_values = train_values[radar_valid]
            train_radar = train_radar[radar_valid]

            # Test stations outside the radar extent get no drift value —
            # substitute 0 (no echo) rather than dropping the station.
            test_radar = np.nan_to_num(test_radar, nan=0.0)

        try:
            if np.allclose(train_values, train_values[0]):
                # Zero-variance field (e.g. no rain reported anywhere) - variogram
                # fitting is undefined, so just predict the constant value.
                predicted = np.full(len(test_stations), train_values[0])
            elif method == "universal":
                model = UniversalKriging(
                    train_lons, train_lats, train_values,
                    variogram_model=variogram_model,
                    drift_terms=["regional_linear"],
                    pseudo_inv=True,
                )
                predicted, _ = model.execute("points", test_lons, test_lats)
            elif method == "ked":
                model = UniversalKriging(
                    train_lons, train_lats, train_values,
                    variogram_model=variogram_model,
                    drift_terms=["specified"],
                    specified_drift=[train_radar],
                    pseudo_inv=True,
                )
                predicted, _ = model.execute(
                    "points", test_lons, test_lats,
                    specified_drift_arrays=[test_radar],
                )
            else:  # "ordinary" default
                model = OrdinaryKriging(
                    train_lons, train_lats, train_values,
                    variogram_model=variogram_model,
                    pseudo_inv=True,
                )
                predicted, _ = model.execute("points", test_lons, test_lats)
        except Exception:
            invalid_count += 1
            continue

        # Rainfall cannot be negative, but kriging can undershoot near zero
        predicted = np.clip(np.asarray(predicted), 0, None)

        actual_row = np.full(len(test_stations), np.nan)
        for i, station in enumerate(test_stations):
            if present_mask[i]:
                actual_row[i] = row[station]

        actual_full_rows.append(actual_row)
        predicted_full_rows.append(predicted)
        valid_timestamps.append(timestamp)

    print(f"Invalid/skipped timesteps: {invalid_count}")

    actual_full_arr = np.array(actual_full_rows)      # [T, n_test_stations]
    predicted_full_arr = np.array(predicted_full_rows)

    # Per-timestep RMSE / MAE (ignoring stations with no ground truth that step)
    timestep_MSE_arr = []
    timestep_MAE_arr = []
    for i in range(len(actual_full_arr)):
        diff = actual_full_arr[i] - predicted_full_arr[i]
        if np.all(np.isnan(diff)):
            continue
        timestep_MSE_arr.append(np.nanmean(diff ** 2))
        timestep_MAE_arr.append(np.nanmean(np.abs(diff)))

    timestep_RMSE_arr = np.sqrt(np.array(timestep_MSE_arr))
    average_timestep_RMSE = np.mean(timestep_RMSE_arr)
    average_timestep_MAE = np.mean(np.array(timestep_MAE_arr))

    actual_values_arr = actual_full_arr.flatten()
    predicted_values_arr = predicted_full_arr.flatten()

    mask = ~(np.isnan(actual_values_arr) | np.isnan(predicted_values_arr))
    actual_values_arr = actual_values_arr[mask]
    predicted_values_arr = predicted_values_arr[mask]

    print(f"Predictions evaluated = {len(actual_values_arr)}")

    pearson_r, pearson_p = pearsonr(actual_values_arr, predicted_values_arr)

    squared_errors = (actual_values_arr - predicted_values_arr) ** 2
    absolute_errors = np.abs(actual_values_arr - predicted_values_arr)
    average_MSE_loss = np.mean(squared_errors)
    average_RMSE_loss = np.sqrt(average_MSE_loss)
    average_MAE_loss = np.mean(absolute_errors)

    actual_binary = (actual_values_arr >= rain_threshold).astype(int)
    predicted_binary = (predicted_values_arr >= rain_threshold).astype(int)
    f1 = f1_score(actual_binary, predicted_binary, zero_division=0)

    # POD (Probability of Detection) = TP / (TP + FN) — same value as recall
    # FAR (False Alarm Ratio) = FP / (TP + FP) — NOT the same as 1 - precision
    # CSI (Critical Success Index / Threat Score) = TP / (TP + FP + FN)
    tn, fp, fn, tp = confusion_matrix(actual_binary, predicted_binary, labels=[0, 1]).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"Average RMSE loss: {average_RMSE_loss:.4f} mm/hr")
    print(f"Average RMSE per timestep: {average_timestep_RMSE:.4f} mm/hr")
    print(f"Average MSE loss: {average_MSE_loss:.4f} mm²/hr²")
    print(f"Average MAE loss: {average_MAE_loss:.4f} mm/hr")
    print(f"Average MAE per timestep: {average_timestep_MAE:.4f} mm/hr")
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"F1 Score (threshold={rain_threshold} mm/hr): {f1:.4f}")
    print(f"POD (threshold={rain_threshold} mm/hr): {pod:.4f}")
    print(f"FAR (threshold={rain_threshold} mm/hr): {far:.4f}")
    print(f"CSI (threshold={rain_threshold} mm/hr): {csi:.4f}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Number of predictions: {len(actual_values_arr)}")

    if regression_plot:
        plt.figure(figsize=(10, 10))
        plt.scatter(actual_values_arr, predicted_values_arr, alpha=0.5)

        text = (
            f"Pearson r = {pearson_r:.3f}\n"
            f"RMSE = {average_RMSE_loss:.3f} mm/hr\n"
            f"TimestepRMSE = {average_timestep_RMSE:.3f} mm/hr\n"
            f"MAE = {average_MAE_loss:.3f} mm/hr\n"
            f"F1 = {f1:.3f} (threshold={rain_threshold} mm/hr)\n"
            f"POD = {pod:.3f}  FAR = {far:.3f}  CSI = {csi:.3f}"
        )
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment="top",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"))

        plot_bound = max(np.nanmax(actual_values_arr), np.nanmax(predicted_values_arr))
        plt.plot([0, plot_bound], [0, plot_bound], 'r--', label='Perfect prediction')

        plt.xlabel('Actual values (mm/hr)')
        plt.ylabel('Predicted values (mm/hr)')
        plt.title(f'Kriging ({method}, {variogram_model}) Point Interpolation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'kriging_results/{method}_plot{fold}.png')
        plt.close()

    result = {
        "average_RMSE_loss": average_RMSE_loss,
        "average_MAE_loss": average_MAE_loss,
        "f1": f1,
        "pod": pod,
        "far": far,
        "csi": csi,
        "pearson_r": pearson_r,
    }

    if return_detailed:
        per_station_data = {}
        for i, station in enumerate(test_stations):
            col_actual = actual_full_arr[:, i]
            col_predicted = predicted_full_arr[:, i]
            valid = ~np.isnan(col_actual)
            per_station_data[station] = {
                "actual": col_actual[valid].tolist(),
                "predicted": col_predicted[valid].tolist(),
            }

        predictions_df = pd.DataFrame(predicted_full_arr, index=pd.Index(valid_timestamps, name=raingauge_data.index.name),
                                       columns=test_stations)
        actuals_df = pd.DataFrame(actual_full_arr, index=pd.Index(valid_timestamps, name=raingauge_data.index.name),
                                   columns=test_stations)

        result.update({
            "per_station_data": per_station_data,
            "predictions_df": predictions_df,
            "actuals_df": actuals_df,
            "actual_values": actual_values_arr,
            "predicted_values": predicted_values_arr,
            "metrics": {
                "rmse": average_RMSE_loss,
                "mae": average_MAE_loss,
                "f1": f1,
                "pod": pod,
                "far": far,
                "csi": csi,
                "pearson_r": pearson_r,
            },
        })

    return result
