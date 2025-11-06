from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
import pandas as pd
from src import *
import numpy as np


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
