import geopandas as gpd
import matplotlib as mpl
import numpy as np


def visualise_gauge_grid(
    node_df: gpd.GeoDataFrame, country="Singapore", ax=None, bounds=None
):
    node_df.plot(
        ax=ax,
        markersize=50,
        alpha=0.7,
        column="values",
        cmap="turbo",
        norm=mpl.colors.BoundaryNorm(
            boundaries=[0.1, 0.2, 0.5, 1, 2, 4, 7, 10, 20], ncolors=256, extend="both"
        ),
    )

    return


def visualise_gauge_radius(
    node_df: gpd.GeoDataFrame, country="Singapore", ax=None, bounds=None, range=2
):
    radius_km = 2
    lat = 1.3  # approximate latitude for Singapore

    # Calculate degree offset for 2km
    radius_deg_lat = radius_km / 111.0
    radius_deg_lon = radius_km / (111.0 * np.cos(np.radians(lat)))

    # Use average for circular buffer
    radius_deg = (radius_deg_lat + radius_deg_lon) / 2

    # Create circles by buffering the points
    circles = node_df.copy()
    circles["geometry"] = circles.geometry.buffer(radius_deg)

    # Plot the circles
    circles.plot(ax=ax, alpha=0.5, edgecolor="black", linewidth=0.5)


def visualise_gauge_split(
    station_names: list, station_mappings: dict, split_type: str, ax=None
):
    """
    Takes station names and station mappings and plot the points onto a map to see training split
    """
    if split_type == "validation":
        color = "green"
    elif split_type == "training":
        color = "red"

    coordinate_arr = []

    for station in station_names:
        if station not in station_mappings:
            print("Station {station} is not found")
        else:
            y, x = station_mappings[station]
            coordinate_arr.append([x, y, 0])

    coordinate_nparr = np.array(coordinate_arr)
    geometry = gpd.points_from_xy(coordinate_nparr[:, 0], coordinate_nparr[:, 1])
    node_df = gpd.GeoDataFrame(geometry=geometry)
    node_df["values"] = coordinate_nparr[:, 2]
    node_df.plot(ax=ax, markersize=50, alpha=0.7, color=color)

    return
