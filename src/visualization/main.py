import geopandas as gpd
import contextily as cx
import numpy as np
import pandas as pd

from src.raingauge.utils import get_station_coordinate_mappings


def visualise_singapore_outline(ax=None):
    singapore = gpd.read_file("database/NationalMapPolygon.geojson")
    singapore = singapore.loc[
        [522, 523, 533, 550, 551, 552, 558], :
    ]  # perimeter bounds of singapore
    singapore.boundary.plot(ax=ax)


def visualise_with_basemap(ax=None):
    cx.add_basemap(ax, crs=4326, source=cx.providers.CartoDB.Voyager, alpha=0.5)


def pandas_to_geodataframe(df: pd.Series):
    station_mappings = get_station_coordinate_mappings()
    arr = []

    relevant_cols = [col for col in df.keys() if col in station_mappings]

    for station in relevant_cols:
        val = df[station]
        y, x = station_mappings[station]
        arr.append([x, y, val])

    # conversion from processed df to gpd
    nparr = np.array(arr)
    geometry = gpd.points_from_xy(nparr[:, 0], nparr[:, 1])
    node_df = gpd.GeoDataFrame(geometry=geometry)
    node_df["values"] = nparr[:, 2]

    return node_df
