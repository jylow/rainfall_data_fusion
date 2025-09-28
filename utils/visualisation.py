import geopandas as gpd
from shapely.geometry import Polygon
import contextily as cx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from utils.load import get_gauge_coordinate_mappings

def visualise_radar_grid(data: pd.Series, ax=None, zoom=None, scaling=None, alpha=0.7, legend=True):
    '''
    Visualise weather data
    Data should contain:
    0. array of values
    1. bounds
    2. EPSG
    3. Transform
    '''

    d=data['data'] #data is wrapped in 2d array
    bounds = data['bounds']
    crs = data['crs']
    transform = data['transform']


    data_arr = d.copy()

    if scaling == "log":
        data_arr = np.log(data_arr)


    rows, cols = data_arr.shape
    pixel_width = transform[0]
    pixel_height = -transform[4]


    x_min = bounds.left
    y_min = bounds.bottom
    x_max = bounds.right
    y_max = bounds.top

    if zoom is None:
        zoom_x_min = bounds.left
        zoom_y_min = bounds.bottom
        zoom_x_max = bounds.right
        zoom_y_max = bounds.top
    else:
        zoom_x_min = zoom['left']
        zoom_y_min = zoom['bottom']
        zoom_x_max = zoom['right']
        zoom_y_max = zoom['top']


    geometries = []
    values = []
    row_indices = []
    col_indices = []

    for row in range(rows):
        COL = 0
        for col in range(cols):

            # Calculate cell boundaries
            x_left = x_min + (col * pixel_width)
            x_right = x_min + ((col + 1) * pixel_width)
            y_bottom = y_max - (row * pixel_height)
            y_top = y_max - ((row + 1) * pixel_height)

            if x_left < zoom_x_min or x_right > zoom_x_max or y_bottom < zoom_y_min or y_top > zoom_y_max:
                continue
            COL += 1
            polygon = Polygon([
                (x_left, y_bottom),   # bottom-left
                (x_right, y_bottom),  # bottom-right
                (x_right, y_top),     # top-right
                (x_left, y_top),      # top-left
                (x_left, y_bottom)    # close polygon
            ])

            geometries.append(polygon)
            values.append(data_arr[row][col])
            row_indices.append(row)
            col_indices.append(col)


    df = gpd.GeoDataFrame({
            'value': values,
            #'row': row_indices,
            #'col': col_indices,
            'geometry': geometries
    }, crs = crs)

    df.plot(column='value',
             ax=ax,
             cmap='viridis',  # colormap
             legend=legend,
             edgecolor='white',
             linewidth=0.1,
             alpha=alpha)
    return



def visualise_gauge_grid(node_df: gpd.GeoDataFrame, country='Singapore', ax=None):

    node_df.plot(ax=ax, markersize=50, alpha=0.7, column="values")

    return

def visualise_gauge_split(station_names: list, station_mappings: dict, split_type: str, ax=None):
    '''
    Takes station names and station mappings and plot the points onto a map to see training split
    '''
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
    node_df['values']=coordinate_nparr[:,2]
    node_df.plot(ax=ax, markersize=50, alpha=0.7, color=color)

    return

def visualise_with_basemap(ax=None):
    cx.add_basemap(ax, crs=4326, source=cx.providers.CartoDB.Voyager)

def pandas_to_geodataframe(df: pd.Series):
    station_mappings = get_gauge_coordinate_mappings()
    arr = []

    relevant_cols = [col for col in df.keys() if col in station_mappings]

    for station in relevant_cols:
        val = df[station]
        y,x = station_mappings[station]
        arr.append([x, y, val])

    #conversion from processed df to gpd
    nparr = np.array(arr)
    geometry = gpd.points_from_xy(nparr[:, 0], nparr[:, 1])
    node_df = gpd.GeoDataFrame(geometry=geometry)
    node_df['values']=nparr[:,2]

    return node_df