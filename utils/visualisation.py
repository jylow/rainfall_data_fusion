import geopandas as gpd
from shapely.geometry import Polygon
import contextily as cx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from utils import *
from utils.load import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def improved_visualise_radar_grid(data: pd.Series, ax=None, zoom=None, vmin=0, vmax=None,
                         cmap=plt.get_cmap('turbo').copy(), mask_threshold=0.1, add_basemap=True,
                         title=None, colorbar=True, norm=None):
    """
    Visualize weather radar data with proper geographic context.
    
    Parameters:
    -----------
    data : pd.Series with keys 'data', 'bounds', 'crs', 'transform'
    ax : matplotlib/cartopy axis (optional)
    zoom : dict with 'left', 'right', 'bottom', 'top' (optional)
    vmin, vmax : color scale limits
    cmap : colormap name or object
    mask_threshold : minimum value to display (skip light rain)
    add_basemap : whether to add coastlines and borders
    title : plot title
    colorbar : whether to add colorbar
    """
    
    d = data['data']
    bounds = data['bounds']
    if zoom is None:
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        grid_to_plot = d.copy()
    else:
        extent = [zoom['left'], zoom['right'], zoom['bottom'], zoom['top']]

        clip_left = round((zoom['left'] - bounds.left) / (bounds.right - bounds.left) * d.shape[1])
        clip_right = round((zoom['right'] -bounds.left) / (bounds.right- bounds.left) * d.shape[1])
        clip_top = round((bounds.top - zoom['top']) / (bounds.top - bounds.bottom) * d.shape[0])
        clip_bottom = round((bounds.top - zoom['bottom']) / (bounds.top- bounds.bottom) * d.shape[0])
 

        grid_to_plot = d[clip_top:clip_bottom, clip_left:clip_right]
    

    
    # Mask low values (like kriging does)
    #data_arr = np.ma.masked_where(d < mask_threshold, d)
    data_arr = np.array(grid_to_plot)
    
    # Setup extent
    if zoom is not None:
        print(bounds)

    print(data_arr.shape)

    
    # Plot raster data
    im = ax.imshow(data_arr,
                   extent = extent,
                   origin='upper',
                   cmap=cmap,
                   interpolation='nearest',
                   transform=ccrs.PlateCarree(),
                   alpha=1,
                   norm=norm)
    
    # Add geographic features
    if add_basemap:
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                       alpha=1, linestyle='--')
    
    # Add title
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    
    # Add colorbar
    # if colorbar:
        # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label('Rainfall (mm)', rotation=270, labelpad=15)
    
    return im, ax


def visualise_singapore_outline(ax=None):
    singapore = gpd.read_file('database/NationalMapPolygon.geojson')
    singapore = singapore.loc[[522,523,533,550,551,552,558], :] # perimeter bounds of singapore
    singapore.boundary.plot(ax=ax)
  


def visualise_gauge_grid(node_df: gpd.GeoDataFrame, country='Singapore', ax=None, bounds=None):

    node_df.plot(ax=ax, markersize=50, alpha=0.7, column="values", cmap='turbo', norm=mpl.colors.BoundaryNorm(boundaries=[0.1, 0.2, 0.5, 1, 2, 4, 7, 10, 20], ncolors=256, extend='both'))

    return

def visualise_gauge_radius(node_df: gpd.GeoDataFrame, country='Singapore', ax=None, bounds=None, range=2):

    radius_km = 2
    lat = 1.3  # approximate latitude for Singapore
    
    # Calculate degree offset for 2km
    radius_deg_lat = radius_km / 111.0
    radius_deg_lon = radius_km / (111.0 * np.cos(np.radians(lat)))
    
    # Use average for circular buffer
    radius_deg = (radius_deg_lat + radius_deg_lon) / 2
    
    # Create circles by buffering the points
    circles = node_df.copy()
    circles['geometry'] = circles.geometry.buffer(radius_deg)
    
    # Plot the circles
    circles.plot(
        ax=ax, 
        alpha=0.5, 
        edgecolor='black',
        linewidth=0.5
    )


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
    cx.add_basemap(ax, crs=4326, source=cx.providers.CartoDB.Voyager, alpha=0.5)
    

def pandas_to_geodataframe(df: pd.Series):
    station_mappings = get_station_coordinate_mappings()
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