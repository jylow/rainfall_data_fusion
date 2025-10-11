import xarray as xr
import rasterio
import os
import numpy as np
import geopandas as gpd
import pandas as pd
from datetime import datetime
import yaml

class RadarDataObject:
    def __init__(self, data, bounds, crs, transform):
        self.data = data
        self.bounds = bounds
        self.crs = crs
        self.transform = transform

def read_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''   
    
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        
    return cfg

def read_tif_file(tif_path: str):
    '''
    Reads data from .tif files
    '''   

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds
        crs = src.crs
        transform = src.transform

    return data, bounds, crs, transform


def read_nc_file(filepath: str):
    '''
    Reads data from .nc files
    '''   

    data = xr.open_dataset(filepath)

    return data

def load_raingauge_dataset(dataset_name:str , dataset_folder='database', N=0) -> pd.DataFrame:
    '''
    Loads raingauge dataset into a pandas DataFrame object
    ------
    dataset_name: .csv file
    N: filter for timestamp that contains >= N non-zero datapoints
        file containing dictionary with dataset creation information
    '''   

    path = f"{dataset_folder}/{dataset_name}"
    gauge_df = pd.read_csv(path)

    #format time
    gauge_df['time_sgt'] = gauge_df['time_sgt'].apply(lambda x : datetime.strptime(x, '%Y-%m-%dT%H:%M:00+08:00'))

    #convert to table with stations as columns
    formatted_gauge_df = gauge_df.pivot(index='time_sgt', columns='gid', values='rain_rate')

    data_cols = [col for col in formatted_gauge_df.columns if col != 'time_sg']
    filtered_res = formatted_gauge_df[(formatted_gauge_df[data_cols] > 0).sum(axis=1) >= N]

    return filtered_res

def load_weather_station_dataset(dataset_name: str, dataset_folder='database') -> pd.DataFrame:
    '''
    Loads weather station dataset into a pandas DataFrame object
    ------
    dataset_name: .csv file
    N: filter for timestamp that contains >= N non-zero datapoints
        file containing dictionary with dataset creation information

    '''   

    path = f"{dataset_folder}/{dataset_name}"
    gauge_df = pd.read_csv(path)

    #format time
    gauge_df.rename(columns={"timestamp": "time_sgt", "station_id": "gid"}, inplace=True)
    gauge_df['time_sgt'] = gauge_df['time_sgt'].apply(lambda x : datetime.strptime(x, '%Y-%m-%dT%H:%M:00+08:00'))

    #convert to table with stations as columns
    filtered_res = gauge_df

    return filtered_res

def load_cml_dataset(dataset_name, dataset_folder='database') -> pd.DataFrame:
    '''
    Loads cml dataset into a pandas DataFrame object
    ------
    dataset_name: .nc file
    '''   

    data = xr.open_dataset(f"{dataset_folder}/{dataset_name}")
    cml_df = data.to_dataframe().reset_index()
    
    return cml_df

def load_radar_dataset(folder_name:str , dataset_folder='database') -> pd.DataFrame:
    '''
    Loads radar dataset into a pandas DataFrame object
    ------
    folder_name: folder that contains data separated into different folders(date of data) and .tif files containing
                 weather radar information
    '''   

    df = pd.DataFrame()
    tif_folder_path = f"{dataset_folder}/{folder_name}"

    count = 0

    for subdir, dirs, files in os.walk(tif_folder_path):
        for dir in dirs:
            path = os.path.join(tif_folder_path, dir)
            for filename in os.listdir(path):
                if filename.endswith(".tif"):
                    count+=1
                    timestamp = filename.split('_')[2]
                    timestamp = datetime.strptime(timestamp, "%Y%m%d%H%M")
                    data, bounds, crs, transform = read_tif_file(os.path.join(path, filename))
                    d = RadarDataObject(data,bounds,crs, transform)
                    new_row = pd.DataFrame({'time_sgt': [timestamp], 
                                            'data': [data], 
                                            'bounds': [bounds], 
                                            'crs': [crs], 
                                            'transform':[transform]
                                            })
                    df = pd.concat([df, new_row], ignore_index=True)

    print(f"The size of dataset is {count}")
    return df

def get_gauge_coordinate_mappings() -> dict:
    '''
    Returns dictionary containing the mappings of station names to coordinates for raingauge
    
    dict: [key, (lat,long)]
    ------
    '''   

    gauge_df = pd.read_csv('database/station_locations.csv')
    station_locations_df = get_gauge_stations()
    station_locations = station_locations_df['gid'].to_numpy()
    station_name_to_coordinates = station_locations_df[['gid', 'latitude', 'longitude']].to_numpy()
    station_dict = dict()

    for name, lat, long in station_name_to_coordinates:
        station_dict[name] = (lat, long)

    gauge_df = gauge_df[gauge_df['gid'].isin(station_locations)]

    return station_dict

def get_gauge_stations() -> pd.DataFrame:

    station_locations_df = pd.read_csv('database/station_locations.csv')

    return station_locations_df

def get_station_coordinate_mappings() -> dict:
    '''
    Returns dictionary containing the mappings of station names to coordinates for raingauge
    
    dict: [key, (lat,long)]
    ------
    '''   

    gauge_df = pd.read_csv('database/weather_stations.csv')
    station_locations_df = get_gauge_stations()
    station_locations = station_locations_df['gid'].to_numpy()
    station_name_to_coordinates = station_locations_df[['gid', 'latitude', 'longitude']].to_numpy()
    station_dict = dict()

    for name, lat, long in station_name_to_coordinates:
        station_dict[name] = (lat, long)

    gauge_df = gauge_df[gauge_df['gid'].isin(station_locations)]

    return station_dict

def get_weather_stations() -> pd.DataFrame:

    station_location_df = pd.read_csv('database/weather_stations.csv')
    
    return station_location_df

