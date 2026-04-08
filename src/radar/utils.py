import pandas as pd
import os
from datetime import datetime
import numpy as np
from rasterio.transform import from_bounds
from rasterio.coords import BoundingBox

from src.utils import read_tif_file


class RadarDataObject:
    def __init__(self, data, bounds, crs, transform):
        self.data = data
        self.bounds = bounds
        self.crs = crs
        self.transform = transform

def process_radar_dataset(folder_name: str, crop_bounds: dict)-> pd.DataFrame:
    """
    Process the radar dataset by consolidating data and cropping the data.

    The bounds should be in an array (left, bottom, right, top)
    """
    df = pd.DataFrame()

    tif_folder_path = folder_name
    for subdir, dirs, files in os.walk(tif_folder_path):
        for dir in dirs:
            path = os.path.join(tif_folder_path, dir)
            for filename in os.listdir(path):
                if filename.endswith(".tif"):
                    timestamp = filename.split("_")[2]
                    timestamp = datetime.strptime(timestamp, "%Y%m%d%H%M")
                    data, bounds, crs, transform = read_tif_file(
                        os.path.join(path, filename)
                    )

                    #cropping
                    col_start = int((crop_bounds['left'] - bounds.left) // transform[0])
                    col_end = int((crop_bounds['right'] - bounds.left) // transform[0])
                    row_start = int((bounds.top - crop_bounds['top']) // -transform[4])
                    row_end = int((bounds.top - crop_bounds['bottom']) // -transform[4])

                    new_bounds_left = bounds.left + transform[0] * col_start
                    new_bounds_right = bounds.left + transform[0] * col_end
                    new_bounds_top = bounds.top + transform[4] * row_start
                    new_bounds_bottom = bounds.top + transform[4] * row_end
                    new_bounding_box = BoundingBox(left = new_bounds_left,
                                                   right = new_bounds_right,
                                                   top = new_bounds_top,
                                                   bottom = new_bounds_bottom)

                    new_transform = from_bounds(new_bounds_left,
                                             new_bounds_bottom,
                                             new_bounds_right,
                                             new_bounds_top, (new_bounds_right - new_bounds_left) * 10, (new_bounds_top - new_bounds_bottom) * 10)

                    data = data[row_start:row_end, col_start: col_end]
                    #print(data.shape)
                    #print(new_bounding_box)
                    #print(new_transform)

                    new_row = pd.DataFrame(
                        {
                            "timestamp": [timestamp],
                            "data": [data],
                            "bounds": [new_bounding_box],
                            "crs": [crs],
                            "transform": [new_transform],
                        }
                    )
                    df = pd.concat([df, new_row], ignore_index=True)

    df.to_pickle("database/processed_radar_dataset.pkl")
    return df

def load_processed_dataset(folder_name: str) -> pd.DataFrame | pd.Series:
    df = pd.read_pickle(f"{folder_name}")
    return df

def load_radar_dataset(folder_name: str, cropped=False) -> pd.DataFrame:
    """
    Loads radar dataset into a pandas DataFrame object
    ------
    folder_name: folder that contains data separated into different folders(date of data) and .tif files containing
                 weather radar information
    cropped: boolean. Set to true if preprocessing of tif files was done to crop the images
    """

    df = pd.DataFrame()
    tif_folder_path = folder_name
    print(f"Loading radar TIF files from {tif_folder_path}")

    count = 0

    for subdir, dirs, files in os.walk(tif_folder_path):
        for dir in dirs:
            path = os.path.join(tif_folder_path, dir)
            for filename in os.listdir(path):
                if filename.endswith(".tif"):
                    count += 1
                    timestamp = filename.split("_")[3] if cropped else filename.split("_")[2]
                    timestamp = datetime.strptime(timestamp, "%Y%m%d%H%M")
                    data, bounds, crs, transform = read_tif_file(
                        os.path.join(path, filename)
                    )
                    new_row = pd.DataFrame(
                        {
                            "timestamp": [timestamp],
                            "data": [data],
                            "bounds": [bounds],
                            "crs": [crs],
                            "transform": [transform],
                        }
                    )
                    df = pd.concat([df, new_row], ignore_index=True)

    print("Radar dataset loaded!")
    print(f"The size of dataset is {df.shape}")
    return df
