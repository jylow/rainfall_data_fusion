import pandas as pd
import os
from datetime import datetime

from src.utils import read_tif_file


class RadarDataObject:
    def __init__(self, data, bounds, crs, transform):
        self.data = data
        self.bounds = bounds
        self.crs = crs
        self.transform = transform


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
                    d = RadarDataObject(data, bounds, crs, transform)
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
