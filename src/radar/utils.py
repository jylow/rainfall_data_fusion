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

def process_radar_dataset(folder_name: str, crop_bounds: dict) -> pd.DataFrame:
    """
    Process the radar dataset by consolidating data and cropping to crop_bounds.

    All frames are resampled to a single consistent pixel grid derived from the
    first valid file so that every row in the output has the same array shape,
    transform, and bounds — regardless of per-file resolution differences.

    Parameters
    ----------
    folder_name  : str   path to folder containing date subfolders with .tif files
    crop_bounds  : dict  {'left', 'right', 'top', 'bottom'} in the CRS of the TIFs
    """
    from scipy.ndimage import zoom as ndimage_zoom

    target_shape = (31, 50)   # (rows, cols) fixed from the first valid file
    rows = []

    for subdir, dirs, files in os.walk(folder_name):
        for dir_name in sorted(dirs):
            path = os.path.join(folder_name, dir_name)
            for filename in sorted(os.listdir(path)):
                if not filename.endswith(".tif"):
                    continue

                timestamp = datetime.strptime(filename.split("_")[2], "%Y%m%d%H%M")
                # Radar files are in UTC — convert to Singapore Time (UTC+8)
                timestamp = timestamp + pd.Timedelta(hours=8)
                data, bounds, crs, transform = read_tif_file(
                    os.path.join(path, filename)
                )

                # ── Pixel indices for the crop region ──────────────────────
                # Use round() to snap to the nearest pixel; avoids off-by-one
                # errors when crop_bounds don't fall exactly on pixel edges.
                col_start = round((crop_bounds['left']   - bounds.left) / transform[0])
                col_end   = round((crop_bounds['right']  - bounds.left) / transform[0])
                row_start = round((bounds.top - crop_bounds['top'])    / (-transform[4]))
                row_end   = round((bounds.top - crop_bounds['bottom']) / (-transform[4]))

                # Clamp to valid array extents
                col_start = max(0, min(col_start, data.shape[1]))
                col_end   = max(0, min(col_end,   data.shape[1]))
                row_start = max(0, min(row_start, data.shape[0]))
                row_end   = max(0, min(row_end,   data.shape[0]))

                if col_end <= col_start or row_end <= row_start:
                    print(f"  Skipping {filename}: crop region outside raster extent.")
                    continue

                cropped = data[row_start:row_end, col_start:col_end].astype(float)

                # ── Fix resolution differences across timestamps ────────────
                # Lock target_shape from the first valid file.  Any file with a
                # different pixel count (bad resolution) is resampled to match.
                if target_shape is None:
                    target_shape = cropped.shape
                    print(f"Target shape set to {target_shape} from {filename}")

                if cropped.shape != target_shape:
                    print(f"  Resampling {filename}: {cropped.shape} → {target_shape}")
                    zoom_r = target_shape[0] / cropped.shape[0]
                    zoom_c = target_shape[1] / cropped.shape[1]
                    cropped = ndimage_zoom(cropped, (zoom_r, zoom_c), order=1)

                # ── Consistent bounds and transform for all frames ──────────
                # Use pixel-snapped bounds (derived from actual col/row indices)
                # rather than crop_bounds directly.  crop_bounds may not fall on
                # a pixel boundary (e.g. bottom=1.188 snaps to 1.19 at 0.01°
                # resolution), so using crop_bounds would make the stored bounds
                # disagree with the actual data coverage.
                snapped_left   = bounds.left + transform[0] * col_start
                snapped_right  = bounds.left + transform[0] * col_end
                snapped_top    = bounds.top  + transform[4] * row_start
                snapped_bottom = bounds.top  + transform[4] * row_end

                new_bounding_box = BoundingBox(
                    left   = snapped_left,
                    right  = snapped_right,
                    top    = snapped_top,
                    bottom = snapped_bottom,
                )
                # from_bounds(west, south, east, north, width_px, height_px)
                new_transform = from_bounds(
                    snapped_left,  snapped_bottom,
                    snapped_right, snapped_top,
                    target_shape[1], target_shape[0],
                )

                rows.append({
                    "timestamp": timestamp,
                    "data":      cropped,
                    "bounds":    new_bounding_box,
                    "crs":       crs,
                    "transform": new_transform,
                })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_pickle("database/processed_radar_dataset.pkl")
    print(f"Saved {len(df)} radar frames to database/processed_radar_dataset.pkl  (shape {target_shape})")
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
