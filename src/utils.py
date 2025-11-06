import xarray as xr
import rasterio
import yaml


def read_config(config_file):
    """
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    """

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    return cfg


def read_tif_file(tif_path: str):
    """
    Reads data from .tif files
    """

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds
        crs = src.crs
        transform = src.transform

    return data, bounds, crs, transform


def read_nc_file(filepath: str):
    """
    Reads data from .nc files
    """

    data = xr.open_dataset(filepath)

    return data
