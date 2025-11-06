import pandas as pd
import xarray as xr


def load_cml_dataset(dataset_name, dataset_folder="database") -> pd.DataFrame:
    """
    Loads cml dataset into a pandas DataFrame object
    ------
    dataset_name: .nc file
    """

    data = xr.open_dataset(f"{dataset_folder}/{dataset_name}")
    cml_df = data.to_dataframe().reset_index()

    return cml_df
