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
    cml_df.rename(columns={'time': 'timestamp'}, inplace=True)

    #Process cml_df
    filtered_df = cml_df[['site_a_latitude', 'site_a_longitude', 'site_b_latitude', 'site_b_longitude','station', 'link_id', 'length', 'frequency', 'polarization']]
    cml_coordinates_df = filtered_df.drop_duplicates().dropna().reset_index()

    return cml_df, cml_coordinates_df
