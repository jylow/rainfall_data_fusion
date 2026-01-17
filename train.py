import torch
import yaml
import pandas as pd

from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_radar_dataset

def get_dataset(
        raingauge_path: str,
        radar_path: str,
        cml_path: str,
        database_folder = "database",
    ) -> pd.DataFrame:

    '''
    Returns processed dataset
    Processed dataset contains a dataframe of all the aggregated data of the following types
    Rain gauge
    Rain radar
    CML (In progress)
    '''
    # df to be used to combine all the data
    combined_df = pd.DataFrame()

    #Get raingauge dataframe
    raingauge_filepath = f"{database_folder}/{raingauge_path}"
    print(f"Loading raingauge dataset from {raingauge_filepath}")
    raingauge_df = load_raingauge_dataset(
            filepath=raingauge_filepath
    )
    print(f"Loaded raingauge dataset from {raingauge_filepath}")
    print(f"Dataset size: {raingauge_df.shape}")
    return raingauge_df

    '''
    #Get radar dataframe
    radar_filepath = f"{database_folder}/{radar_path}"
    radar_df = load_radar_dataset(folder_name=radar_path, cropped=True)

    combined_df = raingauge_df.merge(radar_df, how='inner', on='time_sgt')
    #Get cml dataframe
    '''

    return combined_df


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(config)
    
    df = get_dataset(
        raingauge_data_path = config['dataset_parameters']['raingauge_file'],
        raingauge_station_info_path = config['dataset_parameters']['raingauge_station_file'],
        radar_path = config['dataset_parameters']['radar_folder'],
        cml_path = config['dataset_parameters']['cml_file'],
    )

    print(df.shape)


if __name__ == '__main__':
    #Read configuration file
    config_file = 'config.yaml'
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    main(cfg)