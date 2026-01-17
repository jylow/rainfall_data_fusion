import torch
import yaml
import pandas as pd

from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_radar_dataset

def get_dataset(
        raingauge_path: str,
        radar_path: str,
        database_folder = "database",
    ) -> pd.DataFrame:

    '''
    Returns processed dataset
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

    #Get radar dataframe
    radar_filepath = f"{database_folder}/{radar_path}"
    radar_df = load_radar_dataset(folder_name=radar_path, cropped=True)

    combined_df = raingauge_df.merge(radar_df, how='inner', on='time_sgt')
    #Get cml dataframe


    return combined_df


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(config)
    '''
    df = get_dataset(
        raingauge_path ='raingauge_nea_data/2025/weather_station_data_2025.csv',
        radar_path = 'sg_radar_data_cropped',
    )
    print(df.shape)
    '''


if __name__ == '__main__':
    #Read configuration file
    config_file = 'config.yaml'
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    main(cfg)