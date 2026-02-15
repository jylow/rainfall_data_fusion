import yaml
import pandas as pd
import numpy as np
from src.raingauge.utils import load_raingauge_dataset, get_station_coordinate_mappings, filter_uptime
from src.sampling.main import stratified_spatial_kfold_dual
from src.radar.utils import load_radar_dataset

from tqdm import tqdm

from benchmarks.models.idw import run_IDW_benchmark

def main():
    '''
    The running of the IDW benchmark is as follows
    1. Load the raingauge data
    2. Run the stratified training split
    3. Use the statistics split to run IDW
    '''
    #1. Load data
    fold_count = 5
    config_file = 'config.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f)

    uptime_threshold = config['filters']['uptime_threshold']
    start_year = config['dataset_parameters']['start_year']
    end_year = config['dataset_parameters']['end_year']
    raingauge_df, raingauge_mappings_df = load_raingauge_dataset(start=start_year, end=end_year, uptime_threshold=uptime_threshold)
    raingauge_df = raingauge_df.resample('15min').first() #resamples df to 15 minsA
    raingauge_mappings = {sid: (row['latitude'], row['longitude']) for sid, row in raingauge_mappings_df.set_index("id").iterrows()}

    radar_df = load_radar_dataset(folder_name='database/sg_radar_data_cropped', cropped=True)

    radar_columns = radar_df.columns
    raingauge_columns = raingauge_df.columns
    merged_df = radar_df.merge(raingauge_df, on="timestamp", how='left')
    raingauge_df = merged_df[raingauge_columns]

    print(raingauge_df.shape)
    print("DEBUG")
    print(raingauge_mappings.keys())
    #2. Get stratified training split
    split_info = stratified_spatial_kfold_dual(
        raingauge_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    #3. Run the IDW for x folds
    for fold in range(fold_count):
        training_gauges = split_info[fold]['ml']['train']
        test_gauges = split_info[fold]['ml']['test']

        # Run idw
        run_IDW_benchmark(raingauge_data=raingauge_df,
                          coordinates=raingauge_mappings,
                          training_stations=training_gauges,
                          test_stations=test_gauges,
                          power = 2,
                          fold=fold,
                          n_nearest=10,
                          regression_plot=True
                          )


main()
