import yaml
import pandas as pd
import numpy as np
from src.raingauge.utils import load_raingauge_dataset, get_station_coordinate_mappings, filter_uptime
from src.sampling.main import stratified_spatial_kfold_dual

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
    fold_count = 1
    config_file = 'config.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f)

    raingauge_df = load_raingauge_dataset(f'database/{config['dataset_parameters']['raingauge_file']}')
    raingauge_mappings = get_station_coordinate_mappings(start = 2021, end = 2025)
    filtered_stations = filter_uptime(raingauge_df, uptime_threshold = 0.9)
    raingauge_df = raingauge_df[filtered_stations.keys()]
    raingauge_df = raingauge_df.resample('15min').first() #resamples df to 15 mins
    raingauge_mappings = {k:v for k, v in raingauge_mappings.items() if k in raingauge_df.keys()}

    #2. Get stratified training split
    split_info = stratified_spatial_kfold_dual(
        raingauge_mappings, seed=123, plot=False, n_splits=fold_count
    )

    print(raingauge_df.head(10))
    #3. Run the IDW for x folds
    for fold in range(fold_count):
        training_gauges = split_info[fold]['statistical']['train']
        test_gauges = split_info[fold]['statistical']['test']

        # Run idw
        run_IDW_benchmark(raingauge_data=raingauge_df,
                          coordinates=raingauge_mappings,
                          training_stations=training_gauges,
                          test_stations=test_gauges,
                          power = 2,
                          n_nearest=10,
                          regression_plot=True
                          )


main()
