import argparse
import json
import time
import yaml
import pandas as pd
import numpy as np
from src.raingauge.utils import load_raingauge_dataset, get_station_coordinate_mappings, filter_uptime
from src.sampling.main import stratified_spatial_kfold_dual
from src.radar.utils import load_radar_dataset

from tqdm import tqdm

from benchmarks.models.kriging import run_kriging_benchmark

def main():
    '''
    The running of the Kriging benchmark is as follows
    1. Load the raingauge data
    2. Run the stratified training split
    3. Use the statistical split to run Kriging
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='ordinary', choices=['ordinary', 'universal', 'ked'],
                         help="Kriging method: 'ordinary', 'universal' (regional_linear drift), "
                              "or 'ked' (external drift using radar reflectivity at station locations)")
    parser.add_argument('--variogram-model', default='spherical',
                         help="pykrige variogram model, e.g. linear/power/gaussian/spherical/exponential")
    args = parser.parse_args()

    #1. Load data
    fold_count = 5
    config_file = 'config.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f)

    uptime_threshold = config['filters']['uptime_threshold']
    start_year = config['dataset_parameters']['start_year']
    end_year = config['dataset_parameters']['end_year']
    raingauge_df, raingauge_mappings_df = load_raingauge_dataset(start=start_year, end=end_year, uptime_threshold=uptime_threshold)
    raingauge_df = raingauge_df.resample('15min').first() #resamples df to 15 mins
    raingauge_mappings = {sid: (row['latitude'], row['longitude']) for sid, row in raingauge_mappings_df.set_index("id").iterrows()}

    radar_df = load_radar_dataset(folder_name='database/sg_radar_data_cropped', cropped=True)

    radar_columns = radar_df.columns
    raingauge_columns = raingauge_df.columns
    merged_df = radar_df.merge(raingauge_df, on="timestamp", how='left')
    raingauge_df = merged_df[raingauge_columns]

    # For KED, keep the radar raster/bounds/transform aligned row-for-row
    # with raingauge_df (both are column slices of the same merged_df, so
    # their indices already match) so run_kriging_benchmark can sample the
    # radar field as the external-drift covariate at each station location.
    radar_aligned_df = merged_df[['data', 'bounds', 'transform']] if args.method == 'ked' else None

    print(raingauge_df.shape)
    print("DEBUG")
    print(raingauge_mappings.keys())
    #2. Get stratified training split
    split_info = stratified_spatial_kfold_dual(
        raingauge_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    #3. Run Kriging for x folds
    total_start = time.time()
    fold_results = {}
    for fold in range(fold_count):
        training_gauges = split_info[fold]['statistical']['train']
        test_gauges = split_info[fold]['statistical']['test']

        # Run kriging
        fold_results[fold] = run_kriging_benchmark(
            raingauge_data=raingauge_df,
            coordinates=raingauge_mappings,
            training_stations=training_gauges,
            test_stations=test_gauges,
            radar_df=radar_aligned_df,
            method=args.method,
            variogram_model=args.variogram_model,
            fold=fold,
            rain_threshold=0.5,
            regression_plot=True,
        )

    total_time = time.time() - total_start

    print(f"\n=== Kriging ({args.method}) benchmark - all folds ===")
    for fold, result in fold_results.items():
        print(f"Fold {fold}: RMSE={result['average_RMSE_loss']:.4f}  "
              f"MAE={result['average_MAE_loss']:.4f}  F1={result['f1']:.4f}  "
              f"POD={result['pod']:.4f}  FAR={result['far']:.4f}  CSI={result['csi']:.4f}  "
              f"PearsonR={result['pearson_r']:.4f}")

    #4. Compile per-fold + aggregate (mean +/- std) results, matching the
    # CNN benchmark's results.json layout for easy cross-model comparison.
    metric_keys = ['average_RMSE_loss', 'average_MAE_loss', 'f1', 'pod', 'far', 'csi', 'pearson_r']
    per_fold = [{k: fold_results[fold][k] for k in metric_keys} for fold in range(fold_count)]

    agg = {k: float(np.mean([r[k] for r in per_fold])) for k in metric_keys}
    agg_std = {f"{k}_std": float(np.std([r[k] for r in per_fold])) for k in metric_keys}

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean +/- std across folds)")
    print(f"{'='*60}")
    for k in metric_keys:
        print(f"  {k}: {agg[k]:.4f} +/- {agg_std[k + '_std']:.4f}")
    print(f"Total time: {total_time:.1f}s")

    summary = {
        "type": "summary",
        "method": args.method,
        "variogram_model": args.variogram_model,
        "total_time_sec": total_time,
        **agg,
        **agg_std,
        "per_fold": per_fold,
        "timestamp": time.time(),
    }

    results_path = f'kriging_results/results_{args.method}.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return summary


main()
