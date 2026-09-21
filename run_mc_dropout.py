"""
run_mc_dropout.py
==================
MC-Dropout uncertainty analysis for a trained GNNInductiveHetero model.

Runs N stochastic forward passes (dropout kept active) per timestep across the
full test period, for every fold, and computes empirical coverage of the
central 90% interval against ground truth. Long-running (hours) — designed to
be resumable: progress is checkpointed per-fold and per-timestep under
experiments/{EXPERIMENT_NAME}/mc_dropout/, so an interrupted run can simply be
re-launched with the same arguments.

Usage
-----
python run_mc_dropout.py                # all folds
python run_mc_dropout.py --folds 0      # single fold, e.g. for a quick test run
python run_mc_dropout.py --folds 0 2 4  # arbitrary subset
"""

from src.sampling.main import stratified_spatial_kfold_dual  # must be first

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_processed_dataset
from src.cml.utils import load_cml_dataset
from src.graph.gaugegraphnew import GaugeGraphNew
from src.graph.radargraph import RadarGraph
from src.graph.cmlgraph import CMLGraph
from src.visualization.error_analysis import (
    predict_on_test_stations_mc_dropout,
    compute_mc_dropout_coverage,
    estimate_obs_noise_variance,
)

# ── Settings ────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "raingauge_cml_radar_final"
N_SAMPLES       = 50
CENTRAL_INTERVAL = 0.90
CHECKPOINT_EVERY = 200
LOG_EVERY        = 1000
# ────────────────────────────────────────────────────────────────────────────

EXPERIMENT_DIR = f"experiments/{EXPERIMENT_NAME}"
OUT_ROOT = Path(EXPERIMENT_DIR) / "mc_dropout"
_DATA_FEATURE_DIM = 2


def infer_arch(weights_path: str):
    sd = torch.load(weights_path, map_location="cpu")
    conv_keys = [k for k in sd if k.startswith("convs.")]
    num_layers = max(int(k.split(".")[1]) for k in conv_keys) + 1
    lin_w = sd["lin.weight"]
    hidden_channels = lin_w.shape[1]

    raingauge_in = _DATA_FEATURE_DIM
    for k, v in sd.items():
        if k.startswith("convs.0.convs.raingauge") and k.endswith("lin_rel.weight"):
            raingauge_in = int(v.shape[1])
            break

    import re
    edge_type_set = set()
    for k in sd:
        m = re.match(r"convs\.0\.convs\.<(.+?)>\.lin_rel\.weight", k)
        if m:
            parts = m.group(1).split("___")
            if len(parts) == 3:
                edge_type_set.add(tuple(parts))
    edge_types = sorted(edge_type_set)

    return num_layers, hidden_channels, raingauge_in, edge_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                         help="Fold indices to run (default: all folds). "
                              "e.g. --folds 0  for a quick single-fold test run.")
    args = parser.parse_args()

    t_start = time.time()
    config = read_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FOLD_COUNT = config["training_params"]["fold_count"]
    DATA_SOURCES = config["datasources"]
    dropout_p = config["model"]["dropout"]

    requested_folds = args.folds if args.folds is not None else list(range(FOLD_COUNT))
    assert all(0 <= f < FOLD_COUNT for f in requested_folds), \
        f"--folds must be within [0, {FOLD_COUNT - 1}], got {requested_folds}"

    print(f"Experiment    : {EXPERIMENT_NAME}")
    print(f"Device        : {device}")
    print(f"N_SAMPLES     : {N_SAMPLES}")
    print(f"Dropout p     : {dropout_p}")
    print(f"Central interval: {CENTRAL_INTERVAL}")
    print(f"Folds to run  : {requested_folds}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Load & align datasets (identical to analysis.ipynb) ────────────────
    uptime_threshold = config["filters"]["uptime_threshold"]
    start_year = config["dataset_parameters"]["start_year"]
    end_year = config["dataset_parameters"]["end_year"]

    time_start = pd.Timestamp(config["analysis"]["time_start"])
    time_end = pd.Timestamp(config["analysis"]["time_end"])

    raingauge_df, mapping_df = load_raingauge_dataset(
        start=start_year, end=end_year, uptime_threshold=uptime_threshold
    )
    raingauge_df = raingauge_df.resample("15min", closed="left", label="left").mean()
    raingauge_df = raingauge_df[raingauge_df.index.minute % 15 == 0]
    raingauge_df = raingauge_df[
        (raingauge_df.index >= time_start) & (raingauge_df.index <= time_end)
    ]
    raingauge_df = raingauge_df.reset_index().rename(columns={"index": "timestamp"})

    radar_df = None
    if "radar" in DATA_SOURCES:
        radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")
        radar_df = radar_df.sort_values("timestamp").reset_index(drop=True)

    cml_df, cml_coordinates_df = None, None
    if "cml" in DATA_SOURCES:
        cml_df, cml_coordinates_df = load_cml_dataset(config["dataset_parameters"]["cml_folder"])
        cml_df = cml_df.sort_values("timestamp").reset_index(drop=True)

    gauge_station_cols = [c for c in raingauge_df.columns if c != "timestamp"]
    radar_cols = list(radar_df.columns) if radar_df is not None else None
    cml_cols = list(cml_df.columns) if cml_df is not None else None

    if radar_df is not None:
        merged_df = radar_df.merge(raingauge_df, how="inner", on="timestamp")
    else:
        merged_df = raingauge_df.copy()
    if cml_df is not None:
        merged_df = merged_df.merge(cml_df, how="inner", on="timestamp")

    merged_df = merged_df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    raingauge_df = merged_df[gauge_station_cols].reset_index(drop=True)
    if radar_df is not None:
        radar_df = merged_df[radar_cols].reset_index(drop=True)
    if cml_df is not None:
        cml_df = merged_df[cml_cols].reset_index(drop=True)
    aligned_timestamps = merged_df["timestamp"].reset_index(drop=True)

    print(f"[{time.time() - t_start:.1f}s] Data aligned. T = {len(aligned_timestamps)}")

    # ── Build graphs for every fold ─────────────────────────────────────────
    layer_cfg = config["layer_connect"]
    split_info = stratified_spatial_kfold_dual(
        mapping_df, seed=config["training_params"]["seed"], plot=False, n_splits=FOLD_COUNT
    )

    # Only build graphs for the requested folds — split_info still spans all
    # FOLD_COUNT folds (fold assignment is computed jointly), but graph
    # construction (esp. CML/radar KNN joins) is the expensive part per fold.
    gauge_graph_dict = {}
    for i in requested_folds:
        gauge_graph = GaugeGraphNew(
            raingauge_df, mapping_df, split_info=split_info[i], knn=layer_cfg["gauge_gauge"],
        )
        if "radar" in DATA_SOURCES:
            radar_graph = RadarGraph(radar_df)
            radar_heterodata = radar_graph.get_radar_heterodata()
            gauge_graph.add_heterodata(
                heterodata_layer=radar_heterodata, coords=radar_graph.grid_coords,
                layer_name="radar", knn=layer_cfg["radar_gauge"],
            )
        if "cml" in DATA_SOURCES:
            cml_graph = CMLGraph(cml_df, cml_coordinates_df)
            cml_heterodata = cml_graph.get_heterodata()
            gauge_graph.add_heterodata(
                heterodata_layer=cml_heterodata, coords=cml_coordinates_df,
                layer_name="cml", knn=layer_cfg["cml_gauge"],
            )
        gauge_graph_dict[i] = gauge_graph

    print(f"[{time.time() - t_start:.1f}s] Graphs built for fold(s) {requested_folds}.")

    # ── Per-fold MC-Dropout inference ───────────────────────────────────────
    fold_summaries = []
    for fold_idx in requested_folds:
        fold_out_dir = OUT_ROOT / f"fold_{fold_idx}"
        summary_path = fold_out_dir / "coverage_summary.json"

        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            if "coverage_overall_predictive" in summary:
                print(f"[FOLD {fold_idx}] already complete (coverage_predictive="
                      f"{summary['coverage_overall_predictive']:.4f}) — skipping")
                fold_summaries.append(summary)
                continue
            print(f"[FOLD {fold_idx}] found old-format summary (pre-predictive-interval) "
                  f"— recomputing with tau^-1 predictive interval, reusing cached MC samples")

        weights_path = f"{EXPERIMENT_DIR}/weather_gnn_best_{fold_idx}.pth"
        num_layers, hidden_channels, raingauge_in, ckpt_edge_types = infer_arch(weights_path)

        model = GNNInductiveHetero(
            in_channels_dict={src: -1 for src in DATA_SOURCES},
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=num_layers,
            edge_types=ckpt_edge_types,
            dropout=dropout_p,
        ).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        test_heterodata = gauge_graph_dict[fold_idx].get_test_heterodata()
        validation_heterodata = gauge_graph_dict[fold_idx].get_validation_heterodata()

        # tau^-1: observation-noise variance from the validation split (no test
        # leakage) — see estimate_obs_noise_variance for why this is needed:
        # dropout spread alone (epistemic-only) is far too narrow to be a
        # meaningful coverage interval on this data.
        sigma2_obs = estimate_obs_noise_variance(model, validation_heterodata, device)
        print(f"[FOLD {fold_idx}] tau^-1 (val residual var) = {sigma2_obs:.5f} mm^2 "
              f"(sigma_obs = {sigma2_obs ** 0.5:.4f} mm)")

        print(f"[FOLD {fold_idx}] starting MC-Dropout inference "
              f"(N={N_SAMPLES}, dropout_p={dropout_p}) ...")
        fold_t0 = time.time()
        samples_arr, actuals_arr, station_ids = predict_on_test_stations_mc_dropout(
            model=model,
            heterodata=test_heterodata,
            mapping_df=mapping_df,
            device=device,
            n_samples=N_SAMPLES,
            output_dir=str(fold_out_dir),
            checkpoint_every=CHECKPOINT_EVERY,
            log_every=LOG_EVERY,
        )
        fold_elapsed = time.time() - fold_t0
        print(f"[FOLD {fold_idx}] inference done in {fold_elapsed / 60:.1f} min.")

        coverage = compute_mc_dropout_coverage(
            samples_arr, actuals_arr, station_ids=station_ids,
            central_interval=CENTRAL_INTERVAL, sigma2_obs=sigma2_obs,
        )
        coverage["coverage_per_station"].to_frame().join([
            coverage["interval_width_per_station"],
            coverage["coverage_per_station_predictive"],
            coverage["interval_width_per_station_predictive"],
        ]).to_csv(fold_out_dir / "per_station_coverage.csv")

        summary = {
            "fold": fold_idx,
            "n_samples": N_SAMPLES,
            "central_interval": CENTRAL_INTERVAL,
            "coverage_overall": coverage["coverage_overall"],
            "mean_interval_width": coverage["mean_interval_width"],
            "tau_inv_obs_noise_var": coverage["tau_inv_obs_noise_var"],
            "sigma_obs": coverage["sigma_obs"],
            "coverage_overall_predictive": coverage["coverage_overall_predictive"],
            "mean_interval_width_predictive": coverage["mean_interval_width_predictive"],
            "n_test_stations": len(station_ids),
            "T": int(samples_arr.shape[0]),
            "elapsed_minutes": fold_elapsed / 60,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[FOLD {fold_idx} COMPLETE] coverage(epistemic)={coverage['coverage_overall']:.4f} "
              f"coverage(predictive)={coverage['coverage_overall_predictive']:.4f} "
              f"(nominal={CENTRAL_INTERVAL}) mean_width(predictive)={coverage['mean_interval_width_predictive']:.4f}")
        fold_summaries.append(summary)

    # ── Aggregate across folds ──────────────────────────────────────────────
    agg_df = pd.DataFrame(fold_summaries)
    agg_df.to_csv(OUT_ROOT / "coverage_summary_all_folds.csv", index=False)
    print("\n=== ALL FOLDS COMPLETE ===")
    print(agg_df[["fold", "coverage_overall", "coverage_overall_predictive",
                   "mean_interval_width_predictive", "elapsed_minutes"]].to_string(index=False))
    print(f"Mean coverage (epistemic-only) across folds: {agg_df['coverage_overall'].mean():.4f} "
          f"(nominal={CENTRAL_INTERVAL})")
    print(f"Mean coverage (predictive, +tau^-1) across folds: {agg_df['coverage_overall_predictive'].mean():.4f} "
          f"(nominal={CENTRAL_INTERVAL})")
    print(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
