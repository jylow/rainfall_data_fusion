"""
MC-dropout uncertainty analysis (inference-only, no retraining).

Addresses reviewer comment: "No uncertainty quantification, which limits
operational utility in hydrological decision-making."

The trained HGNN already uses dropout (p=0.15). At inference we keep dropout
active and run the model T times per test batch, giving T stochastic
predictions per target node v: y_hat_v^(1..T). We report:
  - mu_v    = mean of the T predictions
  - sigma_v = std of the T predictions (uncertainty proxy)
  - [q05_v, q95_v] = central 90% interval (empirical quantiles over T)
  - Coverage_90 = fraction of held-out gauge values falling inside their interval

TWO coverage numbers are reported:

  coverage_90            — the epistemic-only interval above (empirical
                           quantiles of the T dropout samples). This is the
                           original definition and is kept UNCHANGED so it
                           stays comparable across sites/collaborators.

  coverage_90_predictive — Gal & Ghahramani (2016) predictive variance for
                           MC-dropout regression:

                               sigma^2_pred = sigma^2_dropout + tau^-1

                           where tau^-1 is the observation-noise term,
                           estimated here as the residual variance on the
                           VALIDATION split (never the test split — that
                           would leak). The interval is the Gaussian
                           mu +/- z * sigma_pred, lower bound clamped at 0
                           since rainfall is non-negative.

Why the second number exists: dropout alone captures only *epistemic* (model)
uncertainty. On this data sigma_dropout ~= 0.05 mm while the model's actual
RMSE ~= 0.75 mm, so the epistemic-only interval is ~15x too narrow to ever
cover a real observation. Worse, ~90% of hourly targets are exactly 0 mm and
predictions are clamped at 0, which makes the epistemic coverage on dry nodes
*identically* equal to the fraction of nodes whose lower quantile clamped to
zero — i.e. a threshold test on the sign of a tiny number, not a calibration
measure. That is why it swings wildly between folds (0.02 to 0.89) and even
between machines/torch versions. Adding tau^-1 restores the observation-noise
term the epistemic-only form omits, so the interval reflects the model's real
error scale.

Mirrors the data/model setup in train.py (same convention as
train_optuna_full.py: this codebase does not factor that setup into a shared
module, so it is reproduced here) but loads an already-trained checkpoint
instead of training one.

Usage:
    python mc_dropout_uncertainty.py \\
        --config config_gauge_radar.yaml \\
        --experiment-dir experiments/20260616_135919_sydney_2fold_radar_lat \\
        --fold 0 --T 50
"""
from src.sampling.main import stratified_spatial_kfold_dual  # must init first (see train.py)

import argparse
import json
import os
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeometricDataLoader

from models.gnn import GNNInductiveHetero
from src.raingauge.utils import load_raingauge_dataset
from src.utils import read_config
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive
from training.logic_hetero import bg_predict

_DATA_FEATURE_DIM = 2  # zero rain value + validity flag; matches training/logic_hetero.py masking


def _log_feature(x, log_transform):
    if not log_transform:
        return x
    x = x.clone()
    x[..., 0] = torch.log1p(x[..., 0].clamp(min=0.0))
    return x


def compute_norm_stats(heterodata, log_transform):
    """Same as train.py compute_norm_stats — must match exactly for checkpoint compatibility."""
    stats = {}
    for node_type in heterodata.node_types:
        x = _log_feature(heterodata[node_type].x, log_transform)
        mean = x.mean(dim=(0, 1))
        std = x.std(dim=(0, 1)).clamp(min=1e-8)
        stats[node_type] = (mean, std)
    return stats


def apply_norm(heterodata, stats, log_transform):
    """Same as train.py apply_norm."""
    normed = heterodata.clone()
    for node_type in heterodata.node_types:
        if node_type in stats:
            mean, std = stats[node_type]
            x = _log_feature(heterodata[node_type].x, log_transform)
            normed[node_type].x = (x - mean) / std
    return normed


def enable_mc_dropout(model):
    """Put the model in eval mode (freezes LayerNorm etc.) but leave Dropout
    layers stochastic, so only dropout — not any other eval/train-dependent
    layer — drives the sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_dropout_predict(model, dataloader, device, T, use_bg, use_tweedie):
    """
    Run T stochastic forward passes over the test set with dropout active.

    Returns
    -------
    preds_TxN : torch.Tensor, shape [T, N]  (N = number of valid test-node samples)
    targets_N : torch.Tensor, shape [N]
    station_ids_N : torch.Tensor, shape [N]
    """
    enable_mc_dropout(model)

    preds_per_t = []
    targets_N = None
    station_ids_N = None
    validity_N = None

    with torch.no_grad():
        for t in range(T):
            preds_t = []
            targets_t, station_ids_t, validity_t = [], [], []
            for batch in dataloader:
                batch = batch.to(device)
                x = batch['raingauge'].x
                y = batch['raingauge'].y
                mask = batch['raingauge'].mask
                validity = batch['raingauge'].validity
                edge_index = batch.edge_index_dict
                num_graphs = batch['raingauge'].ptr.size(0) - 1
                num_nodes = x.shape[0] // num_graphs

                x_masked = x.clone()
                x_masked[mask, :_DATA_FEATURE_DIM] = 0.0

                edge_attr_dict = {
                    edge_type: batch[edge_type].edge_attr
                    for edge_type in batch.edge_types
                    if hasattr(batch[edge_type], 'edge_attr')
                }

                x_dict = {nt: batch[nt].x for nt in batch.node_types}
                x_dict['raingauge'] = x_masked

                out = model(x_dict, edge_index, edge_attr_dict)
                if use_bg:
                    out['raingauge'] = bg_predict(out['raingauge'])
                elif use_tweedie:
                    out['raingauge'] = F.softplus(out['raingauge']).clamp(min=0.0)
                else:
                    out['raingauge'] = out['raingauge'].clamp(min=0.0)

                preds_t.append(out['raingauge'][mask].detach().cpu())
                if t == 0:
                    targets_t.append(y[mask].detach().cpu())
                    station_ids_t.append(
                        (mask.nonzero(as_tuple=False).squeeze() % num_nodes).cpu()
                    )
                    validity_t.append(validity[mask].detach().cpu())

            preds_per_t.append(torch.cat(preds_t, dim=0))
            if t == 0:
                targets_N = torch.cat(targets_t, dim=0)
                station_ids_N = torch.cat(station_ids_t, dim=0)
                validity_N = torch.cat(validity_t, dim=0)

    preds_TxN = torch.stack(preds_per_t, dim=0).squeeze(-1)  # [T, N]

    # Drop nodes where the held-out target was NaN (filled with 0) — same
    # filtering as test_model(), so results stay comparable to reported metrics.
    valid_mask = validity_N > 0.5
    preds_TxN = preds_TxN[:, valid_mask]
    targets_N = targets_N[valid_mask].squeeze(-1)
    station_ids_N = station_ids_N[valid_mask]

    return preds_TxN, targets_N, station_ids_N


@torch.no_grad()
def deterministic_predict(model, dataloader, device, use_bg, use_tweedie):
    """
    Single deterministic pass (dropout OFF) over a split. Same masking and
    output mapping as mc_dropout_predict, so residuals are computed exactly
    the way test-time predictions are made.

    Returns (preds_N, targets_N) as 1-D tensors, validity-filtered.
    """
    model.eval()
    preds, targets, validity_all = [], [], []

    for batch in dataloader:
        batch = batch.to(device)
        x = batch['raingauge'].x
        y = batch['raingauge'].y
        mask = batch['raingauge'].mask
        validity = batch['raingauge'].validity

        x_masked = x.clone()
        x_masked[mask, :_DATA_FEATURE_DIM] = 0.0

        edge_attr_dict = {
            et: batch[et].edge_attr
            for et in batch.edge_types
            if hasattr(batch[et], 'edge_attr')
        }
        x_dict = {nt: batch[nt].x for nt in batch.node_types}
        x_dict['raingauge'] = x_masked

        out = model(x_dict, batch.edge_index_dict, edge_attr_dict)
        if use_bg:
            out['raingauge'] = bg_predict(out['raingauge'])
        elif use_tweedie:
            out['raingauge'] = F.softplus(out['raingauge']).clamp(min=0.0)
        else:
            out['raingauge'] = out['raingauge'].clamp(min=0.0)

        preds.append(out['raingauge'][mask].detach().cpu())
        targets.append(y[mask].detach().cpu())
        validity_all.append(validity[mask].detach().cpu())

    preds = torch.cat(preds, dim=0).squeeze(-1)
    targets = torch.cat(targets, dim=0).squeeze(-1)
    validity_all = torch.cat(validity_all, dim=0)

    keep = validity_all > 0.5
    return preds[keep], targets[keep]


def estimate_obs_noise_var(model, val_loader, device, use_bg, use_tweedie):
    """
    Estimate the observation-noise term tau^-1 of Gal & Ghahramani (2016) as
    the residual variance on the VALIDATION split.

    Using validation (not test) keeps the predictive interval honest — the
    test targets are what we then measure coverage against, so calibrating
    on them would leak.

    Returns a scalar float (mm^2).
    """
    preds, targets = deterministic_predict(model, val_loader, device, use_bg, use_tweedie)
    resid = targets - preds
    return float((resid ** 2).mean().item())


def summarize_uncertainty(preds_TxN, targets_N, station_ids_N, alpha=0.90,
                          sigma2_obs=None):
    """
    Compute per-node mu/sigma/interval + coverage.

    Always reports the epistemic-only interval (empirical quantiles of the T
    dropout samples). When `sigma2_obs` (tau^-1) is supplied, ALSO reports the
    Gal & Ghahramani predictive interval mu +/- z*sqrt(sigma^2_dropout + tau^-1),
    lower bound clamped at 0 (rainfall is non-negative).
    """
    mu = preds_TxN.mean(dim=0)
    sigma = preds_TxN.std(dim=0, unbiased=True)

    lo_q = (1 - alpha) / 2
    hi_q = 1 - lo_q
    q_lo = torch.quantile(preds_TxN, lo_q, dim=0)
    q_hi = torch.quantile(preds_TxN, hi_q, dim=0)

    within = (targets_N >= q_lo) & (targets_N <= q_hi)
    coverage = within.float().mean().item()

    per_node_df = pd.DataFrame({
        "station_id": station_ids_N.numpy(),
        "target": targets_N.numpy(),
        "pred_mean": mu.numpy(),
        "pred_std": sigma.numpy(),
        f"q{lo_q:.2f}": q_lo.numpy(),
        f"q{hi_q:.2f}": q_hi.numpy(),
        "within_interval": within.numpy(),
    })

    summary = {
        "alpha": alpha,
        "T": preds_TxN.shape[0],
        "n_nodes": int(preds_TxN.shape[1]),
        f"coverage_{int(alpha*100)}": coverage,
        "mean_sigma": sigma.mean().item(),
        "mean_interval_width": (q_hi - q_lo).mean().item(),
    }

    if sigma2_obs is not None:
        # z for the central `alpha` interval, e.g. 1.6449 for alpha=0.90.
        z = float(NormalDist().inv_cdf(1.0 - (1.0 - alpha) / 2.0))
        sigma_pred = torch.sqrt(sigma ** 2 + float(sigma2_obs))
        p_lo = (mu - z * sigma_pred).clamp(min=0.0)
        p_hi = mu + z * sigma_pred

        within_pred = (targets_N >= p_lo) & (targets_N <= p_hi)

        per_node_df["pred_sigma_total"] = sigma_pred.numpy()
        per_node_df["pred_lo"] = p_lo.numpy()
        per_node_df["pred_hi"] = p_hi.numpy()
        per_node_df["within_interval_predictive"] = within_pred.numpy()

        summary.update({
            "tau_inv_obs_noise_var": float(sigma2_obs),
            "sigma_obs": float(np.sqrt(sigma2_obs)),
            f"coverage_{int(alpha*100)}_predictive": within_pred.float().mean().item(),
            "mean_sigma_total": sigma_pred.mean().item(),
            "mean_interval_width_predictive": (p_hi - p_lo).mean().item(),
        })

    return per_node_df, summary


def build_fold_assets(config, fold, radar_graph, satellite_graph, raingauge_df,
                       raingauge_station_mappings_df, split_info, use_radar,
                       use_satellite, log_transform, batch_size,
                       return_val_loader=False):
    gauge_graph = GaugeGraphNew(
        raingauge_df, raingauge_station_mappings_df,
        split_info=split_info[fold], knn=config['layer_connect']['gauge_gauge'],
        config=config,
    )
    if use_radar:
        gauge_graph.add_heterodata(
            heterodata_layer=radar_graph.get_radar_heterodata(),
            coords=radar_graph.grid_coords, layer_name='radar',
            knn=config['layer_connect']['radar_gauge'],
        )
    if use_satellite:
        gauge_graph.add_heterodata(
            heterodata_layer=satellite_graph.get_satellite_heterodata(),
            coords=satellite_graph.grid_coords, layer_name='satellite',
            knn=config['layer_connect']['satellite_gauge'],
        )

    train_data = gauge_graph.get_train_heterodata()
    test_data = gauge_graph.get_test_heterodata()
    norm_stats = compute_norm_stats(train_data, log_transform)

    test_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(apply_norm(test_data, norm_stats, log_transform)),
        batch_size=batch_size, shuffle=False,
    )
    edge_types = train_data.edge_types

    if not return_val_loader:
        return test_loader, edge_types

    # Validation split, normalised with the SAME train-derived stats — used
    # only to estimate the observation-noise term tau^-1.
    val_data = gauge_graph.get_validation_heterodata()
    val_loader = GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(apply_norm(val_data, norm_stats, log_transform)),
        batch_size=batch_size, shuffle=False,
    )
    return test_loader, edge_types, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config.yaml used to train the checkpoint')
    parser.add_argument('--experiment-dir', required=True, help='experiments/<name> dir containing weather_gnn_best_{fold}.pth')
    parser.add_argument('--fold', default='0', help='Fold index, or "all" to run every fold found in --experiment-dir')
    parser.add_argument('--T', type=int, default=50, help='Number of stochastic MC-dropout forward passes')
    parser.add_argument('--alpha', type=float, default=0.90, help='Central interval coverage level')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-predictive-interval', action='store_true',
                        help='Report only the original epistemic-only coverage; skip the '
                             'Gal & Ghahramani tau^-1 predictive interval (and the extra '
                             'validation-split pass it needs).')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = read_config(args.config)
    batch_size = config['training_params']['batch_size']
    fold_count = config['training_params']['fold_count']
    use_tweedie = config['training_params'].get('use_tweedie', False)
    use_bg = config['training_params'].get('use_bernoulli_gamma', False)
    log_transform = config['dataset_parameters'].get('log_transform', False)

    uptime_threshold = config['filters']['uptime_threshold']
    start_year = config['dataset_parameters']['start_year']
    end_year = config['dataset_parameters']['end_year']
    raingauge_df, raingauge_station_mappings_df = load_raingauge_dataset(
        rainfall_file=config['dataset_parameters'].get('rainfall_file', 'all_stations_rainfall_hourly_combined.csv'),
        metadata_file=config['dataset_parameters'].get('metadata_file', 'database/australia/station_metadata.csv'),
        start=start_year, end=end_year, uptime_threshold=uptime_threshold,
    )
    split_info = stratified_spatial_kfold_dual(
        raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
    )

    datasources = config.get('datasources', ['raingauge'])
    use_radar = 'radar' in datasources
    use_satellite = 'satellite' in datasources

    radar_graph = None
    if use_radar:
        from src.graph.radargraph_au import AURadarGraph
        import pickle
        radar_cfg = config['dataset_parameters']
        radar_base = radar_cfg.get('radar_base_path', '/g/data/rq0/rainfields3')
        radar_cache = radar_cfg.get('radar_cache_path', None)
        radar_id = radar_cfg.get('radar_id', 71)
        radar_stride = radar_cfg.get('radar_stride', 10)
        radar_site = radar_cfg.get('radar_site', 'sydney')
        if radar_site == 'wagga':
            from src.radar.wagga_radar_preprocessor import AURadarPreprocessor
        else:
            from src.radar.au_preprocessor import AURadarPreprocessor

        if radar_cache and os.path.exists(radar_cache):
            with open(radar_cache, 'rb') as f:
                radar_df_raw = pickle.load(f)
            preprocessor = AURadarPreprocessor(radar_base, radar_id, radar_stride)
        else:
            preprocessor = AURadarPreprocessor(radar_base, radar_id, radar_stride)
            radar_df_raw = preprocessor.build_dataset_range(start_year, end_year)

        radar_graph = AURadarGraph(
            radar_df=radar_df_raw, gauge_index=raingauge_df.index,
            node_coords=preprocessor.get_node_coords(), grid_shape=preprocessor.grid_shape,
        )

    satellite_graph = None
    if use_satellite:
        from src.graph.satellitegraph_au import AUSatelliteGraph
        import pickle
        sat_cfg = config['dataset_parameters']
        sat_base = sat_cfg.get('satellite_base_path', '/g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph')
        sat_cache = sat_cfg.get('satellite_cache_path', None)
        sat_site = sat_cfg.get('satellite_site', 'sydney')
        if sat_site == 'wagga':
            from src.satellite.wagga_satellite_preprocessor import AUSatellitePreprocessor
        else:
            from src.satellite.au_preprocessor import AUSatellitePreprocessor

        if sat_cache and os.path.exists(sat_cache):
            with open(sat_cache, 'rb') as f:
                satellite_df_raw = pickle.load(f)
            preprocessor_sat = AUSatellitePreprocessor(sat_base)
        else:
            preprocessor_sat = AUSatellitePreprocessor(sat_base)
            satellite_df_raw = preprocessor_sat.build_dataset_range(start_year, end_year)

        satellite_graph = AUSatelliteGraph(
            satellite_df=satellite_df_raw, gauge_index=raingauge_df.index,
            node_coords=preprocessor_sat.get_node_coords(), grid_shape=preprocessor_sat.grid_shape,
        )

    hidden_channels = config['model']['hidden_channels']
    num_layers = config['model']['num_layers']
    out_channels = 3 if use_bg else 1
    in_channels_dict = {"raingauge": 1}
    if use_radar:
        in_channels_dict["radar"] = 2
    if use_satellite:
        in_channels_dict["satellite"] = 2

    if args.fold == 'all':
        folds = [
            int(fn.split('_')[-1].split('.')[0])
            for fn in os.listdir(args.experiment_dir)
            if fn.startswith('weather_gnn_best_') and fn.endswith('.pth')
        ]
        folds.sort()
    else:
        folds = [int(args.fold)]

    all_summaries = []
    for fold in folds:
        ckpt_path = os.path.join(args.experiment_dir, f"weather_gnn_best_{fold}.pth")
        print(f"\n=== Fold {fold}: loading {ckpt_path} ===")

        test_loader, edge_types, val_loader = build_fold_assets(
            config, fold, radar_graph, satellite_graph, raingauge_df,
            raingauge_station_mappings_df, split_info, use_radar, use_satellite,
            log_transform, batch_size, return_val_loader=True,
        )

        model = GNNInductiveHetero(
            in_channels_dict=in_channels_dict, hidden_channels=hidden_channels,
            out_channels=out_channels, num_layers=num_layers, edge_types=edge_types,
            dropout=config['model']['dropout'], hetero_aggr=config['model'].get('hetero_aggr', 'mean'),
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # tau^-1: observation-noise variance from the validation split (no
        # test leakage). Skipped with --no-predictive-interval.
        sigma2_obs = None
        if not args.no_predictive_interval:
            sigma2_obs = estimate_obs_noise_var(model, val_loader, device, use_bg, use_tweedie)
            print(f"  tau^-1 (val residual var) = {sigma2_obs:.5f} mm^2  "
                  f"(sigma_obs = {np.sqrt(sigma2_obs):.4f} mm)")

        preds_TxN, targets_N, station_ids_N = mc_dropout_predict(
            model, test_loader, device, T=args.T, use_bg=use_bg, use_tweedie=use_tweedie
        )
        per_node_df, summary = summarize_uncertainty(
            preds_TxN, targets_N, station_ids_N, alpha=args.alpha, sigma2_obs=sigma2_obs
        )
        summary["fold"] = fold

        csv_path = os.path.join(args.experiment_dir, f"mc_dropout_f{fold}.csv")
        json_path = os.path.join(args.experiment_dir, f"mc_dropout_summary_f{fold}.json")
        per_node_df.to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        pct = int(args.alpha * 100)
        print(f"Coverage_{pct} (epistemic only): {summary[f'coverage_{pct}']:.4f}  "
              f"(mean sigma={summary['mean_sigma']:.4f}, mean interval width={summary['mean_interval_width']:.4f}, "
              f"n_nodes={summary['n_nodes']})")
        if f'coverage_{pct}_predictive' in summary:
            print(f"Coverage_{pct} (predictive, +tau^-1): {summary[f'coverage_{pct}_predictive']:.4f}  "
                  f"(mean sigma_total={summary['mean_sigma_total']:.4f}, "
                  f"mean interval width={summary['mean_interval_width_predictive']:.4f})")
        print(f"Saved → {csv_path}")
        print(f"Saved → {json_path}")
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        agg_path = os.path.join(args.experiment_dir, "mc_dropout_summary_all_folds.json")
        with open(agg_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        pct = int(args.alpha * 100)
        cov = [s[f'coverage_{pct}'] for s in all_summaries]
        print(f"\nMean coverage_{pct} (epistemic only) across {len(all_summaries)} folds: "
              f"{np.mean(cov):.4f}  (std {np.std(cov):.4f})")
        if f'coverage_{pct}_predictive' in all_summaries[0]:
            covp = [s[f'coverage_{pct}_predictive'] for s in all_summaries]
            print(f"Mean coverage_{pct} (predictive, +tau^-1) across {len(all_summaries)} folds: "
                  f"{np.mean(covp):.4f}  (std {np.std(covp):.4f})")
        print(f"Saved → {agg_path}")


if __name__ == '__main__':
    main()
