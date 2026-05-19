"""
error_analysis.py
=================
Spatial and statistical analysis of per-station prediction errors produced
by test_model() in training/logic_hetero.py.

The CSV files saved by test_model have columns:
    station_id, mae, rmse, bias, precision, recall, f1, support_pos, support_neg

station_id is the 0-based position of the station in mapping_df (i.e.
mapping_df.iloc[station_id] gives the corresponding row).

Exported symbols
----------------
load_fold_metrics       – load and aggregate CSV(s) from one or more folds
plot_spatial_error_map              – geographical scatter plot coloured by chosen metric
plot_spatial_error_map_with_cml     – same + CML link overlay + proximity highlighting
plot_error_ranking      – horizontal bar chart of worst / best stations
plot_bias_map           – spatial map of signed prediction bias
plot_error_vs_isolation – scatter: error vs. distance to nearest training gauge
analyze_station_errors  – run all four plots in one call and return the DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# Must match logic_hetero._DATA_FEATURE_DIM
_DATA_FEATURE_DIM = 2


# ---------------------------------------------------------------------------
# 0.  Config-driven scale loader
# ---------------------------------------------------------------------------

def get_viz_scales(config_path: str = 'config.yaml') -> dict:
    """
    Load fixed visualisation scales from config.yaml and return two ready-to-use dicts.

    Returns
    -------
    dict with keys:
        'compare_metrics' : list[str]   – e.g. ['rmse', 'pearson_r']
        'metric_scales'   : dict        – {metric: {vmin, vmax, boundaries?}}
                            pass as scale_params to analyze_station_errors()
                            or unpack with ** into plot_spatial_error_map()
        'rainfall'        : dict        – {vmin, vmax, boundaries?}
                            unpack with ** into plot_rainfall_grid() etc.

    Example
    -------
    scales = get_viz_scales()
    plot_spatial_error_map(station_df, bounds, metric='rmse',
                           **scales['metric_scales']['rmse'])
    plot_rainfall_grid(predictions, grid_shape, bounds,
                       **scales['rainfall'])
    analyze_station_errors(csv_paths, mapping_df, bounds,
                           metrics=scales['compare_metrics'],
                           scale_params=scales['metric_scales'])
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    vis = cfg.get('visualisation', {})
    return {
        'compare_metrics': vis.get('compare_metrics', ['rmse', 'pearson_r']),
        'metric_scales':   vis.get('metrics', {}),
        'rainfall':        vis.get('rainfall', {'vmin': 0, 'vmax': 20}),
    }


# ---------------------------------------------------------------------------
# 1.  Data loading
# ---------------------------------------------------------------------------

def add_pearson_r(
    station_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-station Pearson R from predictions/actuals DataFrames and
    merge it into station_df as a new 'pearson_r' column.

    Parameters
    ----------
    station_df     : DataFrame  output of load_fold_metrics() (must have 'id' column)
    predictions_df : DataFrame  timestamps × station_ids (predicted values)
    actuals_df     : DataFrame  timestamps × station_ids (actual values)

    Returns
    -------
    station_df with an added 'pearson_r' column (NaN where insufficient data)
    """
    from scipy.stats import pearsonr

    r_map = {}
    for sid in predictions_df.columns:
        pred   = predictions_df[sid].values.astype(float)
        actual = actuals_df[sid].values.astype(float)
        mask   = ~(np.isnan(pred) | np.isnan(actual))
        if mask.sum() > 2:
            r, _ = pearsonr(actual[mask], pred[mask])
            r_map[str(sid)] = float(r)
        else:
            r_map[str(sid)] = np.nan

    out = station_df.copy()
    out['pearson_r'] = out['id'].astype(str).map(r_map)
    return out


def load_fold_metrics(
    csv_paths: list[str],
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load per-station metric CSVs from one or more folds and join with
    geographic coordinates from *mapping_df*.

    Metrics are averaged across folds for stations that appear in multiple
    folds' test sets.

    Parameters
    ----------
    csv_paths   : list[str]  paths to per_station_metrics_f*.csv files
    mapping_df  : DataFrame  must have columns ['id', 'longitude', 'latitude']
                             ordered so that mapping_df.iloc[i] is station i

    Returns
    -------
    DataFrame with columns:
        station_id, id, longitude, latitude,
        mae, rmse, bias, precision, recall, f1,
        support_pos, support_neg, n_folds
    """
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df['source_fold'] = Path(path).stem   # e.g. "per_station_metrics_f0"
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    numeric_cols = ['mae', 'rmse', 'bias', 'pearson_r', 'precision', 'recall', 'f1',
                    'support_pos', 'support_neg']
    # Keep only columns that actually exist in the CSV
    numeric_cols = [c for c in numeric_cols if c in raw.columns]

    agg_dict = {c: 'mean' for c in numeric_cols}
    agg_dict['source_fold'] = 'count'

    aggregated = (
        raw.groupby('station_id')
           .agg(agg_dict)
           .rename(columns={'source_fold': 'n_folds'})
           .reset_index()
    )

    # Map local integer IDs to station info from mapping_df
    mapping_reset = mapping_df.reset_index(drop=True)
    coord_df = mapping_reset[['id', 'longitude', 'latitude']].copy()
    coord_df['station_id'] = coord_df.index

    merged = aggregated.merge(coord_df, on='station_id', how='left')
    return merged


# ---------------------------------------------------------------------------
# 2.  Spatial error map
# ---------------------------------------------------------------------------

def plot_spatial_error_map(
    station_df: pd.DataFrame,
    bounds: dict,
    metric: str = 'rmse',
    top_n_labels: int = 5,
    cmap: str = 'RdYlGn_r',
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    show_outline: bool = True,
    ax=None,
    save_path: str | None = None,
):
    """
    Scatter plot of station locations coloured by a chosen error metric.
    The worst *top_n_labels* stations are annotated with their station ID.

    Parameters
    ----------
    station_df    : DataFrame  output of load_fold_metrics() / add_pearson_r()
    bounds        : dict  {'left', 'right', 'top', 'bottom'}
    metric        : str   column name to colour by
                    ('mae', 'rmse', 'f1', 'bias', 'pearson_r')
    top_n_labels  : int   number of worst stations to annotate
    cmap          : str   matplotlib colourmap (reversed Green→Red for errors)
    title         : str   optional plot title
    vmin          : float optional fixed colour-scale minimum (overrides auto)
    vmax          : float optional fixed colour-scale maximum (overrides auto)
    boundaries    : list[float] optional explicit colour-transition breakpoints;
                    when supplied, a BoundaryNorm is used so each interval maps
                    to a distinct colour band.  Example: [0, 0.5, 1, 2, 4]
    show_outline  : bool  overlay Singapore coastline (default True)
    ax            : Axes  optional existing axes
    save_path     : str   optional file path to save the figure

    Returns
    -------
    ax : matplotlib Axes
    """
    if metric not in station_df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(station_df.columns)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    valid = station_df.dropna(subset=[metric, 'longitude', 'latitude'])
    values = valid[metric].values

    # Choose colourmap direction: higher-is-better metrics use green at the top
    _cmap = cmap
    if metric in ('f1', 'pearson_r'):
        _cmap = 'RdYlGn'
    elif metric == 'bias':
        _cmap = 'RdBu_r'

    # Build norm — no silent auto-scaling: vmin/vmax must be set explicitly or
    # loaded from config via get_viz_scales().
    if vmin is None or (vmax is None and boundaries is None):
        raise ValueError(
            f"Fixed scale required for metric='{metric}'. "
            "Pass vmin/vmax (and optionally boundaries), or load from config:\n"
            "  scales = get_viz_scales()\n"
            f"  plot_spatial_error_map(..., **scales['metric_scales']['{metric}'])"
        )

    if boundaries is not None:
        cmap_obj = cm.get_cmap(_cmap)
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_obj.N, clip=True)
    elif metric == 'bias':
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        valid['longitude'], valid['latitude'],
        c=values, cmap=_cmap, norm=norm,
        s=120, zorder=4, edgecolors='black', linewidths=0.5,
    )
    cbar_label = 'Pearson R' if metric == 'pearson_r' else metric.upper()
    plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)

    # Annotate the worst stations
    # "worst" = largest value for error metrics, smallest for skill metrics
    if metric in ('f1', 'pearson_r'):
        worst = valid.nsmallest(top_n_labels, metric)
    elif metric == 'bias':
        worst = valid.reindex(valid['bias'].abs().nlargest(top_n_labels).index)
    else:
        worst = valid.nlargest(top_n_labels, metric)

    for _, row in worst.iterrows():
        ax.annotate(
            str(row['id']),
            xy=(row['longitude'], row['latitude']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        )

    ax.set_xlim(bounds['left'],  bounds['right'])
    ax.set_ylim(bounds['bottom'], bounds['top'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title or f'Per-station {metric.upper()} — spatial distribution')
    ax.grid(True, alpha=0.3)

    if show_outline:
        from src.visualization.main import visualise_singapore_outline
        visualise_singapore_outline(ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 3.  Spatial error map with CML overlay
# ---------------------------------------------------------------------------

def plot_spatial_error_map_with_cml(
    station_df: pd.DataFrame,
    cml_coordinates_df: pd.DataFrame,
    bounds: dict,
    metric: str = 'rmse',
    cml_influence_km: float = 5.0,
    top_n_labels: int = 5,
    cmap: str = 'RdYlGn_r',
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    show_outline: bool = True,
    ax=None,
    save_path: str | None = None,
):
    """
    Spatial error map overlaid with the CML link network, highlighting which
    rain gauges are within *cml_influence_km* of a CML link (i.e. gauges most
    likely to benefit from CML as an additional data source).

    Parameters
    ----------
    station_df         : DataFrame  output of load_fold_metrics() / add_pearson_r()
                         must contain 'longitude', 'latitude', 'id', and *metric*
    cml_coordinates_df : DataFrame  with columns 'site_a_longitude',
                         'site_a_latitude', 'site_b_longitude', 'site_b_latitude'
    bounds             : dict  {'left', 'right', 'top', 'bottom'}
    metric             : str   error column to colour by
    cml_influence_km   : float distance threshold (km) — gauges closer than this
                         to any CML link are outlined in teal and marked
                         "within CML influence"
    top_n_labels       : int   worst-performing stations to annotate
    cmap, title, vmin, vmax, boundaries, show_outline, ax, save_path : same as
                         plot_spatial_error_map

    Returns
    -------
    ax : matplotlib Axes
    """
    from shapely.geometry import LineString, Point
    from shapely import STRtree

    # ── Base error map ────────────────────────────────────────────────────────
    ax = plot_spatial_error_map(
        station_df=station_df,
        bounds=bounds,
        metric=metric,
        top_n_labels=top_n_labels,
        cmap=cmap,
        title=title or f'Per-station {metric.upper()} with CML overlay',
        vmin=vmin,
        vmax=vmax,
        boundaries=boundaries,
        show_outline=show_outline,
        ax=ax,
    )

    # ── CML links as lines ───────────────────────────────────────────────────
    # cml_coordinates_df rows come in pairs (site_a, site_b); use every other row.
    cml_links = cml_coordinates_df.iloc[::2].reset_index(drop=True)

    cml_shapely = []
    for _, row in cml_links.iterrows():
        ax.plot(
            [row['site_a_longitude'], row['site_b_longitude']],
            [row['site_a_latitude'],  row['site_b_latitude']],
            color='purple', linewidth=0.8, alpha=0.5, zorder=3,
        )
        cml_shapely.append(
            LineString([
                (row['site_a_longitude'], row['site_a_latitude']),
                (row['site_b_longitude'], row['site_b_latitude']),
            ])
        )

    # Dummy handle for CML lines in the legend
    ax.plot([], [], color='purple', linewidth=0.8, alpha=0.7, label='CML links')

    # ── Per-gauge distance to nearest CML link ────────────────────────────────
    # Convert influence radius from km to approximate degrees (1 deg ≈ 111 km)
    influence_deg = cml_influence_km / 111.0

    tree = STRtree(cml_shapely)
    valid = station_df.dropna(subset=[metric, 'longitude', 'latitude'])

    near_lon, near_lat = [], []
    for _, row in valid.iterrows():
        pt = Point(row['longitude'], row['latitude'])
        nearest_idx = tree.nearest(pt)
        dist = pt.distance(cml_shapely[nearest_idx])
        if dist <= influence_deg:
            near_lon.append(row['longitude'])
            near_lat.append(row['latitude'])

    if near_lon:
        ax.scatter(
            near_lon, near_lat,
            facecolors='none', edgecolors='teal', linewidths=2.0,
            s=220, zorder=6, label=f'Within {cml_influence_km} km of CML',
        )

    ax.legend(fontsize=8, loc='lower right')

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 4.  Bar chart ranking
# ---------------------------------------------------------------------------

def plot_error_ranking(
    station_df: pd.DataFrame,
    metric: str = 'mae',
    top_n: int = 15,
    title: str | None = None,
    ax=None,
    save_path: str | None = None,
):
    """
    Horizontal bar chart of the *top_n* worst-performing stations.

    Parameters
    ----------
    station_df : DataFrame  output of load_fold_metrics()
    metric     : str   column to rank by ('mae', 'rmse', 'f1')
    top_n      : int   number of stations to show
    title      : str   optional title
    ax         : Axes  optional existing axes
    save_path  : str   optional save path

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))

    valid = station_df.dropna(subset=[metric]).copy()

    if metric == 'f1':
        ranked = valid.nsmallest(top_n, metric)
        colour = 'salmon'
        xlabel = f'{metric.upper()} (lower = worse)'
    else:
        ranked = valid.nlargest(top_n, metric)
        colour = 'steelblue'
        xlabel = f'{metric.upper()} (higher = worse)'

    ranked = ranked.sort_values(metric, ascending=(metric == 'f1'))
    labels = ranked['id'].astype(str).values
    vals   = ranked[metric].values

    bars = ax.barh(labels, vals, color=colour, edgecolor='black', linewidth=0.4)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + 0.01 * vals.max(),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', fontsize=8,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Station ID')
    ax.set_title(title or f'Top {top_n} stations by {metric.upper()}')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 4.  Bias map (over/under-prediction geography)
# ---------------------------------------------------------------------------

def plot_bias_map(
    station_df: pd.DataFrame,
    bounds: dict,
    top_n_labels: int = 5,
    title: str | None = None,
    ax=None,
    save_path: str | None = None,
    vmax: float | None = None,
):
    """
    Spatial map of signed prediction bias (mean(pred − target)).
    Red = model over-predicts; blue = model under-predicts.
    Circle size scales with absolute bias magnitude.

    Parameters
    ----------
    station_df    : DataFrame  must include 'bias' column (from load_fold_metrics)
    bounds        : dict
    top_n_labels  : int  annotate the N stations with largest absolute bias
    vmax          : float  optional fixed symmetric colour-scale limit (mm).
                    When set, the colourbar spans [-vmax, +vmax] regardless of
                    the actual data range, making plots comparable across runs.
                    When None (default) the scale is derived from the data.
    title, ax, save_path : standard

    Returns
    -------
    ax : matplotlib Axes
    """
    if 'bias' not in station_df.columns:
        raise ValueError("'bias' column not found. Re-run test_model with the updated logic_hetero.py.")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    valid = station_df.dropna(subset=['bias', 'longitude', 'latitude'])
    biases = valid['bias'].values
    abs_max = vmax if vmax is not None else max(np.abs(biases).max(), 1e-6)

    # Size proportional to absolute bias (min 30 for visibility)
    sizes = 30 + 200 * np.abs(biases) / abs_max

    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    sc = ax.scatter(
        valid['longitude'], valid['latitude'],
        c=biases, cmap='RdBu_r', norm=norm,
        s=sizes, zorder=4, edgecolors='black', linewidths=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax, label='Bias  (pred − actual, mm)', shrink=0.8)
    cbar.ax.axhline(0, color='black', linewidth=1)   # zero-bias reference line

    # Annotate largest biases
    worst = valid.reindex(valid['bias'].abs().nlargest(top_n_labels).index)
    for _, row in worst.iterrows():
        ax.annotate(
            f"{row['id']} ({row['bias']:+.2f})",
            xy=(row['longitude'], row['latitude']),
            xytext=(6, 6), textcoords='offset points',
            fontsize=7, color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        )

    ax.set_xlim(bounds['left'],  bounds['right'])
    ax.set_ylim(bounds['bottom'], bounds['top'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title or 'Prediction bias by station  (red = over-predict, blue = under-predict)')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 5.  Error vs. spatial isolation
# ---------------------------------------------------------------------------

def plot_error_vs_isolation(
    station_df: pd.DataFrame,
    train_station_ids: list,
    metric: str = 'mae',
    title: str | None = None,
    ax=None,
    save_path: str | None = None,
):
    """
    Scatter plot of per-station error vs. distance to the nearest TRAINING
    gauge.  For spatial interpolation, stations that are far from any training
    gauge should be harder to predict.

    Parameters
    ----------
    station_df        : DataFrame  output of load_fold_metrics()  (test stations)
    train_station_ids : list[str]  station IDs used for training in this fold
                        (from split_info['ml']['train'])
    metric            : str   error column to plot on y-axis
    title, ax, save_path : standard

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    # Separate train and test station coordinates
    test_df  = station_df.dropna(subset=[metric, 'longitude', 'latitude'])
    train_df = station_df[station_df['id'].isin(train_station_ids)].dropna(
        subset=['longitude', 'latitude']
    )

    if train_df.empty:
        ax.text(0.5, 0.5, 'No training station coordinates available.',
                transform=ax.transAxes, ha='center')
        return ax

    # Compute distance from each test station to nearest training station
    train_latlon = np.radians(train_df[['latitude', 'longitude']].values)
    test_latlon  = np.radians(test_df[['latitude', 'longitude']].values)

    nbrs = NearestNeighbors(n_neighbors=1, metric='haversine')
    nbrs.fit(train_latlon)
    dists_rad, _ = nbrs.kneighbors(test_latlon)
    dists_km = dists_rad.flatten() * 6371.0

    errors = test_df[metric].values

    ax.scatter(dists_km, errors, alpha=0.7, edgecolors='black', linewidths=0.4, s=60)

    # Trend line
    if len(dists_km) > 2:
        z = np.polyfit(dists_km, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(dists_km.min(), dists_km.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, label=f'Trend (slope={z[0]:.3f})')
        ax.legend(fontsize=8)

    # Pearson r annotation
    if len(dists_km) > 2:
        from scipy.stats import pearsonr
        r, pval = pearsonr(dists_km, errors)
        ax.text(
            0.97, 0.95, f'r = {r:.3f}  (p={pval:.3f})',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7),
        )

    ax.set_xlabel('Distance to nearest training gauge (km)')
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f'{metric.upper()} vs. distance to nearest training gauge')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 6.  Combined entry point
# ---------------------------------------------------------------------------

def analyze_station_errors(
    csv_paths: list[str],
    mapping_df: pd.DataFrame,
    bounds: dict,
    train_station_ids: list | None = None,
    output_dir: str = '.',
    top_n: int = 10,
    metrics: tuple[str, str] = ('rmse', 'pearson_r'),
    scale_params: dict | None = None,
) -> pd.DataFrame:
    """
    Run all four diagnostic plots and return the aggregated station DataFrame.

    Produces four figures:
        error_analysis_map1.png        – spatial map for metrics[0]
        error_analysis_map2.png        – spatial map for metrics[1]
        error_analysis_bias_map.png    – spatial bias map
        error_analysis_ranking.png     – worst-station bar chart
        error_analysis_isolation.png   – MAE vs. distance to training gauge
                                         (only if train_station_ids is given)

    Parameters
    ----------
    csv_paths         : list[str]  per_station_metrics_f*.csv paths
    mapping_df        : DataFrame  with ['id', 'longitude', 'latitude', 'order']
    bounds            : dict  {'left', 'right', 'top', 'bottom'}
    train_station_ids : list  optional – station IDs in the training set of one fold
    output_dir        : str   directory for saved figures
    top_n             : int   stations to highlight in ranking / annotation
    metrics           : tuple[str, str]  two metrics to compare side-by-side
                        choose from: 'rmse', 'mae', 'f1', 'pearson_r'
                        default: ('rmse', 'pearson_r')
    scale_params      : dict | None  optional per-metric scale overrides, e.g.:
                        {'rmse': {'vmin': 0, 'vmax': 3, 'boundaries': [0,0.5,1,2,3]},
                         'pearson_r': {'vmin': -1, 'vmax': 1}}

    Returns
    -------
    station_df : pd.DataFrame  aggregated metrics with coordinates
    """
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scale_params = scale_params or {}

    station_df = load_fold_metrics(csv_paths, mapping_df)

    print(f"\nLoaded metrics for {len(station_df)} unique test stations "
          f"across {len(csv_paths)} fold(s).\n")
    show_cols = ['id'] + [c for c in ('mae', 'rmse', 'bias', 'f1') if c in station_df.columns]
    print(station_df[show_cols].sort_values('mae' if 'mae' in station_df.columns else show_cols[1],
                                            ascending=False)
          .head(top_n).to_string(index=False))

    metrics_list = list(metrics)
    metric1 = metrics_list[0] if len(metrics_list) > 0 else 'mae'
    metric2 = metrics_list[1] if len(metrics_list) > 1 else ('f1' if 'f1' in station_df.columns else 'mae')

    # --- Plot 1: First metric spatial map ---
    if metric1 in station_df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        plot_spatial_error_map(
            station_df, bounds, metric=metric1, top_n_labels=top_n, ax=ax1,
            **scale_params.get(metric1, {}),
        )
        fig1.tight_layout()
        fig1.savefig(out / f'error_analysis_{metric1}_map.png', dpi=200, bbox_inches='tight')
        plt.close(fig1)

    # --- Plot 2: Second metric spatial map ---
    if metric2 in station_df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        plot_spatial_error_map(
            station_df, bounds, metric=metric2, top_n_labels=top_n, ax=ax2,
            **scale_params.get(metric2, {}),
        )
        fig2.tight_layout()
        fig2.savefig(out / f'error_analysis_{metric2}_map.png', dpi=200, bbox_inches='tight')
        plt.close(fig2)

    # --- Plot 3: Bias map (if bias column present) ---
    if 'bias' in station_df.columns:
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        plot_bias_map(station_df, bounds, top_n_labels=top_n, ax=ax3)
        fig3.tight_layout()
        fig3.savefig(out / 'error_analysis_bias_map.png', dpi=200, bbox_inches='tight')
        plt.close(fig3)

    # --- Plot 4: Ranking bar chart (metric1 and metric2 side by side) ---
    fig4, (axA, axB) = plt.subplots(1, 2, figsize=(14, max(4, top_n * 0.45)))
    rank_m1 = metric1 if metric1 in station_df.columns else 'mae'
    rank_m2 = metric2 if metric2 in station_df.columns else 'f1'
    plot_error_ranking(station_df, metric=rank_m1, top_n=top_n, ax=axA)
    plot_error_ranking(station_df, metric=rank_m2, top_n=top_n, ax=axB)
    fig4.suptitle('Worst-performing stations', fontsize=13)
    fig4.tight_layout()
    fig4.savefig(out / 'error_analysis_ranking.png', dpi=200, bbox_inches='tight')
    plt.close(fig4)

    # --- Plot 5: Error vs. spatial isolation ---
    if train_station_ids is not None:
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        iso_metric = metric1 if metric1 in ('mae', 'rmse') else 'mae'
        plot_error_vs_isolation(
            station_df, train_station_ids, metric=iso_metric, ax=ax5,
        )
        fig5.tight_layout()
        fig5.savefig(out / 'error_analysis_isolation.png', dpi=200, bbox_inches='tight')
        plt.close(fig5)

    print(f"\nFigures saved to: {out.resolve()}")
    return station_df


# ---------------------------------------------------------------------------
# 7.  Per-station time-series inference (GNN spatial model)
# ---------------------------------------------------------------------------

def predict_on_test_stations(
    model,
    heterodata,
    mapping_df: pd.DataFrame,
    timestamps: pd.Series,
    device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the spatial GNN over every test timestep and return per-station
    predicted and actual values as DataFrames.

    Parameters
    ----------
    model       : trained GNNInductiveHetero model (eval mode)
    heterodata  : HeteroData from GaugeGraphNew.get_test_heterodata()
                  shape: raingauge.x → [n_nodes, T, F]
    mapping_df  : DataFrame with station metadata (must have 'id' column,
                  ordered so that mapping_df.iloc[i] is node i)
    timestamps  : pd.Series of length T — timestamps aligned with heterodata
    device      : torch device

    Returns
    -------
    predictions_df : DataFrame  shape [T, n_test_stations]
    actuals_df     : DataFrame  shape [T, n_test_stations]
    Both are indexed by *timestamps* with station IDs as columns.
    """
    import torch
    from torch_geometric.data import HeteroData

    model.eval()

    raw_mask = heterodata['raingauge'].mask
    if isinstance(raw_mask, np.ndarray):
        test_mask = torch.tensor(raw_mask, dtype=torch.bool)
    else:
        test_mask = raw_mask.bool()

    T = heterodata['raingauge'].x.shape[1]
    test_node_indices = test_mask.nonzero(as_tuple=False).squeeze(1).tolist()
    station_ids = [mapping_df.iloc[i]['id'] for i in test_node_indices]

    all_preds:   list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []

    with torch.no_grad():
        for t in range(T):
            snap = HeteroData()

            snap['raingauge'].x         = heterodata['raingauge'].x[:, t, :].to(device)
            snap['raingauge'].y         = heterodata['raingauge'].y[:, t, :].to(device)
            snap['raingauge'].mask      = test_mask.to(device)
            snap['raingauge'].num_nodes = heterodata['raingauge'].x.shape[0]

            for node_type in heterodata.node_types:
                if node_type == 'raingauge':
                    continue
                snap[node_type].x         = heterodata[node_type].x[:, t, :].to(device)
                snap[node_type].num_nodes = heterodata[node_type].x.shape[0]

            for edge_type in heterodata.edge_types:
                snap[edge_type].edge_index = heterodata[edge_type].edge_index.to(device)
                if hasattr(heterodata[edge_type], 'edge_attr'):
                    snap[edge_type].edge_attr = heterodata[edge_type].edge_attr.to(device)

            x_input = snap['raingauge'].x.clone()
            x_input[test_mask.to(device), :_DATA_FEATURE_DIM] = 0.0

            x_dict = {nt: snap[nt].x for nt in snap.node_types}
            x_dict['raingauge'] = x_input

            edge_attr_dict = {
                et: snap[et].edge_attr
                for et in snap.edge_types
                if hasattr(snap[et], 'edge_attr')
            }

            out = model(x_dict, snap.edge_index_dict, edge_attr_dict)
            out['raingauge'] = out['raingauge'].clamp(min=0.0)  # match test_model: rainfall >= 0

            preds_t   = out['raingauge'][test_mask.to(device)].cpu().numpy().flatten()
            actuals_t = snap['raingauge'].y[test_mask.to(device)].cpu().numpy().flatten()

            all_preds.append(preds_t)
            all_actuals.append(actuals_t)

    preds_arr   = np.stack(all_preds,   axis=0)  # [T, n_test_stations]
    actuals_arr = np.stack(all_actuals, axis=0)  # [T, n_test_stations]

    predictions_df = pd.DataFrame(preds_arr,   index=timestamps, columns=station_ids)
    actuals_df     = pd.DataFrame(actuals_arr, index=timestamps, columns=station_ids)

    return predictions_df, actuals_df


# ---------------------------------------------------------------------------
# 7b. Pearson R across all folds
# ---------------------------------------------------------------------------

def compute_pearson_r_all_folds(
    gauge_graph_arr: list,
    experiment_dir: str,
    mapping_df: pd.DataFrame,
    fold_count: int,
    model_cls,
    device,
    infer_arch_fn,
) -> pd.DataFrame:
    """
    Run inference on every fold's test set, compute per-station Pearson R,
    and return a DataFrame averaged across folds.

    Parameters
    ----------
    gauge_graph_arr : list  – GaugeGraphNew objects, one per fold (length fold_count)
    experiment_dir  : str   – path to experiment directory containing weight files
    mapping_df      : DataFrame with 'id', 'longitude', 'latitude' columns
    fold_count      : int   – number of folds
    model_cls       : class – GNNInductiveHetero (passed to avoid circular import)
    device          : torch.device
    infer_arch_fn   : callable – infer_arch(weights_path) → (num_layers, hidden_channels,
                                 raingauge_in, edge_types)  (defined in notebook)

    Returns
    -------
    DataFrame with columns ['id', 'pearson_r', 'longitude', 'latitude']
    pearson_r is averaged over folds in which the station appears as a test node.
    """
    import torch
    from scipy.stats import pearsonr

    fold_r: dict[str, list[float]] = {}  # station_id → list of per-fold R values

    for fold_idx in range(fold_count):
        weights_path = f"{experiment_dir}/weather_gnn_best_{fold_idx}.pth"
        num_layers, hidden_channels, _, ckpt_edge_types = infer_arch_fn(weights_path)

        heterodata = gauge_graph_arr[fold_idx].get_test_heterodata()

        # Build a dummy aligned_timestamps Series of length T
        T = heterodata['raingauge'].x.shape[1]
        timestamps = pd.RangeIndex(T)

        model = model_cls(
            in_channels_dict={src: -1 for src in {et[0] for et in ckpt_edge_types}
                              | {'raingauge'}},
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=num_layers,
            edge_types=ckpt_edge_types,
        ).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        preds_df, actuals_df = predict_on_test_stations(
            model=model,
            heterodata=heterodata,
            mapping_df=mapping_df,
            timestamps=timestamps,
            device=device,
        )

        for sid in preds_df.columns:
            pred   = preds_df[sid].values.astype(float)
            actual = actuals_df[sid].values.astype(float)
            mask   = ~(np.isnan(pred) | np.isnan(actual))
            if mask.sum() > 2:
                r, _ = pearsonr(actual[mask], pred[mask])
                fold_r.setdefault(str(sid), []).append(float(r))

        print(f"  Fold {fold_idx}: {len(preds_df.columns)} test stations processed.")

    rows = []
    for sid, r_list in fold_r.items():
        rows.append({'id': sid, 'pearson_r': float(np.mean(r_list))})

    result_df = pd.DataFrame(rows)

    # Merge coordinates from mapping_df
    coord_df = mapping_df[['id', 'longitude', 'latitude']].copy()
    coord_df['id'] = coord_df['id'].astype(str)
    result_df = result_df.merge(coord_df, on='id', how='left')

    return result_df.sort_values('pearson_r').reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7c. Per-station time-series inference (ST model — GNNInductiveHeteroST)
# ---------------------------------------------------------------------------

def predict_on_test_stations_st(
    model,
    heterodata,
    mapping_df: pd.DataFrame,
    timestamps: pd.Series,
    window_size: int,
    device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the ST-GNN over every valid test timestep and return per-station
    predicted and actual values as DataFrames.

    Parameters
    ----------
    model       : trained GNNInductiveHeteroST (eval mode)
    heterodata  : normalised HeteroData (raingauge.x → [n_nodes, T, F])
    mapping_df  : DataFrame with 'id' column
    timestamps  : pd.Series of length T  (full, un-windowed)
    window_size : int  LSTM context window W; first valid prediction at t=W
    device      : torch device

    Returns
    -------
    predictions_df : DataFrame  shape [T - W, n_test_stations]
    actuals_df     : DataFrame  shape [T - W, n_test_stations]
    Both indexed by timestamps[W:] with station IDs as columns.
    """
    import torch

    model.eval()

    raw_mask = heterodata['raingauge'].mask
    if isinstance(raw_mask, np.ndarray):
        test_mask = torch.tensor(raw_mask, dtype=torch.bool)
    else:
        test_mask = raw_mask.bool()

    T = heterodata['raingauge'].x.shape[1]
    test_node_indices = test_mask.nonzero(as_tuple=False).squeeze(1).tolist()
    station_ids = [mapping_df.iloc[i]['id'] for i in test_node_indices]

    edge_index_dict = {
        et: heterodata[et].edge_index.to(device)
        for et in heterodata.edge_types
    }
    edge_attr_dict = {
        et: heterodata[et].edge_attr.to(device)
        for et in heterodata.edge_types
        if hasattr(heterodata[et], 'edge_attr')
    }

    all_preds:   list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []

    with torch.no_grad():
        for t in range(window_size, T):
            ctx_idx = list(range(t - window_size, t))

            x_cur = heterodata['raingauge'].x[:, t, :].clone().to(device)
            x_cur[test_mask.to(device), :_DATA_FEATURE_DIM] = 0.0

            x_dict = {'raingauge': x_cur}
            x_context_dict = {
                'raingauge': heterodata['raingauge'].x[:, ctx_idx, :].to(device)
            }

            for ntype in heterodata.node_types:
                if ntype == 'raingauge':
                    continue
                x_dict[ntype]         = heterodata[ntype].x[:, t, :].to(device)
                x_context_dict[ntype] = heterodata[ntype].x[:, ctx_idx, :].to(device)

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            preds_t   = out['raingauge'][test_mask.to(device)].cpu().numpy().flatten()
            actuals_t = heterodata['raingauge'].y[test_mask, t, :].cpu().numpy().flatten()

            all_preds.append(preds_t)
            all_actuals.append(actuals_t)

    preds_arr   = np.stack(all_preds,   axis=0)
    actuals_arr = np.stack(all_actuals, axis=0)

    valid_ts = timestamps.iloc[window_size:].reset_index(drop=True)
    predictions_df = pd.DataFrame(preds_arr,   index=valid_ts, columns=station_ids)
    actuals_df     = pd.DataFrame(actuals_arr, index=valid_ts, columns=station_ids)

    return predictions_df, actuals_df


# ---------------------------------------------------------------------------
# 8.  Per-station rainfall time-series plot
# ---------------------------------------------------------------------------

def plot_station_rainfall_timeseries(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    station_id: str,
    time_start: str | None = None,
    time_end:   str | None = None,
    title: str | None = None,
    ax=None,
    save_path: str | None = None,
) -> 'plt.Axes':
    """
    Plot actual vs predicted rainfall for a single station over a time window.

    Parameters
    ----------
    predictions_df : DataFrame  timestamps × station_ids (predicted values)
    actuals_df     : DataFrame  timestamps × station_ids (actual values)
    station_id     : str        column name to plot
    time_start     : str | None e.g. "2024-08-01" — restricts x-axis start
    time_end       : str | None e.g. "2024-08-31" — restricts x-axis end
    title          : str        optional plot title
    ax             : Axes       optional existing axes
    save_path      : str        optional file path to save the figure

    Returns
    -------
    ax : matplotlib Axes
    """
    if station_id not in predictions_df.columns:
        raise ValueError(
            f"Station '{station_id}' not found. Available: {list(predictions_df.columns)}"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(16, 4))

    actual    = actuals_df[station_id].copy()
    predicted = predictions_df[station_id].copy()

    if time_start is not None:
        ts = pd.Timestamp(time_start)
        actual    = actual[actual.index >= ts]
        predicted = predicted[predicted.index >= ts]
    if time_end is not None:
        te = pd.Timestamp(time_end)
        actual    = actual[actual.index <= te]
        predicted = predicted[predicted.index <= te]

    ax.plot(actual.index,    actual.values,    label='Actual',    color='steelblue',
            linewidth=0.8, alpha=0.9)
    ax.plot(predicted.index, predicted.values, label='Predicted', color='orangered',
            linewidth=0.8, alpha=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Rainfall (mm/hr)')
    ax.set_title(title or f'Station {station_id} — Actual vs Predicted rainfall')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax
