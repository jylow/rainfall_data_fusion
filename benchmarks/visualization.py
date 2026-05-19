"""
benchmarks/visualization.py
============================
Visualization utilities for benchmark model results, matching the style of
src/visualization/error_analysis.py and src/visualization/grid_prediction.py.

All plots follow the same aesthetic:
    dpi=200, grid alpha=0.3, colorbar shrink=0.8, edgecolors='black'

Exported symbols
----------------
compute_per_station_metrics  – build per-station metrics DataFrame from raw predictions
plot_spatial_error_map       – station locations coloured by MAE / RMSE / bias / F1
plot_bias_map                – spatial map of signed prediction bias
plot_error_ranking           – horizontal bar chart of worst/best stations
plot_error_vs_isolation      – scatter: error vs. distance to nearest training gauge
plot_regression_scatter      – actual vs. predicted scatter for one model
plot_density_scatter         – density-coloured scatter (KDE) to resolve low-rainfall crowding
plot_regression_comparison   – side-by-side regression plots for multiple models
plot_metrics_comparison      – grouped bar chart comparing aggregate metrics
plot_timestep_metrics        – time series line plot of per-timestep RMSE / MAE
plot_timestep_comparison     – overlaid time series for multiple models
interpolate_to_grid          – IDW-interpolate station predictions onto a regular grid
plot_rainfall_grid           – heatmap of a single grid timestep
plot_rainfall_sequence       – row of N timestep heatmap panels
animate_rainfall_grid        – FuncAnimation of the rainfall grid over time
analyze_benchmark            – run all diagnostic plots for a single benchmark result
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from pathlib import Path
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# 1.  Per-station metric computation
# ---------------------------------------------------------------------------

def compute_per_station_metrics(
    per_station_data: dict,
    coordinates: dict,
    rain_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute per-station error metrics from raw per-station predictions.

    Parameters
    ----------
    per_station_data : dict
        {station_id: {'actual': list[float], 'predicted': list[float]}}
        Returned by benchmark functions when return_detailed=True.
    coordinates : dict
        {station_id: (lat, lon)}
    rain_threshold : float
        Threshold (mm/hr) for binary rain/no-rain F1 classification.

    Returns
    -------
    DataFrame with columns:
        station_id, latitude, longitude, mae, rmse, bias, f1, n_samples
    """
    records = []
    for station_id, data in per_station_data.items():
        actual = np.array(data['actual'], dtype=float)
        predicted = np.array(data['predicted'], dtype=float)
        if len(actual) == 0:
            continue
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual, predicted = actual[mask], predicted[mask]
        if len(actual) == 0:
            continue

        mae  = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        bias = float(np.mean(predicted - actual))
        f1   = float(f1_score(
            (actual >= rain_threshold).astype(int),
            (predicted >= rain_threshold).astype(int),
            zero_division=0,
        ))
        r = float(pearsonr(actual, predicted)[0]) if len(actual) > 2 else np.nan

        lat, lon = coordinates.get(station_id, (np.nan, np.nan))
        records.append({
            'station_id': station_id,
            'latitude':   lat,
            'longitude':  lon,
            'mae':        mae,
            'rmse':       rmse,
            'bias':       bias,
            'f1':         f1,
            'pearson_r':  r,
            'n_samples':  int(len(actual)),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2.  Spatial error map
# ---------------------------------------------------------------------------

def plot_spatial_error_map(
    station_df: pd.DataFrame,
    bounds: dict | None = None,
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
    Matches the style of src/visualization/error_analysis.plot_spatial_error_map.

    Parameters
    ----------
    station_df   : DataFrame  output of compute_per_station_metrics()
    bounds       : dict  {'left','right','top','bottom'} – axis limits; auto if None
    metric       : str   column to colour by ('mae', 'rmse', 'bias', 'f1', 'pearson_r')
    top_n_labels : int   number of worst stations to annotate
    cmap         : str   matplotlib colormap
    title        : str   optional plot title
    vmin         : float optional fixed colour-scale minimum
    vmax         : float optional fixed colour-scale maximum
    boundaries   : list[float] optional BoundaryNorm breakpoints for discrete colour bands
    show_outline : bool  overlay Singapore coastline (default True)
    ax           : matplotlib Axes
    save_path    : str   optional save path

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.cm as cm_mod

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    valid  = station_df.dropna(subset=[metric, 'longitude', 'latitude'])
    values = valid[metric].values

    _cmap = cmap
    if metric in ('f1', 'pearson_r'):
        _cmap = 'RdYlGn'
    elif metric == 'bias':
        _cmap = 'RdBu_r'

    if vmin is None or (vmax is None and boundaries is None):
        raise ValueError(
            f"Fixed scale required for metric='{metric}'. "
            "Pass vmin/vmax (and optionally boundaries), or load from config:\n"
            "  from src.visualization.error_analysis import get_viz_scales\n"
            "  scales = get_viz_scales()\n"
            f"  plot_spatial_error_map(..., **scales['metric_scales']['{metric}'])"
        )

    if boundaries is not None:
        cmap_obj = cm_mod.get_cmap(_cmap)
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

    # Annotate worst stations
    if metric in ('f1', 'pearson_r'):
        worst = valid.nsmallest(top_n_labels, metric)
    elif metric == 'bias':
        worst = valid.reindex(valid['bias'].abs().nlargest(top_n_labels).index)
    else:
        worst = valid.nlargest(top_n_labels, metric)

    for _, row in worst.iterrows():
        ax.annotate(
            str(row['station_id']),
            xy=(row['longitude'], row['latitude']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        )

    if bounds:
        ax.set_xlim(bounds['left'],  bounds['right'])
        ax.set_ylim(bounds['bottom'], bounds['top'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title or f'Per-station {metric.upper()} — spatial distribution')
    ax.grid(True, alpha=0.3)

    if show_outline:
        from src.visualization.main import visualise_singapore_outline, visualise_with_basemap
        visualise_singapore_outline(ax=ax)
        visualise_with_basemap(ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 3.  Bias map
# ---------------------------------------------------------------------------

def plot_bias_map(
    station_df: pd.DataFrame,
    bounds: dict | None = None,
    top_n_labels: int = 5,
    title: str | None = None,
    ax=None,
    save_path: str | None = None,
    vmax=1
):
    """
    Spatial map of signed prediction bias (mean(pred − actual)).
    Red = over-predict; blue = under-predict. Circle size scales with |bias|.
    Matches the style of src/visualization/error_analysis.plot_bias_map.

    Parameters
    ----------
    station_df   : DataFrame  must include 'bias' column
    bounds       : dict  optional axis limits
    top_n_labels : int   stations with largest |bias| to annotate
    title, ax, save_path : standard

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    valid   = station_df.dropna(subset=['bias', 'longitude', 'latitude'])
    biases  = valid['bias'].values
    abs_max = max(np.abs(biases).max(), 1e-6)
    sizes   = 30 + 200 * np.abs(biases) / abs_max

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    sc   = ax.scatter(
        valid['longitude'], valid['latitude'],
        c=biases, cmap='RdBu_r', norm=norm,
        s=sizes, zorder=4, edgecolors='black', linewidths=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax, label='Bias  (pred − actual, mm)', shrink=0.8)
    cbar.ax.axhline(0, color='black', linewidth=1)

    worst = valid.reindex(valid['bias'].abs().nlargest(top_n_labels).index)
    for _, row in worst.iterrows():
        ax.annotate(
            f"{row['station_id']} ({row['bias']:+.2f})",
            xy=(row['longitude'], row['latitude']),
            xytext=(6, 6), textcoords='offset points',
            fontsize=7, color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        )

    if bounds:
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
    Horizontal bar chart of the top_n worst-performing stations.
    Matches the style of src/visualization/error_analysis.plot_error_ranking.

    Parameters
    ----------
    station_df : DataFrame  output of compute_per_station_metrics()
    metric     : str   column to rank by ('mae', 'rmse', 'f1')
    top_n      : int   number of stations to show
    title, ax, save_path : standard

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
    labels = ranked['station_id'].astype(str).values
    vals   = ranked[metric].values

    bars = ax.barh(labels, vals, color=colour, edgecolor='black', linewidth=0.4)
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
# 5.  Error vs. spatial isolation
# ---------------------------------------------------------------------------

def plot_error_vs_isolation(
    station_df: pd.DataFrame,
    train_station_ids: list,
    coordinates: dict,
    metric: str = 'mae',
    title: str | None = None,
    ax=None,
    save_path: str | None = None,
):
    """
    Scatter: per-station error vs. distance to the nearest training gauge.
    Matches the style of src/visualization/error_analysis.plot_error_vs_isolation.

    Parameters
    ----------
    station_df        : DataFrame  output of compute_per_station_metrics()
    train_station_ids : list[str]  station IDs used for training in this fold
    coordinates       : dict  {station_id: (lat, lon)}
    metric            : str   error column to plot on y-axis
    title, ax, save_path : standard

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    test_df = station_df.dropna(subset=[metric, 'longitude', 'latitude'])

    train_rows = [
        (coordinates[s][0], coordinates[s][1])
        for s in train_station_ids if s in coordinates
    ]
    if not train_rows:
        ax.text(0.5, 0.5, 'No training station coordinates available.',
                transform=ax.transAxes, ha='center')
        return ax

    train_latlon = np.radians(np.array(train_rows))
    test_latlon  = np.radians(test_df[['latitude', 'longitude']].values)

    nbrs = NearestNeighbors(n_neighbors=1, metric='haversine')
    nbrs.fit(train_latlon)
    dists_rad, _ = nbrs.kneighbors(test_latlon)
    dists_km = dists_rad.flatten() * 6371.0

    errors = test_df[metric].values
    ax.scatter(dists_km, errors, alpha=0.7, edgecolors='black', linewidths=0.4, s=60)

    if len(dists_km) > 2:
        z = np.polyfit(dists_km, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(dists_km.min(), dists_km.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=1.5,
                label=f'Trend (slope={z[0]:.3f})')
        ax.legend(fontsize=8)

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
# 6.  Regression scatter (single model)
# ---------------------------------------------------------------------------

def plot_regression_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = 'Model',
    metrics: dict | None = None,
    title: str | None = None,
    ax=None,
    save_path: str | None = None,
):
    """
    Scatter plot of actual vs. predicted rainfall with metrics annotation.

    Parameters
    ----------
    actual      : np.ndarray  ground-truth values (mm/hr)
    predicted   : np.ndarray  model predictions (mm/hr)
    model_name  : str         used as default title and metrics-box label
    metrics     : dict | None  pre-computed {label: value}; computed if None
    title, ax, save_path : standard

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask      = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]

    if metrics is None:
        r, _  = pearsonr(actual, predicted)
        rmse  = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae   = float(np.mean(np.abs(actual - predicted)))
        metrics = {'Pearson r': r, 'RMSE (mm/hr)': rmse, 'MAE (mm/hr)': mae}

    ax.scatter(actual, predicted, alpha=0.4, s=10, edgecolors='none')

    plot_bound = max(float(np.nanmax(actual)), float(np.nanmax(predicted)), 1.0)
    ax.plot([0, plot_bound], [0, plot_bound], 'r--', linewidth=1.5,
            label='Perfect prediction')

    text = '\n'.join(f'{k} = {v:.3f}' for k, v in metrics.items())
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, verticalalignment='top', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    )

    ax.set_xlabel('Actual (mm/hr)')
    ax.set_ylabel('Predicted (mm/hr)')
    ax.set_title(title or model_name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 6b. Density regression scatter (single model)
# ---------------------------------------------------------------------------

def plot_density_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = 'Model',
    metrics: dict | None = None,
    title: str | None = None,
    log_scale: bool = True,
    ax=None,
    save_path: str | None = None,
):
    """
    Actual vs. predicted scatter where each point is coloured by local point
    density (via kernel density estimation).  Helps distinguish the crowded
    low-rainfall region from sparser high-rainfall events.

    Parameters
    ----------
    actual, predicted : np.ndarray  ground-truth / model values (mm/hr)
    model_name        : str         used as default title and metrics label
    metrics           : dict | None pre-computed {label: value}; computed if None
    title, ax, save_path : standard
    log_scale         : bool        if True, axes are log-scaled (log(x+1))
                                    which spreads out the dense low-value region

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask      = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]

    if metrics is None:
        r, _  = pearsonr(actual, predicted)
        rmse  = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae   = float(np.mean(np.abs(actual - predicted)))
        metrics = {'Pearson r': r, 'RMSE (mm/hr)': rmse, 'MAE (mm/hr)': mae}

    # Compute KDE density for each point
    xy  = np.vstack([actual, predicted])
    kde = gaussian_kde(xy)
    density = kde(xy)
    # Sort so densest points are drawn on top
    sort_idx = np.argsort(density)
    a_sorted, p_sorted, d_sorted = actual[sort_idx], predicted[sort_idx], density[sort_idx]

    if log_scale:
        a_plot = np.log1p(a_sorted)
        p_plot = np.log1p(p_sorted)
    else:
        a_plot, p_plot = a_sorted, p_sorted

    sc = ax.scatter(a_plot, p_plot, c=d_sorted, cmap='plasma', s=10,
                    edgecolors='none', alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Density', shrink=0.8)

    plot_bound = max(float(a_plot.max()), float(p_plot.max()), 0.1)
    ax.plot([0, plot_bound], [0, plot_bound], 'r--', linewidth=1.5,
            label='Perfect prediction')

    text = '\n'.join(f'{k} = {v:.3f}' for k, v in metrics.items())
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, verticalalignment='top', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
    )

    xlabel = 'Actual — log(mm/hr + 1)' if log_scale else 'Actual (mm/hr)'
    ylabel = 'Predicted — log(mm/hr + 1)' if log_scale else 'Predicted (mm/hr)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or model_name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 7.  Regression comparison (multiple models)
# ---------------------------------------------------------------------------

def plot_regression_comparison(
    results: dict,
    title: str = 'Model Comparison — Actual vs Predicted',
    figsize_per_panel: tuple[float, float] = (5, 5),
    save_path: str | None = None,
):
    """
    Side-by-side regression scatter plots for multiple models / benchmarks.

    Parameters
    ----------
    results : dict
        {model_name: {
            'actual':     np.ndarray,
            'predicted':  np.ndarray,
            'metrics':    dict | None,   # optional pre-computed metrics
        }}
        Add as many models as needed; layout adjusts automatically.
    title            : str   figure suptitle
    figsize_per_panel: (w, h) size of each subplot panel
    save_path        : str   optional save path

    Returns
    -------
    fig : matplotlib Figure

    Example
    -------
    plot_regression_comparison({
        'IDW':     {'actual': idw_result['actual_values'],
                    'predicted': idw_result['predicted_values'],
                    'metrics': idw_result['metrics']},
        'Kriging': {'actual': kriging_result['actual_values'],
                    'predicted': kriging_result['predicted_values'],
                    'metrics': kriging_result['metrics']},
    })
    """
    n    = len(results)
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(w * n, h))
    if n == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, results.items()):
        plot_regression_scatter(
            actual=np.asarray(data['actual']),
            predicted=np.asarray(data['predicted']),
            model_name=model_name,
            metrics=data.get('metrics'),
            ax=ax,
        )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# 8.  Aggregate metrics comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    results: dict,
    metrics: list[str] | tuple[str, ...] = ('rmse', 'mae', 'f1'),
    title: str = 'Benchmark Comparison',
    ax=None,
    save_path: str | None = None,
):
    """
    Grouped bar chart comparing aggregate metrics across multiple models.

    Parameters
    ----------
    results : dict
        {model_name: {'rmse': float, 'mae': float, 'f1': float, ...}}
        Use the keys from the benchmark return dicts, e.g.
        {'average_RMSE_loss', 'average_MAE_loss', 'f1'} or the normalised
        keys ('rmse', 'mae', 'f1') – both are accepted.
    metrics : list[str]
        Which metrics to include. Accepts both 'rmse' and 'average_RMSE_loss'.
    title, ax, save_path : standard

    Returns
    -------
    ax : matplotlib Axes

    Example
    -------
    plot_metrics_comparison({
        'IDW':     idw_result,
        'Kriging': kriging_result,
    })
    """
    # Normalise metric names: accept both 'rmse' and 'average_RMSE_loss' etc.
    _alias = {
        'rmse': ['rmse', 'average_rmse_loss', 'average_RMSE_loss'],
        'mae':  ['mae',  'average_mae_loss',  'average_MAE_loss'],
        'f1':   ['f1'],
    }

    def _get(d: dict, key: str):
        for alias in _alias.get(key.lower(), [key]):
            if alias in d:
                return d[alias]
        return np.nan

    model_names = list(results.keys())
    valid_metrics = [m for m in metrics
                     if any(not np.isnan(_get(v, m)) for v in results.values())]

    n_models  = len(model_names)
    n_metrics = len(valid_metrics)
    x     = np.arange(n_metrics)
    width = 0.8 / n_models

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, n_metrics * 2), 5))

    colours = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    for i, (model_name, colour) in enumerate(zip(model_names, colours)):
        vals   = [_get(results[model_name], m) for m in valid_metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=model_name,
                        color=colour, edgecolor='black', linewidth=0.4)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in valid_metrics])
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 9.  Per-timestep metrics (single model)
# ---------------------------------------------------------------------------

def plot_timestep_metrics(
    per_timestep_metrics: list[dict],
    metrics: list[str] | tuple[str, ...] = ('rmse', 'mae'),
    model_name: str = 'Model',
    title: str | None = None,
    ts_display_start: str | None = None,
    ts_frame_count: int | None = None,
    ax=None,
    save_path: str | None = None,
):
    """
    Line plot of per-timestep RMSE / MAE for a single model.

    Parameters
    ----------
    per_timestep_metrics : list of dicts
        [{'timestamp': ..., 'rmse': float, 'mae': float}]
        Returned by benchmark functions when return_detailed=True.
    metrics          : which keys to plot ('rmse', 'mae')
    model_name       : str  used in legend labels
    ts_display_start : str | None  timestamp string to start display from
                       e.g. "2024-08-01"; None = from beginning
    ts_frame_count   : int | None  number of timesteps to display; None = all
    title, ax, save_path : standard

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    df = pd.DataFrame(per_timestep_metrics)
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Apply display window
    if ts_display_start is not None:
        df = df[df.index >= pd.Timestamp(ts_display_start)]
    if ts_frame_count is not None:
        df = df.iloc[:ts_frame_count]

    for metric in metrics:
        if metric in df.columns:
            ax.plot(df.index, df[metric], linewidth=0.8, alpha=0.8,
                    label=f'{model_name} {metric.upper()}')

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Error (mm/hr)')
    ax.set_title(title or f'{model_name} — per-timestep error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 10.  Per-timestep comparison (multiple models)
# ---------------------------------------------------------------------------

def plot_timestep_comparison(
    results: dict,
    metric: str = 'rmse',
    title: str | None = None,
    rolling_window: int | None = None,
    ts_display_start: str | None = None,
    ts_frame_count: int | None = None,
    ax=None,
    save_path: str | None = None,
):
    """
    Overlaid time series of per-timestep error for multiple models.

    Parameters
    ----------
    results : dict
        {model_name: {'per_timestep_metrics': list[dict]}}
        The inner list must contain 'timestamp' and the requested metric key.
    metric           : str   'rmse' or 'mae'
    title            : str   optional title
    rolling_window   : int   optional rolling-average window (number of timesteps)
    ts_display_start : str | None  timestamp string to start display from; None = beginning
    ts_frame_count   : int | None  number of timesteps to display; None = all
    ax, save_path    : standard

    Returns
    -------
    ax : matplotlib Axes

    Example
    -------
    plot_timestep_comparison({
        'IDW':     idw_result,
        'Kriging': kriging_result,
    }, metric='rmse', rolling_window=96)   # 96 × 15-min = 24-hour rolling avg
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    colours = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    for (model_name, data), colour in zip(results.items(), colours):
        ts_list = data.get('per_timestep_metrics', [])
        if not ts_list:
            continue

        df = pd.DataFrame(ts_list)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        if metric not in df.columns:
            continue

        if ts_display_start is not None:
            df = df[df.index >= pd.Timestamp(ts_display_start)]
        if ts_frame_count is not None:
            df = df.iloc[:ts_frame_count]

        series = df[metric]
        if rolling_window:
            series = series.rolling(rolling_window, min_periods=1).mean()

        ax.plot(series.index, series.values, linewidth=1.0, alpha=0.85,
                label=model_name, color=colour)

    suffix = f' (rolling {rolling_window})' if rolling_window else ''
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(f'{metric.upper()} (mm/hr)')
    ax.set_title(title or f'Per-timestep {metric.upper()} comparison{suffix}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return ax


# ---------------------------------------------------------------------------
# 11.  Grid interpolation + spatial heatmap (mirrors grid_prediction.py)
# ---------------------------------------------------------------------------

def interpolate_to_grid(
    predictions_df: pd.DataFrame,
    coordinates: dict,
    bounds: dict,
    resolution_km: float = 1.0,
    power: float = 2.0,
    n_nearest: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    IDW-interpolate per-station benchmark predictions onto a regular grid for
    every timestep, producing the same [T, n_rows, n_cols] array that
    `predict_on_grid` returns for GNN models.

    This lets you use `plot_rainfall_grid`, `plot_rainfall_sequence`, and
    `animate_rainfall_grid` with IDW / Kriging results, exactly as in
    analysis.ipynb.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Indexed by timestamp, columns = station IDs, values = predicted
        rainfall (mm/hr). Returned by benchmark functions with
        return_detailed=True as ``result['predictions_df']``.
    coordinates : dict
        {station_id: (lat, lon)}
    bounds : dict
        {'left', 'right', 'top', 'bottom'}  in decimal degrees
    resolution_km : float
        Grid spacing in kilometres (default 1 km).
    power : float
        IDW exponent for the spatial re-interpolation (default 2).
    n_nearest : int
        Number of nearest stations to use per grid point.

    Returns
    -------
    predictions : np.ndarray  shape [T, n_rows, n_cols]
    actuals     : np.ndarray  shape [T, n_rows, n_cols]  (all-NaN if actuals_df
                  is not available – pass predictions_df for both)
    lons        : np.ndarray  shape [n_cols]
    lats        : np.ndarray  shape [n_rows]
    grid_shape  : (n_rows, n_cols)
    """
    # Build regular grid
    lat_centre = (bounds['top'] + bounds['bottom']) / 2.0
    lat_step   = resolution_km / 111.0
    lon_step   = resolution_km / (111.32 * np.cos(np.radians(lat_centre)))

    lats = np.arange(bounds['bottom'], bounds['top'],  lat_step)
    lons = np.arange(bounds['left'],   bounds['right'], lon_step)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    n_rows, n_cols = len(lats), len(lons)
    grid_shape = (n_rows, n_cols)

    grid_latlon = np.radians(
        np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1)
    )

    # Station coordinates for the columns that appear in predictions_df
    station_ids = list(predictions_df.columns)
    station_latlon = np.radians(
        np.array([coordinates[s] for s in station_ids])   # (lat, lon)
    )

    k = min(n_nearest, len(station_ids))
    nbrs = NearestNeighbors(n_neighbors=k, metric='haversine')
    nbrs.fit(station_latlon)
    dists_rad, nbr_idx = nbrs.kneighbors(grid_latlon)  # [n_grid, k]

    # Pre-compute IDW weights  [n_grid, k]
    dists_km = dists_rad * 6371.0
    dists_km  = np.maximum(dists_km, 1e-3)
    raw_w     = 1.0 / dists_km ** power
    weights   = raw_w / raw_w.sum(axis=1, keepdims=True)  # [n_grid, k]

    station_vals = predictions_df[station_ids].values  # [T, n_stations]

    T = len(predictions_df)
    pred_flat = np.full((T, n_rows * n_cols), np.nan)

    for t in range(T):
        row_vals = station_vals[t]                       # [n_stations]
        neigh    = row_vals[nbr_idx]                     # [n_grid, k]
        valid    = ~np.isnan(neigh)                      # [n_grid, k]
        w_masked = weights * valid
        w_sum    = w_masked.sum(axis=1)
        with np.errstate(invalid='ignore'):
            interpolated = np.where(
                w_sum > 0,
                (w_masked * np.nan_to_num(neigh)).sum(axis=1) / w_sum,
                np.nan,
            )
        pred_flat[t] = np.clip(interpolated, 0, None)

    predictions = pred_flat.reshape(T, n_rows, n_cols)
    return predictions, lons, lats, grid_shape


def plot_rainfall_grid(
    predictions: np.ndarray,
    grid_shape: tuple[int, int],
    bounds: dict,
    lons: np.ndarray | None = None,
    lats: np.ndarray | None = None,
    timestamp_idx: int = 0,
    mapping_df: pd.DataFrame | None = None,
    title: str | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    cmap: str = 'YlGnBu',
    show_outline: bool = True,
    log_scale: bool = False,
    ax=None,
):
    """
    Heatmap of a single timestep of gridded predictions.
    Mirrors src/visualization/grid_prediction.plot_rainfall_grid.

    Parameters
    ----------
    predictions   : np.ndarray  [T, n_rows, n_cols]
    grid_shape    : (n_rows, n_cols)
    bounds        : dict  {'left','right','top','bottom'}
    lons / lats   : optional 1-D coordinate arrays (unused for imshow, kept for API parity)
    timestamp_idx : int
    mapping_df    : optional DataFrame with 'longitude'/'latitude' columns
    title         : str   optional title
    vmin          : float colour-scale minimum (default 0.0)
    vmax          : float optional fixed maximum; auto if None
    boundaries    : list[float] optional BoundaryNorm breakpoints for discrete colour bands
    cmap          : str   matplotlib colourmap (default 'YlGnBu')
    show_outline  : bool  overlay Singapore coastline (default True)
    ax            : matplotlib Axes

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.cm as cm_mod

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    frame = predictions[timestamp_idx]

    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall grid. Pass vmax or boundaries, "
            "or load from config:\n"
            "  from src.visualization.error_analysis import get_viz_scales\n"
            "  scales = get_viz_scales()\n"
            "  plot_rainfall_grid(..., **scales['rainfall'])"
        )

    cmap_obj = cm_mod.get_cmap(cmap)
    if log_scale:
        _lv_min = max(vmin if vmin and vmin > 0 else 0.01, 0.01)
        _lv_max = vmax if vmax else frame.max()
        norm = mcolors.LogNorm(vmin=_lv_min, vmax=_lv_max)
        cbar_label = 'Predicted Rainfall (mm/hr) [log scale]'
    elif boundaries is not None:
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_obj.N, clip=True)
        cbar_label = 'Predicted Rainfall (mm/hr)'
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Predicted Rainfall (mm/hr)'

    im = ax.imshow(
        frame,
        extent=[bounds['left'], bounds['right'], bounds['bottom'], bounds['top']],
        origin='lower',
        cmap=cmap_obj,
        norm=norm,
        aspect='auto',
        interpolation='bilinear',
    )
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

    if mapping_df is not None:
        ax.scatter(
            mapping_df['longitude'], mapping_df['latitude'],
            c='red', s=25, zorder=5, label='Gauge stations',
            edgecolors='black', linewidths=0.4,
        )
        ax.legend(loc='lower right', fontsize=8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title or f'Predicted Rainfall — timestep {timestamp_idx}')
    ax.set_xlim(bounds['left'],  bounds['right'])
    ax.set_ylim(bounds['bottom'], bounds['top'])

    if show_outline:
        from src.visualization.main import visualise_singapore_outline
        visualise_singapore_outline(ax=ax)

    return ax


def plot_rainfall_sequence(
    predictions: np.ndarray,
    grid_shape: tuple[int, int],
    bounds: dict,
    timestep_indices: list[int],
    mapping_df: pd.DataFrame | None = None,
    titles: list[str] | None = None,
    figsize_per_panel: tuple[float, float] = (5, 4),
    vmin: float = 0.0,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    cmap: str = 'YlGnBu',
    show_outline: bool = True,
    log_scale: bool = False,
    save_path: str | None = None,
):
    """
    Row of N timestep heatmap panels in a single figure.
    Mirrors src/visualization/grid_prediction.plot_rainfall_sequence.

    Parameters
    ----------
    predictions      : np.ndarray  [T, n_rows, n_cols]
    timestep_indices : list[int]
    titles           : optional per-panel titles (e.g. timestamp strings)
    figsize_per_panel: (w, h)
    vmin             : float shared colour-scale minimum (default 0.0)
    vmax             : float shared maximum; auto if None
    boundaries       : list[float] optional BoundaryNorm breakpoints
    cmap             : str   colourmap (default 'YlGnBu')
    show_outline     : bool  overlay Singapore coastline (default True)
    save_path        : optional path

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(timestep_indices)
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(w * n, h))
    if n == 1:
        axes = [axes]

    subset = predictions[np.array(timestep_indices)]
    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall sequence. Pass vmax or boundaries, "
            "or load from config:\n"
            "  from src.visualization.error_analysis import get_viz_scales\n"
            "  scales = get_viz_scales()\n"
            "  plot_rainfall_sequence(..., **scales['rainfall'])"
        )
    global_vmax = vmax

    for i, (t_idx, ax) in enumerate(zip(timestep_indices, axes)):
        panel_title = titles[i] if titles else f'Timestep {t_idx}'
        plot_rainfall_grid(
            predictions, grid_shape, bounds,
            timestamp_idx=t_idx,
            mapping_df=mapping_df,
            title=panel_title,
            vmin=vmin,
            vmax=global_vmax,
            boundaries=boundaries,
            cmap=cmap,
            show_outline=show_outline,
            log_scale=log_scale,
            ax=ax,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def animate_rainfall_grid(
    predictions: np.ndarray,
    grid_shape: tuple[int, int],
    bounds: dict,
    timestep_indices: list[int] | None = None,
    timestamps: list | None = None,
    mapping_df: pd.DataFrame | None = None,
    interval_ms: int = 200,
    vmin: float = 0.0,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    cmap: str = 'YlGnBu',
    show_outline: bool = True,
    log_scale: bool = False,
    figsize: tuple[float, float] = (9, 7),
    save_path: str | None = None,
):
    """
    Animate gridded predictions as a spatial heatmap time series.
    Mirrors src/visualization/grid_prediction.animate_rainfall_grid.

    Parameters
    ----------
    predictions      : np.ndarray  [T, n_rows, n_cols]
    timestep_indices : optional subset; defaults to all T
    timestamps       : optional display labels per frame (e.g. pd.Timestamps)
    interval_ms      : delay between frames in ms
    vmin             : float  colour-scale minimum (default 0.0)
    vmax             : float  fixed colour-scale max (required unless boundaries set)
    boundaries       : list[float] optional BoundaryNorm breakpoints
    cmap             : str   matplotlib colourmap (default 'YlGnBu')
    show_outline     : bool  overlay Singapore coastline (default True)
    save_path        : optional – save as .gif (pillow) or .mp4 (ffmpeg)

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        In a notebook: HTML(anim.to_jshtml())
    """
    if timestep_indices is None:
        timestep_indices = list(range(predictions.shape[0]))

    subset = predictions[np.array(timestep_indices)]
    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall animation. Pass vmax or boundaries, "
            "or load from config:\n"
            "  from src.visualization.error_analysis import get_viz_scales\n"
            "  scales = get_viz_scales()\n"
            "  animate_rainfall_grid(..., **scales['rainfall'])"
        )
    import matplotlib.cm as cm_mod
    cmap_obj = cm_mod.get_cmap(cmap)
    if log_scale:
        _lv_min = max(vmin if vmin and vmin > 0 else 0.01, 0.01)
        _lv_max = vmax if vmax else subset.max()
        _norm = mcolors.LogNorm(vmin=_lv_min, vmax=_lv_max)
        cbar_label = 'Predicted Rainfall (mm/hr) [log scale]'
    elif boundaries is not None:
        _norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_obj.N, clip=True)
        cbar_label = 'Predicted Rainfall (mm/hr)'
    else:
        _norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Predicted Rainfall (mm/hr)'

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        subset[0],
        extent=[bounds['left'], bounds['right'], bounds['bottom'], bounds['top']],
        origin='lower',
        cmap=cmap_obj,
        norm=_norm,
        aspect='auto',
        interpolation='bilinear',
        animated=True,
    )
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

    if mapping_df is not None:
        ax.scatter(
            mapping_df['longitude'], mapping_df['latitude'],
            c='red', s=25, zorder=5, label='Gauge stations',
            edgecolors='black', linewidths=0.4,
        )
        ax.legend(loc='lower right', fontsize=8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(bounds['left'],  bounds['right'])
    ax.set_ylim(bounds['bottom'], bounds['top'])

    if show_outline:
        from src.visualization.main import visualise_singapore_outline
        visualise_singapore_outline(ax=ax)

    label = str(timestamps[0]) if timestamps else f'Timestep {timestep_indices[0]}'
    title = ax.set_title(label, fontsize=11)

    def _update(frame_i):
        im.set_data(subset[frame_i])
        lbl = str(timestamps[frame_i]) if timestamps else f'Timestep {timestep_indices[frame_i]}'
        title.set_text(lbl)
        return im, title

    anim = animation.FuncAnimation(
        fig, _update,
        frames=len(timestep_indices),
        interval=interval_ms,
        blit=True,
    )

    if save_path:
        writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
        anim.save(save_path, writer=writer, dpi=150)
        print(f"Animation saved to {save_path}")

    return anim


# ---------------------------------------------------------------------------
# 12.  Combined entry point for a single benchmark
# ---------------------------------------------------------------------------

def analyze_benchmark(
    result: dict,
    coordinates: dict,
    bounds: dict | None = None,
    train_station_ids: list | None = None,
    model_name: str = 'Benchmark',
    output_dir: str = '.',
    top_n: int = 10,
    rain_threshold: float = 0.5,
    metrics: tuple[str, str] = ('rmse', 'pearson_r'),
    scale_params: dict | None = None,
) -> pd.DataFrame:
    """
    Run all diagnostic plots for a single benchmark result and save them.

    Requires the benchmark to have been run with return_detailed=True so that
    'per_station_data', 'actual_values', 'predicted_values', and
    'per_timestep_metrics' are present in *result*.

    Produces up to six figures (saved as <slug>_*.png in output_dir):
        <slug>_<metric1>_map.png – spatial map for metrics[0]
        <slug>_<metric2>_map.png – spatial map for metrics[1]
        <slug>_bias_map.png      – spatial bias map
        <slug>_ranking.png       – worst-station bar chart
        <slug>_isolation.png     – MAE vs. distance to training gauge (optional)
        <slug>_regression.png    – actual vs. predicted scatter

    Parameters
    ----------
    result           : dict  benchmark return value (return_detailed=True)
    coordinates      : dict  {station_id: (lat, lon)}
    bounds           : dict  {'left','right','top','bottom'}  optional
    train_station_ids: list  optional – enables the isolation plot
    model_name       : str   used as file-name prefix and plot titles
    output_dir       : str   directory for saved figures
    top_n            : int   stations to highlight in rankings
    rain_threshold   : float for per-station F1 classification
    metrics          : tuple[str, str]  two metrics to compare side-by-side
                       choose from: 'rmse', 'mae', 'f1', 'pearson_r'
                       default: ('rmse', 'pearson_r')
    scale_params     : dict | None  optional per-metric scale overrides, e.g.:
                       {'rmse': {'vmin': 0, 'vmax': 3, 'boundaries': [0,0.5,1,2,3]},
                        'pearson_r': {'vmin': 0, 'vmax': 1}}

    Returns
    -------
    station_df : pd.DataFrame  per-station metrics with coordinates
    """
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(' ', '_')
    scale_params = scale_params or {}

    station_df = compute_per_station_metrics(
        result['per_station_data'], coordinates, rain_threshold=rain_threshold
    )

    print(f"\n=== {model_name} — per-station analysis ===")
    print(f"Stations analysed: {len(station_df)}")
    show_cols = ['station_id', 'mae', 'rmse', 'bias', 'f1', 'pearson_r']
    show_cols = [c for c in show_cols if c in station_df.columns]
    print(station_df[show_cols].sort_values('mae', ascending=False)
          .head(top_n).to_string(index=False))

    metric1, metric2 = metrics

    # Plot 1: First metric spatial map
    if metric1 in station_df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        plot_spatial_error_map(station_df, bounds=bounds, metric=metric1,
                               top_n_labels=top_n, ax=ax1,
                               title=f'{model_name} — per-station {metric1.upper()}',
                               **scale_params.get(metric1, {}))
        fig1.tight_layout()
        fig1.savefig(out / f'{slug}_{metric1}_map.png', dpi=200, bbox_inches='tight')
        plt.close(fig1)

    # Plot 2: Second metric spatial map
    if metric2 in station_df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        plot_spatial_error_map(station_df, bounds=bounds, metric=metric2,
                               top_n_labels=top_n, ax=ax2,
                               title=f'{model_name} — per-station {metric2.upper()}',
                               **scale_params.get(metric2, {}))
        fig2.tight_layout()
        fig2.savefig(out / f'{slug}_{metric2}_map.png', dpi=200, bbox_inches='tight')
        plt.close(fig2)

    # Plot 3: Bias map
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    plot_bias_map(station_df, bounds=bounds, top_n_labels=top_n, ax=ax3,
                  title=f'{model_name} — prediction bias')
    fig3.tight_layout()
    fig3.savefig(out / f'{slug}_bias_map.png', dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # Plot 4: Ranking bar chart (metric1 + metric2 side-by-side)
    fig4, (axA, axB) = plt.subplots(1, 2, figsize=(14, max(4, top_n * 0.45)))
    rank_m1 = metric1 if metric1 in station_df.columns else 'mae'
    rank_m2 = metric2 if metric2 in station_df.columns else 'f1'
    plot_error_ranking(station_df, metric=rank_m1, top_n=top_n, ax=axA,
                       title=f'{model_name} — worst by {rank_m1.upper()}')
    plot_error_ranking(station_df, metric=rank_m2, top_n=top_n, ax=axB,
                       title=f'{model_name} — worst by {rank_m2.upper()}')
    fig4.suptitle(f'{model_name} — worst-performing stations', fontsize=13)
    fig4.tight_layout()
    fig4.savefig(out / f'{slug}_ranking.png', dpi=200, bbox_inches='tight')
    plt.close(fig4)

    # Plot 5: Error vs. isolation (optional)
    if train_station_ids is not None:
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        plot_error_vs_isolation(
            station_df, train_station_ids, coordinates,
            metric='mae', ax=ax5,
            title=f'{model_name} — MAE vs. distance to training gauge',
        )
        fig5.tight_layout()
        fig5.savefig(out / f'{slug}_isolation.png', dpi=200, bbox_inches='tight')
        plt.close(fig5)

    # Plot 6: Regression scatter
    if 'actual_values' in result and 'predicted_values' in result:
        fig6, ax6 = plt.subplots(figsize=(6, 6))
        plot_regression_scatter(
            actual=np.asarray(result['actual_values']),
            predicted=np.asarray(result['predicted_values']),
            model_name=model_name,
            metrics=result.get('metrics'),
            ax=ax6,
        )
        fig6.tight_layout()
        fig6.savefig(out / f'{slug}_regression.png', dpi=200, bbox_inches='tight')
        plt.close(fig6)

    print(f"Figures saved to: {out.resolve()}")
    return station_df


# ---------------------------------------------------------------------------
# Per-station rainfall time-series plot
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
                     e.g. result["predictions_df"] from run_IDW_benchmark
    actuals_df     : DataFrame  timestamps × station_ids (actual values)
                     e.g. result["actuals_df"] from run_IDW_benchmark
    station_id     : str        column to plot (station ID)
    time_start     : str | None e.g. "2024-08-01" — restricts x-axis start
    time_end       : str | None e.g. "2024-08-31" — restricts x-axis end
    title          : str        optional plot title
    ax             : Axes       optional existing axes
    save_path      : str        optional save path

    Returns
    -------
    ax : matplotlib Axes
    """
    if station_id not in predictions_df.columns:
        raise ValueError(
            f"Station '{station_id}' not in predictions_df. "
            f"Available: {list(predictions_df.columns)}"
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
