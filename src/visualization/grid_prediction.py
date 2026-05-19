"""
grid_prediction.py
==================
Produce a regular 1 km × 1 km rainfall grid by augmenting the trained
heterogeneous GNN graph with virtual grid-point nodes.

Core idea
---------
Grid points are inserted as new raingauge nodes with zeroed data features
(rainfall value + validity flag).  The model predicts their rainfall from
the surrounding real gauge nodes and, transitively, from radar / CML via
those gauges.  LPE is recomputed over the full augmented graph so grid
nodes get consistent positional encodings.

Exported symbols
----------------
generate_grid_coords   – build a lon/lat grid DataFrame within bounds
predict_on_grid        – run the model over every timestep; return [T, rows, cols]
plot_rainfall_grid     – visualise a single timestep as a spatial heatmap
plot_rainfall_sequence – plot a row of N timestep snapshots in one figure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import torch

from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from src.visualization.main import visualise_singapore_outline

# Number of data features per raingauge node – must match logic_hetero._DATA_FEATURE_DIM
_DATA_FEATURE_DIM = 2  # [rainfall_value, validity_flag]; LPE columns start at index 2


# ---------------------------------------------------------------------------
# 1.  Grid generation
# ---------------------------------------------------------------------------

def generate_grid_coords(
    bounds: dict,
    resolution_km: float = 1.0,
) -> tuple[pd.DataFrame, tuple[int, int]]:
    """
    Return a regular grid of (longitude, latitude) points within *bounds*
    at the requested *resolution_km* spacing.

    Parameters
    ----------
    bounds       : dict  {'left', 'right', 'top', 'bottom'}  in decimal degrees
    resolution_km: float  grid spacing in kilometres

    Returns
    -------
    grid_coords  : pd.DataFrame  columns ['longitude', 'latitude']
    grid_shape   : (n_rows, n_cols)  – rows = latitude axis, cols = longitude axis
    """
    lat_centre = (bounds['top'] + bounds['bottom']) / 2.0
    lat_step = resolution_km / 111.0                                   # degrees per km
    lon_step = resolution_km / (111.32 * np.cos(np.radians(lat_centre)))

    lats = np.arange(bounds['bottom'], bounds['top'],  lat_step)
    lons = np.arange(bounds['left'],   bounds['right'], lon_step)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_coords = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude':  lat_grid.flatten(),
    })
    return grid_coords, (len(lats), len(lons))


# ---------------------------------------------------------------------------
# 2.  Graph augmentation + inference
# ---------------------------------------------------------------------------

def predict_on_grid(
    model,
    heterodata: HeteroData,
    mapping_df: pd.DataFrame,
    bounds: dict,
    resolution_km: float = 1.0,
    knn_gauge: int = 5,
    device: str = 'cpu',
    include_lpe: bool = True,
    lpe_k: int = 4,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int]]:
    """
    Predict rainfall on a regular grid for every timestep in *heterodata*.

    The function augments the test graph by appending virtual grid nodes as
    new raingauge nodes with zeroed data features.  Each grid node is
    connected to its *knn_gauge* nearest real gauges.  LPE is optionally
    recomputed for the full augmented graph.

    Parameters
    ----------
    model       : trained GNNInductiveHetero (eval mode will be set internally)
    heterodata  : normalised HeteroData returned by GaugeGraphNew.get_test_heterodata()
                  (shape: [N, T, F] tensors)
    mapping_df  : DataFrame with columns ['longitude', 'latitude', 'id'] for real gauges
    bounds      : dict  {'left', 'right', 'top', 'bottom'}
    resolution_km : float  grid spacing in km
    knn_gauge   : int  nearest real gauges to connect to each grid node
    device      : str  torch device string
    include_lpe : bool  recompute Laplacian PE for the augmented graph
    lpe_k       : int  number of LPE eigenvectors (must match model training)

    Returns
    -------
    predictions : np.ndarray  shape [T, n_rows, n_cols]  (in original rainfall units)
    grid_coords : pd.DataFrame  columns ['longitude', 'latitude']
    grid_shape  : (n_rows, n_cols)
    """
    model.eval()

    grid_coords, grid_shape = generate_grid_coords(bounds, resolution_km)
    n_grid = len(grid_coords)
    n_real = heterodata['raingauge'].x.shape[0]
    T      = heterodata['raingauge'].x.shape[1]
    F      = heterodata['raingauge'].x.shape[2]

    # ------------------------------------------------------------------
    # Step 1 – append grid nodes to the raingauge node set
    # ------------------------------------------------------------------
    augmented = heterodata.clone()

    # Always work from the raw data features only (strip any pre-existing LPE
    # columns so that include_lpe alone controls whether LPE is present).
    real_x = heterodata['raingauge'].x[:, :, :_DATA_FEATURE_DIM]  # [n_real, T, 2]
    grid_x = torch.zeros(n_grid, T, _DATA_FEATURE_DIM)            # [n_grid, T, 2]
    grid_y = torch.zeros(n_grid, T, 1)
    augmented['raingauge'].x = torch.cat([real_x, grid_x], dim=0)
    augmented['raingauge'].y = torch.cat([heterodata['raingauge'].y, grid_y], dim=0)

    # Mask: only grid points are "unknown" (to be predicted)
    grid_mask = torch.zeros(n_real + n_grid, dtype=torch.bool)
    grid_mask[n_real:] = True
    augmented['raingauge'].mask = grid_mask
    augmented['raingauge'].num_nodes = n_real + n_grid

    # ------------------------------------------------------------------
    # Step 2 – build grid-to-gauge KNN edges
    # ------------------------------------------------------------------
    # haversine metric requires (lat, lon) in radians
    gauge_latlon  = np.radians(mapping_df[['latitude', 'longitude']].values)
    grid_latlon   = np.radians(grid_coords[['latitude', 'longitude']].values)

    nbrs = NearestNeighbors(n_neighbors=knn_gauge, metric='haversine')
    nbrs.fit(gauge_latlon)
    distances_rad, nbr_indices = nbrs.kneighbors(grid_latlon)

    earth_radius_km = 6371.0
    distances_km = distances_rad * earth_radius_km

    src, dst, weights = [], [], []
    for grid_i in range(n_grid):
        for k in range(knn_gauge):
            gauge_j = int(nbr_indices[grid_i, k])
            dist_km = float(distances_km[grid_i, k])
            src.append(n_real + grid_i)
            dst.append(gauge_j)
            weights.append(1.0 / max(dist_km, 1e-3))   # inverse-distance weight

    new_edge_index  = torch.tensor([src, dst], dtype=torch.long)
    new_edge_weight = torch.tensor(weights, dtype=torch.float32)
    new_edge_weight = new_edge_weight / new_edge_weight.max()   # normalise to [0, 1]

    # Add both directions (grid→gauge and gauge→grid)
    rev_edge_index  = new_edge_index.flip(0)
    existing_idx    = augmented['raingauge', 'connects', 'raingauge'].edge_index
    existing_attr   = augmented['raingauge', 'connects', 'raingauge'].edge_attr
    augmented['raingauge', 'connects', 'raingauge'].edge_index = torch.cat(
        [existing_idx, new_edge_index, rev_edge_index], dim=1
    )
    augmented['raingauge', 'connects', 'raingauge'].edge_attr = torch.cat(
        [existing_attr, new_edge_weight, new_edge_weight], dim=0
    )

    # ------------------------------------------------------------------
    # Step 3 – recompute LPE for the augmented graph (optional)
    # ------------------------------------------------------------------
    # If LPE was used during training, recompute it so that grid nodes get a
    # consistent positional encoding over the augmented graph topology.
    if include_lpe and lpe_k > 0:
        temp = Data(
            x=torch.zeros(n_real + n_grid, 1),
            edge_index=augmented['raingauge', 'connects', 'raingauge'].edge_index,
            num_nodes=n_real + n_grid,
        )
        lpe_transform = AddLaplacianEigenvectorPE(k=lpe_k, attr_name='laplacian_pe')
        temp = lpe_transform(temp)
        lpe = temp.laplacian_pe                              # [N_total, lpe_k]
        lpe_expanded = lpe.unsqueeze(1).expand(-1, T, -1)   # [N_total, T, lpe_k]

        data_part = augmented['raingauge'].x[:, :, :_DATA_FEATURE_DIM]
        augmented['raingauge'].x = torch.cat([data_part, lpe_expanded], dim=2)

    # ------------------------------------------------------------------
    # Step 4 – run inference one timestep at a time
    # ------------------------------------------------------------------
    all_grid_preds: list[np.ndarray] = []

    with torch.no_grad():
        for t in range(T):
            snap = HeteroData()

            # Raingauge snapshot at timestep t
            snap['raingauge'].x         = augmented['raingauge'].x[:, t, :].to(device)
            snap['raingauge'].y         = augmented['raingauge'].y[:, t, :].to(device)
            snap['raingauge'].mask      = augmented['raingauge'].mask.to(device)
            snap['raingauge'].num_nodes = n_real + n_grid

            # All other node types (radar, cml) – slice their timestep
            for node_type in augmented.node_types:
                if node_type == 'raingauge':
                    continue
                snap[node_type].x         = augmented[node_type].x[:, t, :].to(device)
                snap[node_type].num_nodes = augmented[node_type].x.shape[0]

            # Edge indices and attributes
            for edge_type in augmented.edge_types:
                snap[edge_type].edge_index = augmented[edge_type].edge_index.to(device)
                if hasattr(augmented[edge_type], 'edge_attr'):
                    snap[edge_type].edge_attr = augmented[edge_type].edge_attr.to(device)

            # Zero data features for grid nodes (same masking convention as training)
            mask_t  = snap['raingauge'].mask
            x_input = snap['raingauge'].x.clone()
            x_input[mask_t, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {nt: snap[nt].x for nt in snap.node_types}
            x_dict['raingauge'] = x_input

            edge_attr_dict = {
                et: snap[et].edge_attr
                for et in snap.edge_types
                if hasattr(snap[et], 'edge_attr')
            }

            out = model(x_dict, snap.edge_index_dict, edge_attr_dict)

            grid_preds = out['raingauge'][mask_t].cpu().numpy().flatten()
            all_grid_preds.append(grid_preds)

    predictions = np.stack(all_grid_preds, axis=0)          # [T, n_grid]
    predictions = predictions.reshape(T, *grid_shape)       # [T, n_rows, n_cols]
    return predictions, grid_coords, grid_shape


# ---------------------------------------------------------------------------
# 3.  Visualisation helpers
# ---------------------------------------------------------------------------

def plot_rainfall_grid(
    predictions: np.ndarray,
    grid_shape: tuple[int, int],
    bounds: dict,
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
    Plot a single timestep of grid predictions as a filled spatial heatmap.

    Parameters
    ----------
    predictions   : np.ndarray  [T, n_rows, n_cols] from predict_on_grid
    grid_shape    : (n_rows, n_cols)
    bounds        : dict  {'left', 'right', 'top', 'bottom'}
    timestamp_idx : int   which timestep to show
    mapping_df    : DataFrame  optional – overlay real gauge positions as red dots
    title         : str   optional plot title
    vmin          : float colour-scale minimum (default 0.0)
    vmax          : float optional fixed colour-scale maximum (mm); auto if None
    boundaries    : list[float] optional explicit colour-transition breakpoints;
                    when supplied, a BoundaryNorm is used for discrete colour bands.
                    Example: [0, 0.5, 2, 5, 10, 20]
    cmap          : str   matplotlib colourmap name (default 'YlGnBu')
    ax            : matplotlib Axes  optional – draws into existing axes

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    frame = predictions[timestamp_idx]                       # [n_rows, n_cols]

    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall grid. Pass vmax or boundaries, "
            "or load from config:\n"
            "  scales = get_viz_scales()\n"
            "  plot_rainfall_grid(..., **scales['rainfall'])"
        )

    import matplotlib.cm as cm_mod
    cmap_obj = cm_mod.get_cmap(cmap)
    if log_scale:
        _lv_min = max(vmin if vmin and vmin > 0 else 0.01, 0.01)
        _lv_max = vmax if vmax else frame.max()
        norm = mcolors.LogNorm(vmin=_lv_min, vmax=_lv_max)
        cbar_label = 'Predicted Rainfall (mm) [log scale]'
    elif boundaries is not None:
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_obj.N, clip=True)
        cbar_label = 'Predicted Rainfall (mm)'
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Predicted Rainfall (mm)'

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

    if show_outline:
        visualise_singapore_outline(ax=ax)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title or f'Predicted Rainfall — timestep {timestamp_idx}')
    ax.set_xlim(bounds['left'],  bounds['right'])
    ax.set_ylim(bounds['bottom'], bounds['top'])
    return ax


def plot_rainfall_sequence(
    predictions: np.ndarray,
    grid_shape: tuple[int, int],
    bounds: dict,
    timestep_indices: list[int],
    mapping_df: pd.DataFrame | None = None,
    titles: list[str] | None = None,
    figsize_per_panel: tuple[float, float] = (5, 4),
    show_outline: bool = True,
    vmin: float = 0.0,
    vmax: float | None = None,
    boundaries: list[float] | None = None,
    cmap: str = 'YlGnBu',
    log_scale: bool = False,
    save_path: str | None = None,
):
    """
    Plot a row of N timestep snapshots in a single figure.

    Parameters
    ----------
    predictions      : np.ndarray  [T, n_rows, n_cols]
    timestep_indices : list[int]   which timesteps to include (up to 6 recommended)
    titles           : list[str]   optional per-panel titles
    figsize_per_panel: (w, h)      size of each individual panel
    vmin             : float       shared colour-scale minimum (default 0.0)
    vmax             : float       shared colour-scale maximum; auto if None
    boundaries       : list[float] optional BoundaryNorm breakpoints (shared across panels)
    cmap             : str         matplotlib colourmap (default 'YlGnBu')
    save_path        : str         optional path to save the figure

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(timestep_indices)
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(w * n, h))
    if n == 1:
        axes = [axes]

    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall sequence. Pass vmax or boundaries, "
            "or load from config:\n"
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
) -> animation.FuncAnimation:
    """
    Animate grid predictions as a spatial heatmap time series.

    Parameters
    ----------
    predictions      : np.ndarray  [T, n_rows, n_cols] from predict_on_grid
    grid_shape       : (n_rows, n_cols)
    bounds           : dict  {'left', 'right', 'top', 'bottom'}
    timestep_indices : list[int]  subset of timesteps to animate; defaults to all
    timestamps       : list  optional display labels per timestep (e.g. pandas Timestamps)
    mapping_df       : DataFrame  optional – overlay real gauge positions as red dots
    interval_ms      : int  delay between frames in milliseconds
    vmin             : float  colour-scale minimum (default 0.0)
    vmax             : float  optional fixed colour-scale maximum (mm); auto if None
    boundaries       : list[float] optional BoundaryNorm breakpoints for discrete bands
    cmap             : str   matplotlib colourmap (default 'YlGnBu')
    figsize          : (w, h) figure size in inches
    save_path        : str  optional – save as .mp4 or .gif (requires ffmpeg / pillow)

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Call `HTML(anim.to_jshtml())` in a notebook cell to display inline.
    """
    import matplotlib.cm as cm_mod

    if timestep_indices is None:
        timestep_indices = list(range(predictions.shape[0]))

    subset = predictions[timestep_indices]

    if vmax is None and boundaries is None:
        raise ValueError(
            "Fixed scale required for rainfall animation. Pass vmax or boundaries, "
            "or load from config:\n"
            "  scales = get_viz_scales()\n"
            "  animate_rainfall_grid(..., **scales['rainfall'])"
        )

    cmap_obj = cm_mod.get_cmap(cmap)
    if log_scale:
        _lv_min = max(vmin if vmin and vmin > 0 else 0.01, 0.01)
        _lv_max = vmax if vmax else subset.max()
        norm = mcolors.LogNorm(vmin=_lv_min, vmax=_lv_max)
        cbar_label = 'Predicted Rainfall (mm) [log scale]'
    elif boundaries is not None:
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_obj.N, clip=True)
        cbar_label = 'Predicted Rainfall (mm)'
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Predicted Rainfall (mm)'

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        subset[0],
        extent=[bounds['left'], bounds['right'], bounds['bottom'], bounds['top']],
        origin='lower',
        cmap=cmap_obj,
        norm=norm,
        aspect='auto',
        interpolation='bilinear',
        animated=True,
    )
    cbar = plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

    if mapping_df is not None:
        ax.scatter(
            mapping_df['longitude'], mapping_df['latitude'],
            c='red', s=25, zorder=5, label='Gauge stations',
            edgecolors='black', linewidths=0.4,
        )
        ax.legend(loc='lower right', fontsize=8)

    if show_outline:
        visualise_singapore_outline(ax=ax)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(bounds['left'],  bounds['right'])
    ax.set_ylim(bounds['bottom'], bounds['top'])

    label = (
        str(timestamps[0]) if timestamps else f'Timestep {timestep_indices[0]}'
    )
    title = ax.set_title(label, fontsize=11)

    def _update(frame_i):
        im.set_data(subset[frame_i])
        lbl = (
            str(timestamps[frame_i]) if timestamps
            else f'Timestep {timestep_indices[frame_i]}'
        )
        title.set_text(lbl)
        return im, title

    anim = animation.FuncAnimation(
        fig,
        _update,
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
# 4.  ST model grid inference (GNNInductiveHeteroST)
# ---------------------------------------------------------------------------

def predict_on_grid_st(
    model,
    heterodata: HeteroData,
    mapping_df: pd.DataFrame,
    bounds: dict,
    window_size: int = 6,
    resolution_km: float = 1.0,
    knn_gauge: int = 5,
    device: str = 'cpu',
    include_lpe: bool = True,
    lpe_k: int = 4,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int]]:
    """
    Predict rainfall on a regular grid for every valid timestep in *heterodata*
    using a spatio-temporal model (GNNInductiveHeteroST).

    Mirrors predict_on_grid but additionally assembles a context window of
    W = *window_size* preceding timesteps for each node and passes it as
    x_context_dict to the ST model's forward method.

    Parameters
    ----------
    model        : trained GNNInductiveHeteroST (eval mode set internally)
    heterodata   : normalised HeteroData with [N, T, F] node features
    mapping_df   : DataFrame  columns ['longitude', 'latitude'] for real gauges
    bounds       : dict  {'left', 'right', 'top', 'bottom'}
    window_size  : int   number of preceding context timesteps (W)
    resolution_km: float grid spacing in km
    knn_gauge    : int   nearest real gauges to connect to each grid node
    device       : str   torch device string
    include_lpe  : bool  recompute Laplacian PE for the augmented graph
    lpe_k        : int   number of LPE eigenvectors (must match model training)

    Returns
    -------
    predictions  : np.ndarray  shape [T - window_size, n_rows, n_cols]
                   predictions[i] corresponds to timestep window_size + i
    grid_coords  : pd.DataFrame  columns ['longitude', 'latitude']
    grid_shape   : (n_rows, n_cols)
    """
    model.eval()

    grid_coords, grid_shape = generate_grid_coords(bounds, resolution_km)
    n_grid = len(grid_coords)
    n_real = heterodata['raingauge'].x.shape[0]
    T      = heterodata['raingauge'].x.shape[1]

    # ------------------------------------------------------------------
    # Step 1 – append grid nodes to the raingauge node set
    # ------------------------------------------------------------------
    augmented = heterodata.clone()

    real_x = heterodata['raingauge'].x[:, :, :_DATA_FEATURE_DIM]   # [n_real, T, 2]
    grid_x = torch.zeros(n_grid, T, _DATA_FEATURE_DIM)             # [n_grid, T, 2]
    grid_y = torch.zeros(n_grid, T, 1)
    augmented['raingauge'].x = torch.cat([real_x, grid_x], dim=0)
    augmented['raingauge'].y = torch.cat([heterodata['raingauge'].y, grid_y], dim=0)
    augmented['raingauge'].num_nodes = n_real + n_grid

    # ------------------------------------------------------------------
    # Step 2 – build grid-to-gauge KNN edges
    # ------------------------------------------------------------------
    gauge_latlon = np.radians(mapping_df[['latitude', 'longitude']].values)
    grid_latlon  = np.radians(grid_coords[['latitude', 'longitude']].values)

    nbrs = NearestNeighbors(n_neighbors=knn_gauge, metric='haversine')
    nbrs.fit(gauge_latlon)
    distances_rad, nbr_indices = nbrs.kneighbors(grid_latlon)

    distances_km = distances_rad * 6371.0

    src, dst, weights = [], [], []
    for grid_i in range(n_grid):
        for k in range(knn_gauge):
            gauge_j = int(nbr_indices[grid_i, k])
            dist_km = float(distances_km[grid_i, k])
            src.append(n_real + grid_i)
            dst.append(gauge_j)
            weights.append(1.0 / max(dist_km, 1e-3))

    new_edge_index  = torch.tensor([src, dst], dtype=torch.long)
    new_edge_weight = torch.tensor(weights, dtype=torch.float32)
    new_edge_weight = new_edge_weight / new_edge_weight.max()

    rev_edge_index = new_edge_index.flip(0)
    existing_idx   = augmented['raingauge', 'connects', 'raingauge'].edge_index
    existing_attr  = augmented['raingauge', 'connects', 'raingauge'].edge_attr
    augmented['raingauge', 'connects', 'raingauge'].edge_index = torch.cat(
        [existing_idx, new_edge_index, rev_edge_index], dim=1
    )
    augmented['raingauge', 'connects', 'raingauge'].edge_attr = torch.cat(
        [existing_attr, new_edge_weight, new_edge_weight], dim=0
    )

    # ------------------------------------------------------------------
    # Step 3 – recompute LPE for the augmented graph (optional)
    # ------------------------------------------------------------------
    if include_lpe and lpe_k > 0:
        temp = Data(
            x=torch.zeros(n_real + n_grid, 1),
            edge_index=augmented['raingauge', 'connects', 'raingauge'].edge_index,
            num_nodes=n_real + n_grid,
        )
        lpe_transform = AddLaplacianEigenvectorPE(k=lpe_k, attr_name='laplacian_pe')
        temp = lpe_transform(temp)
        lpe          = temp.laplacian_pe                              # [N_total, lpe_k]
        lpe_expanded = lpe.unsqueeze(1).expand(-1, T, -1)            # [N_total, T, lpe_k]

        data_part = augmented['raingauge'].x[:, :, :_DATA_FEATURE_DIM]
        augmented['raingauge'].x = torch.cat([data_part, lpe_expanded], dim=2)

    # ------------------------------------------------------------------
    # Step 4 – precompute static edge data (same for every timestep)
    # ------------------------------------------------------------------
    edge_index_dict = {
        et: augmented[et].edge_index.to(device)
        for et in augmented.edge_types
    }
    edge_attr_dict = {
        et: augmented[et].edge_attr.to(device)
        for et in augmented.edge_types
        if hasattr(augmented[et], 'edge_attr')
    }

    # ------------------------------------------------------------------
    # Step 5 – run inference one timestep at a time with context window
    # ------------------------------------------------------------------
    all_grid_preds: list[np.ndarray] = []

    with torch.no_grad():
        for t in range(window_size, T):
            ctx_idx = list(range(t - window_size, t))

            x_dict: dict = {}
            x_context_dict: dict = {}

            # Raingauge: zero the grid nodes' current-timestep data features
            x_cur = augmented['raingauge'].x[:, t, :].clone().to(device)
            x_cur[n_real:, :_DATA_FEATURE_DIM] = 0.0
            x_dict['raingauge']         = x_cur
            x_context_dict['raingauge'] = augmented['raingauge'].x[:, ctx_idx, :].to(device)

            # All other node types
            for ntype in augmented.node_types:
                if ntype == 'raingauge':
                    continue
                x_dict[ntype]         = augmented[ntype].x[:, t, :].to(device)
                x_context_dict[ntype] = augmented[ntype].x[:, ctx_idx, :].to(device)

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            grid_preds = out['raingauge'][n_real:].cpu().numpy().flatten()
            all_grid_preds.append(grid_preds)

    predictions = np.stack(all_grid_preds, axis=0)                # [T-W, n_grid]
    predictions = predictions.reshape(len(all_grid_preds), *grid_shape)
    return predictions, grid_coords, grid_shape
