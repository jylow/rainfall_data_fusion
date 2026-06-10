"""
benchmarks/processing/gridify.py
=================================
Utilities for converting sparse point observations to/from a regular spatial
grid, used to build CNN input channels.

Grid coordinates match the radar dataset: 0.01° resolution over Singapore,
lons ascending left→right, lats descending top→bottom.
"""

import numpy as np
from scipy.spatial import cKDTree

GRID_RES = 0.01  # degrees
GRID_BOUNDS = {"left": 103.6, "right": 104.1, "top": 1.5, "bottom": 1.188}


# ---------------------------------------------------------------------------
# Grid coordinate helpers
# ---------------------------------------------------------------------------

def make_grid_coords(bounds=None, resolution=GRID_RES):
    """Return (lons, lats) 1-D arrays for the regular grid.

    lons: ascending,  shape (W,)
    lats: descending, shape (H,)  — matches rasterio/radar convention
    """
    if bounds is None:
        bounds = GRID_BOUNDS
    lons = np.arange(bounds["left"] + resolution / 2, bounds["right"], resolution)
    lats = np.arange(bounds["top"] - resolution / 2, bounds["bottom"], -resolution)
    return lons.astype(np.float64), lats.astype(np.float64)


# ---------------------------------------------------------------------------
# IDW weights (precomputed once per fold / data-source)
# ---------------------------------------------------------------------------

def precompute_idw_weights(source_lons, source_lats, grid_lons, grid_lats,
                           power=2, k=10):
    """Build a dense IDW weight matrix [H*W, N_sources].

    The matrix is independent of the actual observation values, so it only
    needs to be computed once per set of source coordinates.  Grid values are
    then obtained cheaply via matrix multiplication::

        grid_flat [T, H*W] = values [T, N] @ weights.T [N, H*W]

    Parameters
    ----------
    source_lons, source_lats : array-like, shape (N,)
    grid_lons : shape (W,)   — ascending
    grid_lats : shape (H,)   — descending
    power     : IDW exponent (default 2)
    k         : number of nearest neighbours used per grid cell

    Returns
    -------
    weights : np.ndarray, shape (H*W, N)
    """
    source_lons = np.asarray(source_lons, dtype=np.float64)
    source_lats = np.asarray(source_lats, dtype=np.float64)
    H = len(grid_lats)
    W = len(grid_lons)
    N = len(source_lons)

    lon_2d, lat_2d = np.meshgrid(grid_lons, grid_lats)
    query_pts = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])  # [H*W, 2]
    source_pts = np.column_stack([source_lons, source_lats])       # [N, 2]

    k_actual = min(k, N)
    if k_actual == 0:
        return np.zeros((H * W, N), dtype=np.float32)

    tree = cKDTree(source_pts)
    dists, idxs = tree.query(query_pts, k=k_actual)

    if dists.ndim == 1:                   # k=1 edge case
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    exact_mask = dists == 0               # [H*W, k]
    any_exact = exact_mask.any(axis=1)   # [H*W]

    # IDW weights: 1/d^power, zero for exact hits (overridden below)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(exact_mask, 0.0, 1.0 / (dists ** power))

    # For cells where a source sits exactly on the grid point, copy it directly
    exact_rows = np.where(any_exact)[0]
    for row in exact_rows:
        first_exact = np.where(exact_mask[row])[0][0]
        w[row] = 0.0
        w[row, first_exact] = 1.0

    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.maximum(w_sum, 1e-12)

    # Scatter sparse k-NN weights into a dense [H*W, N] matrix
    weights = np.zeros((H * W, N), dtype=np.float32)
    rows = np.repeat(np.arange(H * W), k_actual)
    cols = idxs.ravel()
    vals = w.ravel()
    np.add.at(weights, (rows, cols), vals)

    return weights  # [H*W, N]


# ---------------------------------------------------------------------------
# Apply precomputed weights to a time series
# ---------------------------------------------------------------------------

def apply_idw_weights(weights, values, H, W):
    """Multiply precomputed IDW weights by observation values.

    Parameters
    ----------
    weights : np.ndarray, shape (H*W, N)  — from precompute_idw_weights
    values  : np.ndarray, shape (T, N)    — NaN-free (fill before calling)
    H, W    : grid dimensions

    Returns
    -------
    grids : np.ndarray, shape (T, H, W), float32
    """
    values = np.asarray(values, dtype=np.float32)
    # [T, H*W] = [T, N] @ [N, H*W]
    flat = values @ weights.T
    return flat.reshape(len(values), H, W)


# ---------------------------------------------------------------------------
# Bilinear sampling from grid at scattered point locations (NumPy, no grad)
# ---------------------------------------------------------------------------

def grid_sample_at_points(grid, point_lons, point_lats, grid_lons, grid_lats):
    """Bilinear sample from a single (H, W) grid at scattered point positions.

    Parameters
    ----------
    grid       : np.ndarray, shape (H, W)
    point_lons : shape (N,)
    point_lats : shape (N,)
    grid_lons  : shape (W,)   ascending
    grid_lats  : shape (H,)   descending

    Returns
    -------
    values : np.ndarray, shape (N,), float32
    """
    H, W = grid.shape
    dx = grid_lons[1] - grid_lons[0]
    dy = grid_lats[0] - grid_lats[1]     # positive: lats decrease top→bottom

    xi = (np.asarray(point_lons) - grid_lons[0]) / dx
    yi = (grid_lats[0] - np.asarray(point_lats)) / dy

    xi = np.clip(xi, 0, W - 1)
    yi = np.clip(yi, 0, H - 1)

    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    wx = xi - x0
    wy = yi - y0

    return (
        (1 - wx) * (1 - wy) * grid[y0, x0]
        + wx       * (1 - wy) * grid[y0, x1]
        + (1 - wx) * wy       * grid[y1, x0]
        + wx       * wy       * grid[y1, x1]
    ).astype(np.float32)
