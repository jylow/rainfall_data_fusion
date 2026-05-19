"""
diagnostics.py
==============
Graph connectivity and feature sanity checks for the heterogeneous GNN.

Call `run_all_diagnostics(heterodata, model, mapping_df, cml_coordinates_df)`
to get a full printed report plus an optional edge-map figure.

Individual checks
-----------------
check_edge_types            – list all edge types, directions, and shapes
check_index_bounds          – ensure no edge index exceeds its node count
check_feature_stats         – mean / std / zero-fraction for every node type
check_reverse_edges         – confirm that CML and radar send messages TO gauges
ablation_test               – measure how much each node type changes the output
plot_edge_map               – draw edges geographically on a map
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# 1.  Edge-type overview
# ---------------------------------------------------------------------------

def check_edge_types(heterodata: HeteroData) -> None:
    """
    Print every edge type in the graph: source→destination, number of edges,
    and whether a corresponding reverse type exists.

    KEY CHECK: For each (A → B) edge type there must be a (B → A) edge type
    so that information from B reaches A during message passing.
    """
    print("\n" + "=" * 65)
    print("  EDGE TYPE OVERVIEW")
    print("=" * 65)
    print(f"  {'Edge type':<45} {'#edges':>8}  {'Reverse?':>10}")
    print("  " + "-" * 63)

    edge_types = heterodata.edge_types
    # Build set of (dst, src) strings to check for reverses
    existing = {(s, r, d) for s, r, d in edge_types}

    for src, rel, dst in sorted(edge_types):
        n_edges = heterodata[(src, rel, dst)].edge_index.shape[1]

        # A reverse exists if there is any edge type whose source=dst and dest=src
        reverse_present = any(
            s == dst and d == src for s, r, d in existing if (s, r, d) != (src, rel, dst)
        )

        direction_flag = "✓" if reverse_present or src == dst else "✗ MISSING"
        if src == dst:
            direction_flag = "(self-loop type)"

        print(f"  ({src}, {rel}, {dst})  "
              f"{n_edges:>8,}  {direction_flag:>10}")

    print()
    # Highlight the critical missing-reverse problem
    critical = [
        (src, rel, dst) for src, rel, dst in edge_types
        if src != dst and not any(s == dst and d == src
                                  for s, r, d in existing if (s, r, d) != (src, rel, dst))
    ]
    if critical:
        print("  ⚠️  ONE-DIRECTIONAL EDGES FOUND — these source nodes will NOT")
        print("     contribute to raingauge predictions via message passing:")
        for et in critical:
            print(f"     {et}")
        print()
        print("  Fix: set  is_directed: False  in config.yaml\n")
    else:
        print("  ✓ All cross-modal edge types have corresponding reverse edges.\n")


# ---------------------------------------------------------------------------
# 2.  Index-bounds check
# ---------------------------------------------------------------------------

def check_index_bounds(heterodata: HeteroData) -> None:
    """
    Verify that no edge index references a non-existent node.
    Raises a descriptive AssertionError if a violation is found.
    """
    print("=" * 65)
    print("  EDGE INDEX BOUNDS CHECK")
    print("=" * 65)

    errors = []
    for src_type, rel, dst_type in heterodata.edge_types:
        edge_index = heterodata[(src_type, rel, dst_type)].edge_index  # [2, E]
        n_src = heterodata[src_type].x.shape[0]
        n_dst = heterodata[dst_type].x.shape[0]

        src_max = int(edge_index[0].max())
        dst_max = int(edge_index[1].max())

        src_ok = src_max < n_src
        dst_ok = dst_max < n_dst

        status = "✓" if (src_ok and dst_ok) else "✗"
        print(f"  {status}  ({src_type}, {rel}, {dst_type})")
        if not src_ok:
            msg = (f"     Source index {src_max} >= n_{src_type} nodes ({n_src})")
            print(msg)
            errors.append(msg)
        if not dst_ok:
            msg = (f"     Dest   index {dst_max} >= n_{dst_type} nodes ({n_dst})")
            print(msg)
            errors.append(msg)

    if errors:
        print("\n  ✗ OUT-OF-BOUNDS INDICES DETECTED — graph is malformed.\n")
    else:
        print("\n  ✓ All edge indices are within valid node ranges.\n")


# ---------------------------------------------------------------------------
# 3.  Feature statistics
# ---------------------------------------------------------------------------

def check_feature_stats(heterodata: HeteroData) -> None:
    """
    Print shape, mean, std, and zero-fraction for every node type's feature tensor.
    A very high zero-fraction suggests the data source may be mostly empty
    (e.g. CML NaN filled with 0), which will suppress its gradient signal.
    """
    print("=" * 65)
    print("  NODE FEATURE STATISTICS  (x tensor, over all nodes & timesteps)")
    print("=" * 65)
    print(f"  {'Node type':<14} {'shape':<22} {'mean':>8} {'std':>8} {'%zero':>8}")
    print("  " + "-" * 62)

    for node_type in heterodata.node_types:
        x = heterodata[node_type].x.float()            # [N, T, F]
        shape_str = str(tuple(x.shape))
        mean  = x.mean().item()
        std   = x.std().item()
        pct_zero = (x == 0).float().mean().item() * 100

        flag = ""
        if pct_zero > 80:
            flag = "  ⚠️  mostly zeros — data may be empty / all-NaN"

        print(f"  {node_type:<14} {shape_str:<22} {mean:>8.4f} {std:>8.4f} {pct_zero:>7.1f}%{flag}")

    print()


# ---------------------------------------------------------------------------
# 4.  Ablation test
# ---------------------------------------------------------------------------

def ablation_test(
    model,
    heterodata: HeteroData,
    device: str = 'cpu',
    timestep: int = 0,
) -> dict[str, float]:
    """
    Measure how much each non-raingauge node type contributes to the output
    by zeroing its features and computing the mean absolute change in
    raingauge predictions.

    A near-zero change means that node type has NO influence on the output,
    which usually means the reverse edge (node_type → raingauge) is missing.

    Parameters
    ----------
    model      : trained GNNInductiveHetero
    heterodata : full test HeteroData (not batched; [N, T, F] tensors)
    device     : str
    timestep   : int  which timestep to evaluate on

    Returns
    -------
    dict mapping node_type → mean absolute output change
    """
    print("=" * 65)
    print("  ABLATION TEST  (zero each node type, measure output change)")
    print("=" * 65)
    print("  A change near 0.0 means that node type does NOT reach raingauge.\n")

    model.eval()

    # Build single-timestep snapshot
    def _build_snap(ht: HeteroData) -> dict:
        x_dict = {nt: ht[nt].x[:, timestep, :].to(device) for nt in ht.node_types}
        edge_index_dict = {et: ht[et].edge_index.to(device) for et in ht.edge_types}
        edge_attr_dict = {
            et: ht[et].edge_attr.to(device)
            for et in ht.edge_types
            if hasattr(ht[et], 'edge_attr')
        }
        return x_dict, edge_index_dict, edge_attr_dict

    with torch.no_grad():
        x_base, ei, ea = _build_snap(heterodata)
        out_base = model(x_base, ei, ea)['raingauge'].cpu()

    results = {}
    for node_type in heterodata.node_types:
        if node_type == 'raingauge':
            continue
        with torch.no_grad():
            x_ablated, ei, ea = _build_snap(heterodata)
            x_ablated[node_type] = torch.zeros_like(x_ablated[node_type])
            out_ablated = model(x_ablated, ei, ea)['raingauge'].cpu()

        delta = (out_base - out_ablated).abs().mean().item()
        results[node_type] = delta

        status = "✓" if delta > 1e-4 else "✗ NO EFFECT"
        print(f"  {status}  Zeroing '{node_type}' changes raingauge output by {delta:.6f} (mean abs)")

    print()
    return results


# ---------------------------------------------------------------------------
# 5.  Geographic edge map
# ---------------------------------------------------------------------------

def plot_edge_map(
    heterodata: HeteroData,
    mapping_df: pd.DataFrame,
    cml_coordinates_df: pd.DataFrame | None = None,
    radar_grid_coords: pd.DataFrame | None = None,
    edge_types_to_plot: list | None = None,
    max_edges_per_type: int = 200,
    bounds: dict | None = None,
    save_path: str | None = None,
):
    """
    Draw gauge positions and a sample of cross-modal edges on a 2-D map.

    Parameters
    ----------
    heterodata          : HeteroData  (test split, so all stations are visible)
    mapping_df          : DataFrame   ['longitude', 'latitude', 'id']  for gauges
    cml_coordinates_df  : DataFrame   ['site_a_longitude', 'site_a_latitude',
                                       'site_b_longitude', 'site_b_latitude']
    radar_grid_coords   : DataFrame   ['longitude', 'latitude']  for radar grid points
    edge_types_to_plot  : list of (src, rel, dst) tuples; None = all cross-modal
    max_edges_per_type  : int  sample this many edges to avoid overdrawing
    bounds              : dict  {'left', 'right', 'top', 'bottom'}
    save_path           : str  optional path to save

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    gauge_coords = mapping_df[['longitude', 'latitude']].values
    ax.scatter(
        gauge_coords[:, 0], gauge_coords[:, 1],
        c='red', s=50, zorder=5, label='Rain gauges', edgecolors='black', linewidths=0.5,
    )
    for _, row in mapping_df.iterrows():
        ax.annotate(str(row['id']), (row['longitude'], row['latitude']),
                    fontsize=5, color='darkred', xytext=(2, 2), textcoords='offset points')

    # CML link endpoints
    if cml_coordinates_df is not None:
        cml_even = cml_coordinates_df.iloc[::2]   # one row per link (same as CMLGraph)
        for _, row in cml_even.iterrows():
            ax.plot(
                [row['site_a_longitude'], row['site_b_longitude']],
                [row['site_a_latitude'],  row['site_b_latitude']],
                color='purple', linewidth=0.6, alpha=0.4, zorder=2,
            )
        # midpoints as dots
        mid_lon = (cml_even['site_a_longitude'].values + cml_even['site_b_longitude'].values) / 2
        mid_lat = (cml_even['site_a_latitude'].values  + cml_even['site_b_latitude'].values)  / 2
        ax.scatter(mid_lon, mid_lat, c='purple', s=10, zorder=3, alpha=0.5,
                   label='CML links (midpoints)')

    # Radar grid (subsample to avoid overdrawing)
    if radar_grid_coords is not None:
        step = max(1, len(radar_grid_coords) // 500)
        rc = radar_grid_coords.iloc[::step]
        ax.scatter(rc['longitude'], rc['latitude'],
                   c='grey', s=2, alpha=0.3, zorder=1, label='Radar grid (subsampled)')

    # Cross-modal edges
    colour_map = {
        'radar': 'steelblue',
        'cml':   'darkorange',
    }
    default_colours = iter(['green', 'cyan', 'magenta'])

    for src_type, rel, dst_type in heterodata.edge_types:
        if src_type == dst_type:          # skip gauge-gauge self-edges
            continue
        if edge_types_to_plot and (src_type, rel, dst_type) not in edge_types_to_plot:
            continue

        edge_index = heterodata[(src_type, rel, dst_type)].edge_index  # [2, E]
        n_edges = edge_index.shape[1]
        sample_idx = np.random.choice(n_edges, min(max_edges_per_type, n_edges), replace=False)
        sampled = edge_index[:, sample_idx]

        # Identify which node type is the gauge and which is the auxiliary
        if src_type == 'raingauge':
            gauge_idx = sampled[0].numpy()
            aux_idx   = sampled[1].numpy()
            aux_type  = dst_type
        else:
            gauge_idx = sampled[1].numpy()
            aux_idx   = sampled[0].numpy()
            aux_type  = src_type

        aux_coords = _get_node_coords(aux_type, aux_idx, cml_coordinates_df, radar_grid_coords)
        if aux_coords is None:
            continue

        colour = colour_map.get(aux_type, next(default_colours))
        for gi, (ax_lon, ax_lat) in zip(gauge_idx, aux_coords):
            g_lon, g_lat = float(gauge_coords[gi, 0]), float(gauge_coords[gi, 1])
            ax.plot([g_lon, ax_lon], [g_lat, ax_lat],
                    color=colour, linewidth=0.4, alpha=0.35, zorder=2)

        patch = mpatches.Patch(color=colour,
                               label=f'({src_type} → {dst_type})  {n_edges} edges')
        ax.add_patch(patch)

    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Graph edge map — cross-modal connections')
    if bounds:
        ax.set_xlim(bounds['left'],  bounds['right'])
        ax.set_ylim(bounds['bottom'], bounds['top'])
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, ax


def _get_node_coords(
    node_type: str,
    indices: np.ndarray,
    cml_coordinates_df,
    radar_grid_coords,
) -> list[tuple[float, float]] | None:
    """Return (lon, lat) for each index in *indices* for the given node type."""
    if node_type == 'radar' and radar_grid_coords is not None:
        coords = radar_grid_coords[['longitude', 'latitude']].values
        return [(float(coords[i, 0]), float(coords[i, 1])) for i in indices if i < len(coords)]

    if node_type == 'cml' and cml_coordinates_df is not None:
        # CML graph nodes: even index i → site_a of row i, odd index → site_b of row i-1
        result = []
        for i in indices:
            if i % 2 == 0:
                row_idx = i
                lon = cml_coordinates_df.iloc[row_idx]['site_a_longitude']
                lat = cml_coordinates_df.iloc[row_idx]['site_a_latitude']
            else:
                row_idx = i - 1
                lon = cml_coordinates_df.iloc[row_idx]['site_b_longitude']
                lat = cml_coordinates_df.iloc[row_idx]['site_b_latitude']
            result.append((float(lon), float(lat)))
        return result

    return None


# ---------------------------------------------------------------------------
# 6.  Combined entry point
# ---------------------------------------------------------------------------

def run_all_diagnostics(
    heterodata: HeteroData,
    model,
    mapping_df: pd.DataFrame,
    cml_coordinates_df: pd.DataFrame | None = None,
    radar_grid_coords: pd.DataFrame | None = None,
    bounds: dict | None = None,
    device: str = 'cpu',
    save_dir: str | None = None,
) -> None:
    """
    Run all graph diagnostics and print a full report.

    Parameters
    ----------
    heterodata         : normalised test HeteroData (from get_test_heterodata)
    model              : trained GNNInductiveHetero
    mapping_df         : gauge coordinate DataFrame
    cml_coordinates_df : full (not iloc[::2]) CML coordinates DataFrame
    radar_grid_coords  : radar grid coords from RadarGraph.grid_coords
    bounds             : geography bounds dict for the map plot
    device             : torch device string
    save_dir           : if set, save edge_map figure here
    """
    print("\n" + "=" * 65)
    print("  GRAPH DIAGNOSTICS REPORT")
    print("=" * 65)
    print(f"  Node types : {heterodata.node_types}")
    print(f"  Edge types : {len(heterodata.edge_types)}")
    for node_type in heterodata.node_types:
        n = heterodata[node_type].x.shape[0]
        print(f"    {node_type}: {n} nodes")
    print()

    check_edge_types(heterodata)
    check_index_bounds(heterodata)
    check_feature_stats(heterodata)
    ablation_test(model, heterodata, device=device)

    if save_dir or (cml_coordinates_df is not None or radar_grid_coords is not None):
        import os
        save_path = os.path.join(save_dir, 'graph_edge_map.png') if save_dir else None
        fig, _ = plot_edge_map(
            heterodata, mapping_df,
            cml_coordinates_df=cml_coordinates_df,
            radar_grid_coords=radar_grid_coords,
            bounds=bounds,
            save_path=save_path,
        )
        if save_path:
            print(f"  Edge map saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)
