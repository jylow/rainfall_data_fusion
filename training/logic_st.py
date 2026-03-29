"""
Training, validation and test logic for the spatio-temporal HGNN model
(GNNInductiveHeteroST).

The key difference from logic_hetero.py is that every forward call now
receives both:
  * x_dict          — current-timestep features (with leave-one-out masking)
  * x_context_dict  — context-window features  (never masked)

Masking convention (same as logic_hetero.py)
--------------------------------------------
Features at indices 0.._DATA_FEATURE_DIM-1 are zeroed:
  index 0 = rainfall value
  index 1 = validity flag  (1 = real reading, 0 = NaN/missing)

Setting both to 0 makes the masked node look like a "genuinely missing"
sensor, which is the intended leave-one-out signal.  A real dry reading
(value=0, validity=1) remains distinguishable from a masked node
(value=0, validity=0).  LPE columns (indices >= 2) are never zeroed.

All metric computation / plotting code is re-used from logic_hetero.py.
"""

import torch
import tqdm
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from training.logic_hetero import (
    weighted_mse,
    compute_mae,
    compute_binary_classification_metrics,
    compute_per_station_metrics,
    print_metrics_summary,
)

# Indices 0.._DATA_FEATURE_DIM-1 are zeroed during masking (value + validity).
# LPE columns start at _DATA_FEATURE_DIM and are preserved.
_DATA_FEATURE_DIM = 2


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extract_context_dict(batch):
    """Return {node_type: x_context} from a batched HeteroData object."""
    return {
        ntype: batch[ntype].x_context
        for ntype in batch.node_types
        if hasattr(batch[ntype], "x_context")
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch_st(
    model,
    dataloader,
    optimizer,
    device,
    weighted_loss_alpha: float = 0.0,
):
    """
    Leave-one-out training epoch for the spatio-temporal model.

    For every node position we:
      1. Zero its current-timestep data features (rainfall value + validity flag).
      2. Keep the context window completely unmasked — the target node's own
         history is intentionally visible to the LSTM (no leakage because we
         are predicting the *current* value, not a past one).
      3. Forward pass → compute loss only on the masked node indices.

    Returns
    -------
    float  Mean batch loss over the epoch.
    """
    model.train()
    epoch_losses = []
    charge_bar = tqdm.tqdm(dataloader, desc="training")

    for batch in charge_bar:
        optimizer.zero_grad()
        batch = batch.to(device)

        x = batch["raingauge"].x   # [B*N, F]
        y = batch["raingauge"].y   # [B*N, 1]

        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = {
            et: batch[et].edge_attr
            for et in batch.edge_types
            if hasattr(batch[et], "edge_attr")
        }
        # Context window — extracted once per batch, never modified
        x_context_dict = _extract_context_dict(batch)

        num_graphs = batch["raingauge"].ptr.size(0) - 1
        num_nodes = x.shape[0] // num_graphs

        batch_loss = torch.tensor(0.0, device=device)

        for node_pos in range(num_nodes):
            # Global indices of this node across every graph in the batch
            indices = (
                torch.arange(num_graphs, device=device) * num_nodes + node_pos
            )

            # Mask current-timestep data features; LPE columns are untouched
            x_masked = x.clone()
            x_masked[indices, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {ntype: batch[ntype].x for ntype in batch.node_types}
            x_dict["raingauge"] = x_masked

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            if weighted_loss_alpha > 0.0:
                loss = weighted_mse(
                    out["raingauge"][indices], y[indices], alpha=weighted_loss_alpha
                )
            else:
                loss = F.mse_loss(out["raingauge"][indices], y[indices])

            batch_loss = batch_loss + loss

        batch_loss = batch_loss / num_nodes
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses.append(batch_loss.item())
        charge_bar.set_postfix({"loss": batch_loss.item()})

    return float(np.mean(epoch_losses))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_st(
    model,
    dataloader,
    device,
    weighted_loss_alpha: float = 0.0,
):
    """
    Validation loop for the ST model.

    All validation-split nodes (identified by ``batch['raingauge'].mask``)
    are masked simultaneously, then the loss is computed only on those nodes.

    Returns
    -------
    float  Mean validation loss.
    """
    model.eval()
    epoch_losses = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="validation"):
            batch = batch.to(device)

            x = batch["raingauge"].x
            y = batch["raingauge"].y
            val_mask = batch["raingauge"].mask

            edge_index_dict = batch.edge_index_dict
            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if hasattr(batch[et], "edge_attr")
            }
            x_context_dict = _extract_context_dict(batch)

            # Mask the validation nodes at the current timestep only
            x_masked = x.clone()
            x_masked[val_mask, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {ntype: batch[ntype].x for ntype in batch.node_types}
            x_dict["raingauge"] = x_masked

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            if weighted_loss_alpha > 0.0:
                loss = weighted_mse(
                    out["raingauge"][val_mask], y[val_mask], alpha=weighted_loss_alpha
                )
            else:
                loss = F.mse_loss(out["raingauge"][val_mask], y[val_mask])

            epoch_losses.append(loss.item())

    return float(np.mean(epoch_losses))


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test_model_st(
    model,
    mapping_df,
    dataloader,
    device,
    fold: int = 0,
    experiment_name: str = "test",
    rain_threshold: float = 0.5,
):
    """
    Test loop for the ST model.

    Mirrors ``test_model`` in logic_hetero.py and produces the same CSV and
    plot outputs so ST and spatial-only experiments are directly comparable.

    Returns
    -------
    dict  Keys: rmse, mae, pearson_r, timestep_rmse,
                precision, recall, f1, threshold, per_station_metrics.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_station_ids = []
    epoch_losses = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="testing"):
            batch = batch.to(device)

            x = batch["raingauge"].x
            y = batch["raingauge"].y
            mask = batch["raingauge"].mask

            edge_index_dict = batch.edge_index_dict
            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if hasattr(batch[et], "edge_attr")
            }
            x_context_dict = _extract_context_dict(batch)

            num_graphs = batch["raingauge"].ptr.size(0) - 1
            num_nodes = x.shape[0] // num_graphs

            assert mask.shape[0] == x.shape[0], "Mask/x shape mismatch"

            x_masked = x.clone()
            x_masked[mask, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {ntype: batch[ntype].x for ntype in batch.node_types}
            x_dict["raingauge"] = x_masked

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            loss = F.mse_loss(out["raingauge"][mask], y[mask])
            epoch_losses.append(loss.item())

            all_preds.append(out["raingauge"][mask].detach().cpu())
            all_targets.append(y[mask].detach().cpu())
            all_station_ids.append(
                (mask.nonzero(as_tuple=False).squeeze() % num_nodes).cpu()
            )

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_station_ids = torch.cat(all_station_ids, dim=0)

    print("Prediction shape:", all_preds.shape)
    print("Target shape:    ", all_targets.shape)
    print("Unique stations: ", all_station_ids.unique().shape[0])

    preds_np = all_preds.numpy().flatten()
    targets_np = all_targets.numpy().flatten()
    station_ids_np = all_station_ids.numpy().flatten()

    # -----------------------------------------------------------------------
    # Global regression metrics
    # -----------------------------------------------------------------------
    valid = (~np.isnan(preds_np)) & (~np.isnan(targets_np))
    pearson_r, _ = pearsonr(targets_np[valid], preds_np[valid])
    rmse = torch.sqrt(((all_preds - all_targets) ** 2).mean()).item()
    mae = compute_mae(preds_np, targets_np)

    print(f"Pearson r: {pearson_r:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    # -----------------------------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------------------------
    global_cls = compute_binary_classification_metrics(
        preds_np, targets_np, threshold=rain_threshold
    )
    global_metrics = {"mae": mae, **global_cls}
    print_metrics_summary(global_metrics)

    # -----------------------------------------------------------------------
    # Per-timestep RMSE
    # -----------------------------------------------------------------------
    test_station_count = int(all_station_ids.unique().shape[0])
    ts_preds = all_preds.reshape(-1, test_station_count)
    ts_targets = all_targets.reshape(-1, test_station_count)
    timestep_rmse = torch.sqrt(
        ((ts_preds - ts_targets) ** 2).mean(dim=1)
    ).mean().item()
    print(f"Timestep RMSE: {timestep_rmse:.4f}")

    # -----------------------------------------------------------------------
    # Per-station metrics + CSV
    # -----------------------------------------------------------------------
    per_station = compute_per_station_metrics(
        preds_np, targets_np, station_ids_np, threshold=rain_threshold
    )

    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)

    rows = []
    for sid in sorted(per_station.keys()):
        m = per_station[sid]
        rows.append(
            {
                "station_id": sid,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias": m["bias"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "support_pos": m["support_pos"],
                "support_neg": m["support_neg"],
            }
        )
    csv_path = f"{exp_dir}/per_station_metrics_f{fold}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved per-station metrics → {csv_path}")

    print_metrics_summary(global_metrics, per_station)

    # -----------------------------------------------------------------------
    # Global scatter plot
    # -----------------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_np, preds_np, alpha=0.5)
    max_v = max(float(np.nanmax(preds_np)), float(np.nanmax(targets_np)), 1e-6)
    plt.plot([0, max_v], [0, max_v], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Test Set Performance (ST Model)")
    plt.grid(True)
    plt.text(
        0.05, 0.95,
        f"Pearson r = {pearson_r:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"Timestep RMSE = {timestep_rmse:.3f}\n"
        f"--- threshold = {rain_threshold} mm ---\n"
        f"Precision = {global_cls['precision']:.3f}\n"
        f"Recall    = {global_cls['recall']:.3f}\n"
        f"F1        = {global_cls['f1']:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
    )
    plt.savefig(f"{exp_dir}/test_scatter_plot_{fold}.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------------
    # Per-station scatter + time-series plots
    # -----------------------------------------------------------------------
    unique_stations = all_station_ids.unique().tolist()
    save_dir = f"{exp_dir}/per_station_plots_f{fold}"
    os.makedirs(save_dir, exist_ok=True)

    for sid in unique_stations:
        sid_mask = station_ids_np == sid
        preds_sid = preds_np[sid_mask]
        targets_sid = targets_np[sid_mask]

        if len(preds_sid) < 5:
            continue

        station_m = per_station.get(int(sid), None)

        # Scatter
        plt.figure(figsize=(7, 7))
        plt.scatter(targets_sid, preds_sid, alpha=0.6)
        max_val = max(float(preds_sid.max()), float(targets_sid.max()), 1e-6)
        plt.plot([0, max_val], [0, max_val], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Station {sid} — Actual vs Predicted")
        plt.grid(True)
        if station_m:
            plt.text(
                0.05, 0.95,
                f"MAE  = {station_m['mae']:.3f}\n"
                f"F1   = {station_m['f1']:.3f}\n"
                f"Prec = {station_m['precision']:.3f}\n"
                f"Rec  = {station_m['recall']:.3f}",
                transform=plt.gca().transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
            )
        plt.savefig(f"{save_dir}/station_{sid}_scatter.png", dpi=250)
        plt.close()

        # Time series
        plt.figure(figsize=(15, 6))
        plt.plot(targets_sid, label="Actual")
        plt.plot(preds_sid, label="Predicted")
        plt.axhline(
            y=rain_threshold, color="gray", linestyle=":", alpha=0.5,
            label=f"Threshold ({rain_threshold} mm)",
        )
        plt.title(f"Station {sid} — Time Series")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/station_{sid}_timeseries.png", dpi=250)
        plt.close()

    print(f"Saved per-station plots → {save_dir}")

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "timestep_rmse": timestep_rmse,
        "precision": global_cls["precision"],
        "recall": global_cls["recall"],
        "f1": global_cls["f1"],
        "threshold": rain_threshold,
        "per_station_metrics": per_station,
    }
