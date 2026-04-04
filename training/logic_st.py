"""
Training, validation and test logic for the spatio-temporal HGNN model
(GNNInductiveHeteroST).

Validity-aware loss
-------------------
The heterodata is built on a dense 15-min grid where missing timesteps are
zero-padded.  The raingauge validity flag (feature index 1) is 1 for real
sensor readings and 0 for imputed zeros.

At every forward pass the loss is computed ONLY on nodes where the validity
flag at the current timestep is 1.  Nodes with validity = 0 at time t have
no real ground truth, so training on them would be misleading.

Masking convention (same as logic_hetero.py)
--------------------------------------------
Indices 0.._DATA_FEATURE_DIM-1 are zeroed for the leave-one-out target:
  index 0 = rainfall value
  index 1 = validity flag

Setting both to 0 makes the masked node indistinguishable from a "genuinely
missing" sensor, which is the intended leave-one-out signal.  LPE columns
(indices >= 2) are never zeroed.
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

# Feature indices 0.._DATA_FEATURE_DIM-1 are zeroed during leave-one-out masking.
_DATA_FEATURE_DIM = 2

# Validity flag is at feature index 1 in the raingauge feature vector.
_VALIDITY_IDX = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_context_dict(batch):
    """Return {node_type: x_context} for all node types that carry it."""
    return {
        ntype: batch[ntype].x_context
        for ntype in batch.node_types
        if hasattr(batch[ntype], "x_context")
    }


def _valid_node_indices(x, indices):
    """
    Given global node indices ``indices`` (shape [B]) and the current-timestep
    feature matrix ``x`` [B*N, F], return the subset of ``indices`` where the
    validity flag (feature index 1) is 1.

    This filters out nodes whose current-timestep reading is imputed (zero-pad).
    """
    return indices[x[indices, _VALIDITY_IDX] > 0]


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

    For each node position we:
      1. Zero its current-timestep data features (value + validity flag).
      2. Keep the context window completely unmasked.
      3. Only compute loss for instances where the ORIGINAL validity flag
         was 1 (i.e., the node had a real reading at time t).

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

        x = batch["raingauge"].x   # [B*N, F]  — current timestep
        y = batch["raingauge"].y   # [B*N, 1]

        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = {
            et: batch[et].edge_attr
            for et in batch.edge_types
            if hasattr(batch[et], "edge_attr")
        }
        x_context_dict = _extract_context_dict(batch)

        num_graphs = batch["raingauge"].ptr.size(0) - 1
        num_nodes  = x.shape[0] // num_graphs

        # First pass: find valid node positions without running the model.
        # Knowing the count upfront lets us scale each per-node loss before
        # calling backward(), so gradients accumulate to the same value as a
        # single averaged backward() — but only one computation graph lives
        # in memory at a time instead of all num_nodes simultaneously.
        valid_positions = []
        for node_pos in range(num_nodes):
            indices   = torch.arange(num_graphs, device=device) * num_nodes + node_pos
            valid_idx = _valid_node_indices(x, indices)
            if valid_idx.numel() > 0:
                valid_positions.append((node_pos, indices, valid_idx))

        if not valid_positions:
            continue  # entire batch is imputed — skip gradient step

        scale          = 1.0 / len(valid_positions)
        batch_loss_val = 0.0

        # Second pass: one forward + backward per valid node position.
        # Each backward frees its computation graph immediately, keeping
        # peak GPU memory proportional to a single forward pass.
        for node_pos, indices, valid_idx in valid_positions:
            x_masked = x.clone()
            x_masked[indices, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {ntype: batch[ntype].x for ntype in batch.node_types}
            x_dict["raingauge"] = x_masked

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            if weighted_loss_alpha > 0.0:
                loss = weighted_mse(
                    out["raingauge"][valid_idx], y[valid_idx],
                    alpha=weighted_loss_alpha,
                )
            else:
                loss = F.mse_loss(out["raingauge"][valid_idx], y[valid_idx])

            (loss * scale).backward()
            batch_loss_val += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss = batch_loss_val / len(valid_positions)
        epoch_losses.append(avg_loss)
        charge_bar.set_postfix({"loss": avg_loss})

    return float(np.mean(epoch_losses)) if epoch_losses else float("nan")


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
    Validation loop — evaluates on split nodes that have real data at time t.

    Returns
    -------
    float  Mean validation loss.
    """
    model.eval()
    epoch_losses = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="validation"):
            batch = batch.to(device)

            x        = batch["raingauge"].x
            y        = batch["raingauge"].y
            val_mask = batch["raingauge"].mask

            edge_index_dict = batch.edge_index_dict
            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if hasattr(batch[et], "edge_attr")
            }
            x_context_dict = _extract_context_dict(batch)

            # Only evaluate on split nodes that have real data at this timestep
            valid_at_t = x[:, _VALIDITY_IDX] > 0
            eval_mask  = val_mask & valid_at_t

            if eval_mask.sum() == 0:
                continue

            x_masked = x.clone()
            x_masked[val_mask, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {ntype: batch[ntype].x for ntype in batch.node_types}
            x_dict["raingauge"] = x_masked

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            if weighted_loss_alpha > 0.0:
                loss = weighted_mse(
                    out["raingauge"][eval_mask], y[eval_mask],
                    alpha=weighted_loss_alpha,
                )
            else:
                loss = F.mse_loss(out["raingauge"][eval_mask], y[eval_mask])

            epoch_losses.append(loss.item())

    return float(np.mean(epoch_losses)) if epoch_losses else float("nan")


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
    Test loop — same outputs as test_model in logic_hetero.py so experiments
    are directly comparable.

    Returns
    -------
    dict  Keys: rmse, mae, pearson_r, timestep_rmse,
                precision, recall, f1, threshold, per_station_metrics.
    """
    model.eval()

    all_preds       = []
    all_targets     = []
    all_station_ids = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="testing"):
            batch = batch.to(device)

            x    = batch["raingauge"].x
            y    = batch["raingauge"].y
            mask = batch["raingauge"].mask

            edge_index_dict = batch.edge_index_dict
            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if hasattr(batch[et], "edge_attr")
            }
            x_context_dict = _extract_context_dict(batch)

            num_graphs = batch["raingauge"].ptr.size(0) - 1
            num_nodes  = x.shape[0] // num_graphs

            # Only evaluate on test-split nodes that have real data at time t
            valid_at_t = x[:, _VALIDITY_IDX] > 0
            eval_mask  = mask & valid_at_t

            if eval_mask.sum() == 0:
                continue

            x_masked = x.clone()
            x_masked[mask, :_DATA_FEATURE_DIM] = 0.0

            x_dict = {ntype: batch[ntype].x for ntype in batch.node_types}
            x_dict["raingauge"] = x_masked

            out = model(x_dict, x_context_dict, edge_index_dict, edge_attr_dict)

            all_preds.append(out["raingauge"][eval_mask].detach().cpu())
            all_targets.append(y[eval_mask].detach().cpu())
            all_station_ids.append(
                (eval_mask.nonzero(as_tuple=False).squeeze(-1) % num_nodes).cpu()
            )

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    all_preds       = torch.cat(all_preds, dim=0)
    all_targets     = torch.cat(all_targets, dim=0)
    all_station_ids = torch.cat(all_station_ids, dim=0)

    preds_np       = all_preds.numpy().flatten()
    targets_np     = all_targets.numpy().flatten()
    station_ids_np = all_station_ids.numpy().flatten()

    print("Prediction shape:", all_preds.shape)
    print("Target shape:    ", all_targets.shape)
    print("Unique stations: ", np.unique(station_ids_np).shape[0])

    # -----------------------------------------------------------------------
    # Global regression metrics
    # -----------------------------------------------------------------------
    valid_mask   = (~np.isnan(preds_np)) & (~np.isnan(targets_np))
    pearson_r, _ = pearsonr(targets_np[valid_mask], preds_np[valid_mask])
    rmse         = float(torch.sqrt(((all_preds - all_targets) ** 2).mean()).item())
    mae          = compute_mae(preds_np, targets_np)
    print(f"Pearson r: {pearson_r:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    # -----------------------------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------------------------
    global_cls     = compute_binary_classification_metrics(
        preds_np, targets_np, threshold=rain_threshold
    )
    global_metrics = {"mae": mae, **global_cls}
    print_metrics_summary(global_metrics)

    # -----------------------------------------------------------------------
    # Per-timestep RMSE
    # -----------------------------------------------------------------------
    n_test_stations = int(np.unique(station_ids_np).shape[0])
    try:
        ts_preds      = all_preds.reshape(-1, n_test_stations)
        ts_targets    = all_targets.reshape(-1, n_test_stations)
        timestep_rmse = float(
            torch.sqrt(((ts_preds - ts_targets) ** 2).mean(dim=1)).mean().item()
        )
    except Exception:
        timestep_rmse = rmse  # fallback if reshape fails

    print(f"Timestep RMSE: {timestep_rmse:.4f}")

    # -----------------------------------------------------------------------
    # Per-station metrics + CSV
    # -----------------------------------------------------------------------
    per_station = compute_per_station_metrics(
        preds_np, targets_np, station_ids_np, threshold=rain_threshold
    )

    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)

    rows = [
        {
            "station_id":  sid,
            "mae":         m["mae"],
            "rmse":        m["rmse"],
            "bias":        m["bias"],
            "pearson_r":   m["pearson_r"],
            "precision":   m["precision"],
            "recall":      m["recall"],
            "f1":          m["f1"],
            "support_pos": m["support_pos"],
            "support_neg": m["support_neg"],
        }
        for sid, m in sorted(per_station.items())
    ]
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
    save_dir = f"{exp_dir}/per_station_plots_f{fold}"
    os.makedirs(save_dir, exist_ok=True)

    for sid in np.unique(station_ids_np).tolist():
        sid_mask    = station_ids_np == sid
        preds_sid   = preds_np[sid_mask]
        targets_sid = targets_np[sid_mask]
        if len(preds_sid) < 5:
            continue
        station_m = per_station.get(int(sid), None)

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
        plt.savefig(f"{save_dir}/station_{int(sid)}_scatter.png", dpi=250)
        plt.close()

        plt.figure(figsize=(15, 6))
        plt.plot(targets_sid, label="Actual")
        plt.plot(preds_sid,   label="Predicted")
        plt.axhline(
            y=rain_threshold, color="gray", linestyle=":", alpha=0.5,
            label=f"Threshold ({rain_threshold} mm)",
        )
        plt.title(f"Station {sid} — Time Series")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/station_{int(sid)}_timeseries.png", dpi=250)
        plt.close()

    print(f"Saved per-station plots → {save_dir}")

    return {
        "rmse":                rmse,
        "mae":                 mae,
        "pearson_r":           pearson_r,
        "timestep_rmse":       timestep_rmse,
        "precision":           global_cls["precision"],
        "recall":              global_cls["recall"],
        "f1":                  global_cls["f1"],
        "threshold":           rain_threshold,
        "per_station_metrics": per_station,
    }
