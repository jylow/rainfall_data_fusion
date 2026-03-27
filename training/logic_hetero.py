import torch
import tqdm
import numpy as np
import torch.nn.functional as F
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
)
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.utils import read_config

import os

config = read_config("config.yaml")

# Number of data features per raingauge node (rainfall value + validity flag).
# LPE columns start at index _DATA_FEATURE_DIM and must NOT be zeroed during masking.
_DATA_FEATURE_DIM = 2

def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    scheduler=None,
    weighted_loss_alpha: float = 0.0,
):
    """
    Corrected training loop with gradient debugging.
    """
    model.train()
    epoch_losses = []
    charge_bar = tqdm.tqdm(dataloader, desc="training")

    for batch_idx, batch in enumerate(charge_bar):

        optimizer.zero_grad()

        # PyG Batch object - move to device
        batch = batch.to(device)

        # Extract from PyG Batch format
        x = batch['raingauge'].x  # [B*N, F]
        y = batch['raingauge'].y  # [B*N, Tgt]

        edge_index_dict = batch.edge_index_dict
        num_graphs = batch['raingauge'].ptr.size(0) - 1
        num_nodes = x.shape[0] // num_graphs

        edge_attr_dict = {
            edge_type: batch[edge_type].edge_attr
            for edge_type in batch.edge_types
            if hasattr(batch[edge_type], 'edge_attr')
        }

        batch_loss = torch.tensor(0.0, device=device)
        for node_pos in range(num_nodes):
            x_masked = x.clone()
            indices_to_mask = torch.arange(num_graphs, device=device) * x.shape[0] // num_graphs + node_pos
            x_masked[indices_to_mask, :_DATA_FEATURE_DIM] = 0  # zero data features only; preserve LPE

            x_dict = {}
            for nodetype in batch.node_types:
                x_dict[nodetype] = batch[nodetype].x
            x_dict['raingauge'] = x_masked

            out = model(x_dict, edge_index_dict, edge_attr_dict)

            # Compute loss ONLY on masked node
            if weighted_loss_alpha > 0.0:
                loss = weighted_mse(out['raingauge'][indices_to_mask], y[indices_to_mask], alpha=weighted_loss_alpha)
            else:
                loss = F.mse_loss(out['raingauge'][indices_to_mask], y[indices_to_mask])
            batch_loss += loss


        batch_loss = batch_loss / num_nodes
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        epoch_losses.append(batch_loss.item())
        charge_bar.set_postfix(
            {
                "loss": batch_loss.item(),
            }
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return float(np.mean(epoch_losses))


def validate(
    model, dataloader, device, weighted_loss_alpha: float = 0.0,
):
    """
    Validation loop for PyG batched graph data (inductive setting).

    Key aspects:
    1. Data comes as PyG Batch objects
    2. Features are [B*N, F], already batched and flattened
    3. Mask is [N] - single mask for one graph, replicated across batch
    4. Computes metrics ONLY on validation nodes (where mask=True)
    5. No gradients computed - eval mode
    """
    model.eval()
    epoch_losses = []

    charge_bar = tqdm.tqdm(dataloader, desc="validation")

    with torch.no_grad():
        for batch in charge_bar:
            # PyG Batch object - move to device
            batch = batch.to(device)

            # Extract from PyG Batch format
            x = batch['raingauge'].x  # [B*N, F] - already batched and flattened
            y = batch['raingauge'].y  # [B*N, Tgt] - already batched and flattened
            val_mask = batch['raingauge'].mask  # [N] - single mask for one graph
            edge_index_dict = batch.edge_index_dict  # [2, E*B] - offset edge indices
            edge_attr_dict = {
                edge_type: batch[edge_type].edge_attr
                for edge_type in batch.edge_types
                if hasattr(batch[edge_type], 'edge_attr')
            }

            x_masked = x.clone()
            x_masked[val_mask, :_DATA_FEATURE_DIM] = 0.0  # zero data features only; preserve LPE

            x_dict = {}
            for nodetype in batch.node_types:
                x_dict[nodetype] = batch[nodetype].x
            x_dict['raingauge'] = x_masked

            # Forward pass
            out = model(x_dict, edge_index_dict, edge_attr_dict)  # [B*N, out_channels]

            if weighted_loss_alpha > 0.0:
                loss = weighted_mse(out['raingauge'][val_mask], y[val_mask], alpha=weighted_loss_alpha)
            else:
                loss = F.mse_loss(out['raingauge'][val_mask], y[val_mask])
            epoch_losses.append(loss.item())

    # Compute metrics
    mean_loss = float(np.mean(epoch_losses))

    return mean_loss

def test_model(
    model,
    mapping_df,
    dataloader,
    device,
    fold=0,
    experiment_name="test",
    rain_threshold=0.5,
):
    """
    Test loop following the SAME structure as validate():
      - PyG batch format
      - x, y shaped [B*N, F]
      - mask shaped [B*N]
      - station_id shaped [B*N]
      - Computes metrics ONLY on test nodes

    Parameters
    ----------
    rain_threshold : float
        Rainfall threshold (mm) used to binarise predictions and targets
        for computing Precision, Recall, and F1.
    """

    model.eval()

    all_preds = []
    all_targets = []
    all_station_ids = []
    epoch_losses = []

    test_bar = tqdm.tqdm(dataloader, desc="Testing")

    with torch.no_grad():
        for batch in test_bar:
            batch = batch.to(device)
            # ----- Extract inputs from batch -----
            x = batch['raingauge'].x
            y = batch['raingauge'].y
            mask = batch['raingauge'].mask
            edge_index = batch.edge_index_dict
            num_graphs = batch['raingauge'].ptr.size(0) - 1
            num_nodes = x.shape[0] // num_graphs

            assert mask.shape[0] == x.shape[0], "Mask and x size mismatch"
            x_masked = x.clone()
            x_masked[mask, :_DATA_FEATURE_DIM] = 0.0  # zero data features only; preserve LPE

            edge_attr_dict = {
                edge_type: batch[edge_type].edge_attr
                for edge_type in batch.edge_types
                if hasattr(batch[edge_type], 'edge_attr')
            }

            x_dict = {}
            for nodetype in batch.node_types:
                x_dict[nodetype] = batch[nodetype].x
            x_dict['raingauge'] = x_masked

            # ----- Model forward -----
            out = model(x_dict, edge_index, edge_attr_dict)

            # ----- Compute test loss -----
            loss = F.mse_loss(out['raingauge'][mask], y[mask])
            epoch_losses.append(loss.item())

            # ----- Collect outputs -----
            all_preds.append(out['raingauge'][mask].detach().cpu())
            all_targets.append(y[mask].detach().cpu())
            all_station_ids.append(
                (mask.nonzero(as_tuple=False).squeeze() % num_nodes).cpu()
            )

            test_bar.set_postfix({"loss": loss.item()})

    # ============================================================
    # === CONCATENATE EVERYTHING
    # ============================================================
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_station_ids = torch.cat(all_station_ids, dim=0)

    print("Final aggregated prediction shape:", all_preds.shape)
    print("Final aggregated target shape:", all_targets.shape)
    print("Final aggregated station_id shape:", all_station_ids.shape)

    unique_stations = all_station_ids.unique().tolist()
    print("Total stations in test set:", len(unique_stations))

    # ============================================================
    # === NUMPY ARRAYS (reused everywhere below)
    # ============================================================
    preds_np = all_preds.numpy().flatten()
    targets_np = all_targets.numpy().flatten()
    station_ids_np = all_station_ids.numpy().flatten()

    # ============================================================
    # === GLOBAL REGRESSION METRICS
    # ============================================================
    valid_mask = (~np.isnan(preds_np)) & (~np.isnan(targets_np))
    pearson_r, _ = pearsonr(targets_np[valid_mask], preds_np[valid_mask])

    mse = ((all_preds - all_targets) ** 2).mean()
    rmse = torch.sqrt(mse).item()
    mae = compute_mae(preds_np, targets_np)

    print(f"Pearson correlation (Test Nodes): {pearson_r:.4f}")
    print(f"Final Test RMSE: {rmse:.4f}")
    print(f"Final Test MAE : {mae:.4f}")

    # ============================================================
    # === GLOBAL CLASSIFICATION METRICS  (NEW)
    # ============================================================
    global_cls = compute_binary_classification_metrics(
        preds_np, targets_np, threshold=rain_threshold
    )
    global_metrics = {"mae": mae, **global_cls}
    print_metrics_summary(global_metrics)

    # ============================================================
    # === TIMESTEP METRICS
    # ============================================================
    # test_station_count derived from already-collected IDs: the test mask is fixed across
    # all timesteps (same graph topology), so every unique local ID appears exactly once
    # per timestep in all_station_ids.
    test_station_count = int(all_station_ids.unique().shape[0])
    timestep_preds = all_preds.reshape(-1, test_station_count)
    timestep_targets = all_targets.reshape(-1, test_station_count)
    per_timestep_RMSE = torch.sqrt(
        ((timestep_preds - timestep_targets) ** 2).mean(dim=1)
    )
    timestep_rmse = per_timestep_RMSE.mean().item()
    print(f"Timestep RMSE: {timestep_rmse:.4f}")

    # ============================================================
    # === PER-STATION METRICS  (NEW)
    # ============================================================
    per_station = compute_per_station_metrics(
        preds_np, targets_np, station_ids_np, threshold=rain_threshold
    )

    # Save per-station metrics to CSV
    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)

    rows = []
    for sid in sorted(per_station.keys()):
        m = per_station[sid]
        rows.append({
            "station_id": sid,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "bias": m["bias"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "support_pos": m["support_pos"],
            "support_neg": m["support_neg"],
        })
    metrics_df = pd.DataFrame(rows)
    csv_path = f"{exp_dir}/per_station_metrics_f{fold}.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved per-station metrics CSV → {csv_path}")

    print_metrics_summary(global_metrics, per_station)

    # ============================================================
    # === GLOBAL SCATTER PLOT  (updated annotations)
    # ============================================================
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_np, preds_np, alpha=0.5)
    max_v = max(np.nanmax(preds_np), np.nanmax(targets_np))
    plt.plot([0, max_v], [0, max_v], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Test Set Performance")
    plt.grid(True)

    text = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"Timestep RMSE = {timestep_rmse:.3f}\n"
        f"--- threshold = {rain_threshold} mm ---\n"
        f"Precision = {global_cls['precision']:.3f}\n"
        f"Recall = {global_cls['recall']:.3f}\n"
        f"F1 = {global_cls['f1']:.3f}"
    )
    plt.text(
        0.05, 0.95, text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
    )
    plt.savefig(f"{exp_dir}/test_scatter_plot_{fold}.png", dpi=300)
    plt.close()

    # ============================================================
    # === PER-STATION PLOTS  (updated with MAE & F1)
    # ============================================================
    save_dir = f"{exp_dir}/per_station_plots_f{fold}"
    os.makedirs(save_dir, exist_ok=True)

    for sid in unique_stations:
        mask_sid = station_ids_np == sid
        preds_sid = preds_np[mask_sid]
        targets_sid = targets_np[mask_sid]

        if len(preds_sid) < 5:
            continue

        station_m = per_station.get(int(sid), None)

        # ----- Scatter -----
        plt.figure(figsize=(7, 7))
        plt.scatter(targets_sid, preds_sid, alpha=0.6)
        max_val = max(preds_sid.max(), targets_sid.max())
        plt.plot([0, max_val], [0, max_val], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Station {sid} — Actual vs Predicted")
        plt.grid(True)

        if station_m:
            ann = (
                f"MAE = {station_m['mae']:.3f}\n"
                f"F1 = {station_m['f1']:.3f}\n"
                f"Prec = {station_m['precision']:.3f}\n"
                f"Rec = {station_m['recall']:.3f}"
            )
            plt.text(
                0.05, 0.95, ann,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
            )

        plt.savefig(f"{save_dir}/station_{sid}_scatter.png", dpi=250)
        plt.close()

        # ----- Time series -----
        plt.figure(figsize=(15, 6))
        plt.plot(targets_sid, label="Actual")
        plt.plot(preds_sid, label="Predicted")

        # Draw threshold line for context
        plt.axhline(
            y=rain_threshold, color="gray", linestyle=":", alpha=0.5,
            label=f"Threshold ({rain_threshold} mm)",
        )

        plt.title(f"Station {sid} — Time Series")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/station_{sid}_timeseries.png", dpi=250)
        plt.close()

    print(f"Saved per-station plots in {save_dir}")

    # ============================================================
    # === RETURN RESULTS
    # ============================================================
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
def weighted_mse(pred, target, alpha: float = 1.0):
    """
    Weighted MSE loss that up-weights high-rainfall timesteps.

    For target=0   → weight = 1.0              (dry reading, baseline)
    For target=10  → weight = 1 + alpha*log(11)  ≈ 1 + 2.4*alpha
    For target=50  → weight = 1 + alpha*log(51)  ≈ 1 + 3.9*alpha

    Use alpha=0.0 to recover plain MSE.
    Use alpha=1.0 (default) for mild upweighting.
    Use alpha=2.0–4.0 for aggressive upweighting of heavy rain events.

    NOTE: Use this as the TRAINING loss, not RMSE directly.
    RMSE divides by itself in the gradient, causing instability when the
    loss is already small.  This function is the numerically stable equivalent.
    """
    weights = 1.0 + alpha * torch.log1p(target.clamp(min=0.0))
    return (weights * (pred - target) ** 2).mean()

def compute_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error between predictions and targets."""
    valid = (~np.isnan(preds)) & (~np.isnan(targets))
    return mean_absolute_error(targets[valid], preds[valid])


def compute_binary_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    pos_label: int = 1,
    zero_division: int = 0,
) -> dict:
    """
    Threshold continuous predictions/targets into binary classes
    (rain >= threshold → 1, else → 0) and compute precision, recall, F1.

    Parameters
    ----------
    preds : np.ndarray      – continuous model predictions
    targets : np.ndarray    – continuous ground-truth values
    threshold : float        – rainfall threshold (mm) for positive class
    pos_label : int          – which class is "positive" (default 1 = rain)
    zero_division : int      – value returned when a metric is undefined

    Returns
    -------
    dict with keys: precision, recall, f1, confusion_matrix, threshold,
                    support_pos, support_neg
    """
    valid = (~np.isnan(preds)) & (~np.isnan(targets))
    preds_v = preds[valid]
    targets_v = targets[valid]

    pred_labels = (preds_v >= threshold).astype(int)
    true_labels = (targets_v >= threshold).astype(int)

    precision = precision_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    recall = recall_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    f1 = f1_score(true_labels, pred_labels, pos_label=pos_label, zero_division=zero_division)
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
        "support_pos": int(true_labels.sum()),
        "support_neg": int((1 - true_labels).sum()),
    }


# ================================================================
# ===  PER-STATION WRAPPER
# ================================================================

def compute_per_station_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    station_ids: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute MAE and binary classification metrics for each station.

    Parameters
    ----------
    preds : np.ndarray        – flat array of predictions  [N,]
    targets : np.ndarray      – flat array of targets      [N,]
    station_ids : np.ndarray  – flat array of station IDs  [N,]
    threshold : float          – rainfall threshold (mm)

    Returns
    -------
    dict mapping station_id → { mae, precision, recall, f1, ... }
    """
    unique_ids = np.unique(station_ids)
    results = {}

    for sid in unique_ids:
        mask = station_ids == sid
        p = preds[mask]
        t = targets[mask]

        if len(p) < 2:
            continue

        mae = compute_mae(p, t)
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        bias = float(np.mean(p - t))          # positive → over-prediction
        cls = compute_binary_classification_metrics(p, t, threshold=threshold)

        results[int(sid)] = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            **cls,
        }

    return results


def print_metrics_summary(global_metrics: dict, per_station: dict = None):
    """Pretty-print global and (optionally) per-station metrics."""
    print("\n" + "=" * 60)
    print("  GLOBAL METRICS")
    print("=" * 60)
    print(f"  MAE            : {global_metrics['mae']:.4f}")
    print(f"  Threshold      : {global_metrics['threshold']}")
    print(f"  Precision      : {global_metrics['precision']:.4f}")
    print(f"  Recall         : {global_metrics['recall']:.4f}")
    print(f"  F1 Score       : {global_metrics['f1']:.4f}")
    print(f"  Support (pos)  : {global_metrics['support_pos']}")
    print(f"  Support (neg)  : {global_metrics['support_neg']}")
    cm = global_metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    if per_station:
        print("\n" + "=" * 60)
        print("  PER-STATION METRICS")
        print("=" * 60)
        header = f"  {'Station':>8s} | {'MAE':>8s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'Pos':>5s} | {'Neg':>5s}"
        print(header)
        print("  " + "-" * len(header))
        for sid in sorted(per_station.keys()):
            m = per_station[sid]
            print(
                f"  {sid:>8d} | {m['mae']:8.4f} | {m['precision']:6.4f} | "
                f"{m['recall']:6.4f} | {m['f1']:6.4f} | {m['support_pos']:5d} | {m['support_neg']:5d}"
            )
    print("=" * 60)
