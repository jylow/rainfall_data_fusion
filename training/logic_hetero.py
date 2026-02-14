import torch
import tqdm
import numpy as np
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import os

def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    verbose=False,
    log_file="training_gnn_new_debug.log",
    random_noise_masking=False,
    scheduler=None,
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
            x_masked[indices_to_mask, :] = 0

            x_dict = {}
            for nodetype in batch.node_types:
                x_dict[nodetype] = batch[nodetype].x
            x_dict['raingauge'] = x_masked
            out = model(x_dict, edge_index_dict, edge_attr_dict)

            # Compute loss ONLY on trainable nodes
            loss = F.mse_loss(out['raingauge'][indices_to_mask], y[indices_to_mask])
            batch_loss += loss


        batch_loss = batch_loss / num_nodes
        if scheduler is not None:
            scheduler.step()
        batch_loss.backward()


        epoch_losses.append(batch_loss.item())
        charge_bar.set_postfix(
            {
                "loss": batch_loss.item(),
                #"grad_norm": total_grad_norm,
            }
        )
        #Step only at the end of each batch
        optimizer.step()

    return float(np.mean(epoch_losses))


def validate(
    model, dataloader, device, verbose=False, log_file="validation_gnn_new_debug.log"
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
    all_preds = []
    all_targets = []

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
            x_masked[val_mask, :] = 0.0

            x_dict = {}
            for nodetype in batch.node_types:
                x_dict[nodetype] = batch[nodetype].x
            x_dict['raingauge'] = x_masked

            # Forward pass
            out = model(x_dict, edge_index_dict, edge_attr_dict)  # [B*N, out_channels]

            loss = F.mse_loss(out['raingauge'][val_mask], y[val_mask])
            epoch_losses.append(loss.item())

            # Store predictions and targets for metric computation
            all_preds.append(out['raingauge'][val_mask].detach().cpu())
            all_targets.append(y[val_mask].detach().cpu())

            charge_bar.set_postfix({"loss": loss.item()})

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)  # [Total_val_nodes, out_channels]
    all_targets = torch.cat(all_targets, dim=0)  # [Total_val_nodes, out_channels]

    # Compute metrics
    mean_loss = float(np.mean(epoch_losses))

    return mean_loss


def test_model(model, mapping_df, dataloader, device, fold=0, experiment_name= "test"):
    """
    Test loop following the SAME structure as validate():
      - PyG batch format
      - x, y shaped [B*N, F]
      - mask shaped [B*N]
      - station_id shaped [B*N]  (added)
      - Computes metrics ONLY on test nodes
    """

    model.eval()

    all_preds = []
    all_targets = []
    all_station_ids = []   # <-- FIXED: collect all station IDs here
    epoch_losses = []
    timestep_RMSE = []

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
            x_masked[mask, :] = 0.0

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
            all_station_ids.append((mask.nonzero(as_tuple=False).squeeze() % num_nodes).cpu())   # <-- FIXED

            test_bar.set_postfix({"loss": loss.item()})


    print(all_station_ids)
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
    # === GLOBAL METRICS
    # ============================================================
    preds_np = all_preds.numpy().flatten()
    targets_np = all_targets.numpy().flatten()

    valid_mask = (~np.isnan(preds_np)) & (~np.isnan(targets_np))
    pearson_r, pearson_p = pearsonr(targets_np[valid_mask], preds_np[valid_mask])

    mse = ((all_preds - all_targets) ** 2).mean()
    rmse = torch.sqrt(mse).item()

    print(f"Pearson correlation (Test Nodes): {pearson_r}")
    print(f"Final Test RMSE: {rmse}")

    # ============================================================
    # === TIMESTEP METRICS
    # ============================================================
    test_station_count = next(iter(dataloader)).mask.sum()
    timestep_preds = all_preds.reshape(-1, test_station_count)
    timestep_targets = all_targets.reshape(-1, test_station_count)
    per_timestep_RMSE = torch.sqrt(((timestep_preds - timestep_targets)**2).mean(dim=1))
    timestep_rmse = per_timestep_RMSE.mean().item()
    print(f"Timestep RMSE: {timestep_rmse}")

    # ============================================================
    # === GLOBAL SCATTER
    # ============================================================
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_np, preds_np, alpha=0.5)
    max_v = max(np.nanmax(preds_np), np.nanmax(targets_np))
    plt.plot([0, max_v], [0, max_v], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Test Set Performance")
    plt.grid(True)
    text = f"Pearson r = {pearson_r:.3f}\nRMSE = {rmse:.3f} \n TimestepRMSE = {timestep_rmse:.3f}"
    plt.text( 0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"), )
    plt.savefig(f"experiments/{experiment_name}/test_scatter_plot_{fold}.png", dpi=300)
    plt.close()

    # ============================================================
    # === PER-STATION PLOTS
    # ============================================================
    save_dir = f"experiments/{experiment_name}/per_station_plots_f{fold}"
    os.makedirs(save_dir, exist_ok=True)

    for sid in unique_stations:
        mask_sid = (all_station_ids == sid)

        preds_sid = all_preds[mask_sid].numpy().flatten()
        targets_sid = all_targets[mask_sid].numpy().flatten()

        if len(preds_sid) < 5:
            continue

        # ----- Scatter -----
        plt.figure(figsize=(7, 7))
        plt.scatter(targets_sid, preds_sid, alpha=0.6)
        max_val = max(preds_sid.max(), targets_sid.max())
        plt.plot([0, max_val], [0, max_val], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Station {sid} — Actual vs Predicted")
        plt.grid(True)
        plt.savefig(f"{save_dir}/station_{sid}_scatter.png", dpi=250)
        plt.close()

        # ----- Time series -----
        plt.figure(figsize=(15, 6))
        plt.plot(targets_sid, label="Actual")
        plt.plot(preds_sid, label="Predicted")
        plt.title(f"Station {sid} — Time Series")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/station_{sid}_timeseries.png", dpi=250)
        plt.close()

    print(f"Saved per-station plots in {save_dir}")

    return rmse

