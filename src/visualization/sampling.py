import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.visualization.main import visualise_singapore_outline, visualise_with_basemap


def create_dual_sampling_visualization(
    results, station_coords, station_ids, cluster_labels, centroids
):
    """Create comprehensive visualization of the dual sampling strategy."""

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Create lookup dictionaries for quick checking
    stat_train_set = set(results["statistical"]["train"])
    ml_train_set = set(results["ml"]["train"])
    ml_val_set = set(results["ml"]["validation"])
    test_set = set(results["test_stations"])

    # Plot 1: Clusters (top-left)
    ax1 = axes[0, 0]
    for cluster_id in range(len(centroids)):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = station_coords[cluster_mask]

        ax1.scatter(
            cluster_coords[:, 0],
            cluster_coords[:, 1],
            c=[colors[cluster_id]],
            s=120,
            alpha=0.6,
            label=f"Cluster {cluster_id}",
            edgecolors="black",
            linewidth=0.5,
        )

    # Plot centroids
    ax1.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        s=400,
        alpha=0.5,
        marker="X",
        edgecolors="darkred",
        linewidth=2,
        label="Centroids",
        zorder=5,
    )

    ax1.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax1.set_title("K-means Clustering Results", fontsize=14, fontweight="bold", pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    visualise_singapore_outline(ax=ax1)
    visualise_with_basemap(ax=ax1)

    # Plot 2: Statistical Method Split (top-right)
    ax2 = axes[0, 1]
    for i, station_id in enumerate(station_ids):
        cluster_id = cluster_labels[i]
        lon, lat = station_coords[i]

        if station_id in test_set:
            marker, size, label_text = "s", 180, "Test"
            edgecolor, linewidth = "red", 2
        else:  # training
            marker, size, label_text = "o", 120, "Train"
            edgecolor, linewidth = "black", 0.5

        ax2.scatter(
            lon,
            lat,
            c=[colors[cluster_id]],
            s=size,
            marker=marker,
            alpha=0.7,
            edgecolors=edgecolor,
            linewidth=linewidth,
        )

    ax2.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Statistical Method Split (90% Train / 10% Test)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    visualise_singapore_outline(ax=ax2)
    visualise_with_basemap(ax=ax2)

    # Custom legend for statistical
    train_patch = mpatches.Patch(
        color="gray", label=f"Training: {len(stat_train_set)} stations"
    )
    test_patch = mpatches.Patch(color="red", label=f"Test: {len(test_set)} stations")
    ax2.legend(handles=[train_patch, test_patch], loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: ML Method Split (bottom-left)
    ax3 = axes[1, 0]
    for i, station_id in enumerate(station_ids):
        cluster_id = cluster_labels[i]
        lon, lat = station_coords[i]

        if station_id in test_set:
            marker, size, label_text = "s", 180, "Test"
            edgecolor, linewidth = "red", 2
        elif station_id in ml_val_set:
            marker, size, label_text = "^", 150, "Validation"
            edgecolor, linewidth = "blue", 2
        else:  # training
            marker, size, label_text = "o", 120, "Train"
            edgecolor, linewidth = "black", 0.5

        ax3.scatter(
            lon,
            lat,
            c=[colors[cluster_id]],
            s=size,
            marker=marker,
            alpha=0.7,
            edgecolors=edgecolor,
            linewidth=linewidth,
        )

    ax3.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax3.set_title(
        "ML Method Split (70% Train / 20% Val / 10% Test)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    visualise_singapore_outline(ax=ax3)
    visualise_with_basemap(ax=ax3)

    # Custom legend for ML
    train_patch = mpatches.Patch(
        color="gray", label=f"Training: {len(ml_train_set)} stations"
    )
    val_patch = mpatches.Patch(
        color="blue", label=f"Validation: {len(ml_val_set)} stations"
    )
    test_patch = mpatches.Patch(
        color="red", label=f"Test: {len(test_set)} stations (shared)"
    )
    ax3.legend(
        handles=[train_patch, val_patch, test_patch], loc="upper left", fontsize=10
    )
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dual_sampling_results.png", dpi=300, bbox_inches="tight")
    print(f"\n{'=' * 70}")
    print("Visualization saved as 'dual_sampling_results.png'")
    print(f"{'=' * 70}")
    plt.show()
