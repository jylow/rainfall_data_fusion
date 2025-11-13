import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D

from src.raingauge.utils import get_station_coordinate_mappings
from src.sampling.main import stratified_spatial_kfold_dual
from src.visualization.main import visualise_singapore_outline, visualise_with_basemap

def analyze_and_visualize_station_errors(
    fold_results,
    station_dict,
    experiment_name,
    n_clusters=8,
    error_threshold_percentile=75,
    seed=42
):
    """
    Analyze station errors across all folds and overlay on spatial sampling maps.
    
    Parameters
    ----------
    fold_results : list
        Output from stratified_spatial_kfold_dual()
    station_dict : dict
        {station_id: [lat, lon]}
    experiment_name : str
        Name of experiment (for finding prediction files)
    n_clusters : int
        Number of clusters used in spatial sampling
    error_threshold_percentile : float
        Percentile threshold for "high error" stations
    seed : int
        Random seed (used for K-means to get same cluster labels)
        
    Returns
    -------
    all_station_errors : pd.DataFrame
        Combined error statistics for all stations across all folds
    """
    
    # Load all station error files
    all_station_errors = []
    for fold_idx in range(len(fold_results)):
        error_file = f"experiments/{experiment_name}/predictions/station_errors_fold{fold_idx}.csv"
        try:
            fold_errors = pd.read_csv(error_file)
            all_station_errors.append(fold_errors)
        except FileNotFoundError:
            print(f"Warning: Could not find {error_file}")
    
    if not all_station_errors:
        raise ValueError("No station error files found!")
    
    combined_errors = pd.concat(all_station_errors, ignore_index=True)
    
    # Aggregate errors across folds
    station_avg_errors = combined_errors.groupby("station_id").agg({
        "mean_abs_error": "mean",
        "median_abs_error": "mean",
        "max_abs_error": "max",
        "std_abs_error": "mean",
        "bias": "mean",
        "mean_true": "mean",
        "mean_pred": "mean"
    }).reset_index()
    
    # Add coordinates
    station_avg_errors["lat"] = station_avg_errors["station_id"].map(
        lambda sid: station_dict[sid][0]
    )
    station_avg_errors["lon"] = station_avg_errors["station_id"].map(
        lambda sid: station_dict[sid][1]
    )
    
    # Determine error threshold
    error_threshold = np.percentile(station_avg_errors["mean_abs_error"], error_threshold_percentile)
    station_avg_errors["is_high_error"] = station_avg_errors["mean_abs_error"] >= error_threshold
    
    # Get cluster labels
    station_ids = np.array(list(station_dict.keys()))
    station_coords = np.array([[lon, lat] for lat, lon in station_dict.values()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(station_coords)
    
    # Map cluster labels to stations
    cluster_map = dict(zip(station_ids, cluster_labels))
    station_avg_errors["cluster"] = station_avg_errors["station_id"].map(cluster_map)
    
    print("="*80)
    print("STATION ERROR ANALYSIS ACROSS ALL FOLDS")
    print("="*80)
    print(f"Total unique test stations: {len(station_avg_errors)}")
    print(f"High error threshold ({error_threshold_percentile}th percentile): {error_threshold:.3f}")
    print(f"High error stations: {station_avg_errors['is_high_error'].sum()}")
    
    print("\nTop 10 worst performing stations:")
    top10 = station_avg_errors.nlargest(10, "mean_abs_error")
    print(top10[["station_id", "cluster", "mean_abs_error", "bias", "mean_true", "mean_pred"]].to_string(index=False))
    
    # Save aggregated results
    station_avg_errors.to_csv(
        f"experiments/{experiment_name}/predictions/station_errors_aggregated.csv",
        index=False
    )
    
    # Create visualizations for each fold
    for fold_idx, fold_result in enumerate(fold_results):
        create_error_overlay_map(
            fold_result,
            station_avg_errors,
            station_dict,
            cluster_labels,
            kmeans.cluster_centers_,
            fold_idx=fold_idx,
            error_threshold=error_threshold,
            output_path=f"experiments/{experiment_name}/error_map_fold{fold_idx}.png"
        )
    
    # Create aggregate error map across all folds
    create_aggregate_error_map(
        station_avg_errors,
        station_dict,
        cluster_labels,
        kmeans.cluster_centers_,
        error_threshold=error_threshold,
        output_path=f"experiments/{experiment_name}/error_map_aggregate.png"
    )
    
    return station_avg_errors


def create_error_overlay_map(
    fold_result,
    station_avg_errors,
    station_dict,
    cluster_labels,
    centroids,
    fold_idx=0,
    error_threshold=None,
    output_path="error_overlay.png"
):
    """
    Create spatial map with error overlay for a specific fold.
    """
    
    # Get all station coordinates
    station_ids = np.array(list(station_dict.keys()))
    station_coords = np.array([[lon, lat] for lat, lon in station_dict.values()])
    
    # Get test stations for this fold
    test_stations = set(fold_result["test_stations"])
    test_station_errors = station_avg_errors[station_avg_errors["station_id"].isin(test_stations)]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot clusters (background)
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(cluster_labels))))
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = station_coords[cluster_mask]
        ax.scatter(
            cluster_coords[:, 0],
            cluster_coords[:, 1],
            c=[colors[cluster_id]],
            s=80,
            alpha=0.3,
            edgecolors="gray",
            linewidth=0.5,
            label=f"Cluster {cluster_id}"
        )
    
    # Plot centroids
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=300,
        alpha=0.8,
        marker="X",
        edgecolors="white",
        linewidth=2,
        label="Cluster Centers",
        zorder=10
    )
    
    # Plot train stations for this fold
    train_stations = set(fold_result["statistical"]["train"])
    train_coords = np.array([[station_dict[sid][1], station_dict[sid][0]] 
                             for sid in train_stations])
    ax.scatter(
        train_coords[:, 0],
        train_coords[:, 1],
        c="lightgray",
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
        marker="o",
        label="Train Stations",
        zorder=5
    )
    
    # Plot test stations colored by error magnitude
    if len(test_station_errors) > 0:
        scatter = ax.scatter(
            test_station_errors["lon"],
            test_station_errors["lat"],
            c=test_station_errors["mean_abs_error"],
            cmap="YlOrRd",
            s=300,
            alpha=0.9,
            edgecolors="black",
            linewidth=2,
            marker="s",
            vmin=0,
            vmax=test_station_errors["mean_abs_error"].quantile(0.95),
            label="Test Stations",
            zorder=8
        )
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Mean Absolute Error", fontsize=12, fontweight="bold")
        
        # Highlight high-error stations
        if error_threshold is not None:
            high_error = test_station_errors[test_station_errors["mean_abs_error"] >= error_threshold]
            if len(high_error) > 0:
                ax.scatter(
                    high_error["lon"],
                    high_error["lat"],
                    s=600,
                    facecolors="none",
                    edgecolors="red",
                    linewidth=4,
                    label=f"High Error (>{error_threshold:.2f})",
                    zorder=9
                )
                
                # Annotate worst 3 stations
                worst3 = high_error.nlargest(3, "mean_abs_error")
                for _, row in worst3.iterrows():
                    ax.annotate(
                        f"{row['station_id']}\nErr: {row['mean_abs_error']:.2f}\nBias: {row['bias']:+.2f}",
                        xy=(row["lon"], row["lat"]),
                        xytext=(15, 15),
                        textcoords="offset points",
                        fontsize=9,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor="red", linewidth=2),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="red", lw=2),
                        zorder=11
                    )
    
    ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Fold {fold_idx}: Spatial Distribution with Error Overlay\n"
        f"Test Stations: {len(test_stations)} | High Error: {len(test_station_errors[test_station_errors['mean_abs_error'] >= error_threshold]) if error_threshold else 0}",
        fontsize=14,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved error overlay map for fold {fold_idx} to '{output_path}'")


def create_aggregate_error_map(
    station_avg_errors,
    station_dict,
    cluster_labels,
    centroids,
    error_threshold=None,
    output_path="error_map_aggregate.png"
):
    """
    Create aggregate error map showing all test stations from all folds.
    """
    
    station_ids = np.array(list(station_dict.keys()))
    station_coords = np.array([[lon, lat] for lat, lon in station_dict.values()])
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # --- Left plot: Error magnitude ---
    ax1 = axes[0]
    
    # Plot clusters
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(cluster_labels))))
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = station_coords[cluster_mask]
        ax1.scatter(
            cluster_coords[:, 0],
            cluster_coords[:, 1],
            c=[colors[cluster_id]],
            s=60,
            alpha=0.2,
            edgecolors="gray",
            linewidth=0.5
        )
    
    # Plot centroids
    ax1.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=250,
        alpha=0.7,
        marker="X",
        edgecolors="white",
        linewidth=2,
        zorder=10
    )
    
    # Plot test stations colored by error
    scatter1 = ax1.scatter(
        station_avg_errors["lon"],
        station_avg_errors["lat"],
        c=station_avg_errors["mean_abs_error"],
        cmap="YlOrRd",
        s=300,
        alpha=0.9,
        edgecolors="black",
        linewidth=2,
        marker="s",
        vmin=0,
        vmax=station_avg_errors["mean_abs_error"].quantile(0.95),
        zorder=8
    )
    
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
    cbar1.set_label("Mean Absolute Error (Across All Folds)", fontsize=11, fontweight="bold")
    
    # Highlight high-error stations
    if error_threshold is not None:
        high_error = station_avg_errors[station_avg_errors["is_high_error"]]
        ax1.scatter(
            high_error["lon"],
            high_error["lat"],
            s=600,
            facecolors="none",
            edgecolors="red",
            linewidth=4,
            label="High Error (top 25%)",
            zorder=9
        )
    
    ax1.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax1.set_title("Error Magnitude (All Test Stations)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="best", fontsize=10)
    
    # --- Right plot: Error clusters + bias ---
    ax2 = axes[1]
    
    # Plot background clusters
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = station_coords[cluster_mask]
        ax2.scatter(
            cluster_coords[:, 0],
            cluster_coords[:, 1],
            c=[colors[cluster_id]],
            s=60,
            alpha=0.2,
            edgecolors="gray",
            linewidth=0.5
        )
    
    # Categorize errors
    station_avg_errors["error_category"] = pd.cut(
        station_avg_errors["mean_abs_error"],
        bins=[0, 
              station_avg_errors["mean_abs_error"].quantile(0.33),
              station_avg_errors["mean_abs_error"].quantile(0.67),
              station_avg_errors["mean_abs_error"].max()],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )
    
    category_colors = {"Low": "green", "Medium": "orange", "High": "red"}
    for category, color in category_colors.items():
        subset = station_avg_errors[station_avg_errors["error_category"] == category]
        if len(subset) > 0:
            ax2.scatter(
                subset["lon"],
                subset["lat"],
                c=color,
                s=250,
                edgecolors="black",
                linewidth=2,
                alpha=0.8,
                marker="s",
                label=f"{category} Error",
                zorder=8
            )
    
    # Add bias indicators
    for _, row in station_avg_errors.iterrows():
        if abs(row["bias"]) > 1.0:  # Only show significant bias
            marker = "^" if row["bias"] > 0 else "v"
            color_bias = "blue" if row["bias"] > 0 else "purple"
            ax2.scatter(
                row["lon"],
                row["lat"],
                marker=marker,
                s=120,
                c=color_bias,
                alpha=0.7,
                edgecolors="white",
                linewidth=1,
                zorder=9
            )
    
    # Custom legend for bias
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="green", markersize=10, label="Low Error"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="orange", markersize=10, label="Medium Error"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markersize=10, label="High Error"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="blue", markersize=10, label="Over-prediction"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="purple", markersize=10, label="Under-prediction")
    ]
    
    ax2.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax2.set_title("Error Categories & Prediction Bias", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(handles=legend_elements, loc="best", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved aggregate error map to '{output_path}'")

def visualize_single_fold_errors(
    station_error_csv,
    fold_result,
    station_dict,
    n_clusters=8,
    error_threshold_percentile=75,
    seed=42,
    output_path="single_fold_error_map.png",
    visualise_singapore_outline=visualise_singapore_outline,
    visualise_with_basemap=visualise_with_basemap
):
    """
    Visualize station errors for a single fold on spatial sampling map.
    Uses the same plotting style as create_dual_sampling_visualization.
    
    Parameters
    ----------
    station_error_csv : str
        Path to station_errors_fold{X}.csv file
    fold_result : dict
        Single fold result from stratified_spatial_kfold_dual()
        Contains 'test_stations', 'statistical', 'ml', etc.
    station_dict : dict
        {station_id: [lat, lon]} for ALL stations
    n_clusters : int
        Number of clusters used in spatial sampling
    error_threshold_percentile : float
        Percentile threshold for "high error" stations (default: 75)
    seed : int
        Random seed for K-means clustering
    output_path : str
        Path to save the figure
    visualise_singapore_outline : callable, optional
        Function to add Singapore outline to axis
    visualise_with_basemap : callable, optional
        Function to add basemap to axis
        
    Returns
    -------
    station_errors : pd.DataFrame
        Station error statistics with coordinates and cluster info
    """
    
    # Load station errors
    station_errors = pd.read_csv(station_error_csv)
    
    # Add coordinates
    station_errors["lat"] = station_errors["station_id"].map(
        lambda sid: station_dict[sid][0]
    )
    station_errors["lon"] = station_errors["station_id"].map(
        lambda sid: station_dict[sid][1]
    )
    
    # Get all station data for clustering
    station_ids = np.array(list(station_dict.keys()))
    station_coords = np.array([[lon, lat] for lat, lon in station_dict.values()])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(station_coords)
    centroids = kmeans.cluster_centers_
    
    # Map cluster labels to stations
    cluster_map = dict(zip(station_ids, cluster_labels))
    station_errors["cluster"] = station_errors["station_id"].map(cluster_map)
    
    # Determine error threshold
    error_threshold = np.percentile(station_errors["mean_abs_error"], error_threshold_percentile)
    station_errors["is_high_error"] = station_errors["mean_abs_error"] >= error_threshold
    
    # Print statistics
    print("="*80)
    print(f"STATION ERROR ANALYSIS - Fold {station_errors['fold'].iloc[0]}")
    print("="*80)
    print(f"Total test stations: {len(station_errors)}")
    print(f"Error threshold ({error_threshold_percentile}th percentile): {error_threshold:.3f}")
    print(f"High error stations: {station_errors['is_high_error'].sum()}")
    print(f"\nError statistics:")
    print(f"  Mean: {station_errors['mean_abs_error'].mean():.3f}")
    print(f"  Median: {station_errors['mean_abs_error'].median():.3f}")
    print(f"  Max: {station_errors['mean_abs_error'].max():.3f}")
    print(f"  Std: {station_errors['mean_abs_error'].std():.3f}")
    
    print("\nTop 5 worst performing stations:")
    top5 = station_errors.nlargest(5, "mean_abs_error")
    print(top5[["station_id", "cluster", "mean_abs_error", "bias", "mean_true", "mean_pred"]].to_string(index=False))
    
    # Get train stations for this fold
    train_stations = set(fold_result["statistical"]["train"])
    test_stations = set(fold_result["test_stations"])
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # === LEFT PLOT: Error Magnitude with Spatial Context ===
    ax1 = axes[0]
    
    # Plot clusters (all stations background)
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(cluster_labels))))
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = station_coords[cluster_mask]
        ax1.scatter(
            cluster_coords[:, 0],
            cluster_coords[:, 1],
            c=[colors[cluster_id]],
            s=80,
            alpha=0.25,
            edgecolors="gray",
            linewidth=0.5,
            label=f"Cluster {cluster_id}"
        )
    
    # Plot centroids
    ax1.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=300,
        alpha=0.8,
        marker="X",
        edgecolors="white",
        linewidth=2,
        label="Cluster Centers",
        zorder=10
    )
    
    # Plot train stations
    train_coords = np.array([[station_dict[sid][1], station_dict[sid][0]] 
                             for sid in train_stations])
    ax1.scatter(
        train_coords[:, 0],
        train_coords[:, 1],
        c="lightgray",
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
        marker="o",
        label=f"Train Stations (n={len(train_stations)})",
        zorder=5
    )
    
    # Plot test stations colored by error magnitude
    scatter1 = ax1.scatter(
        station_errors["lon"],
        station_errors["lat"],
        c=station_errors["mean_abs_error"],
        cmap="YlOrRd",
        s=350,
        alpha=0.95,
        edgecolors="black",
        linewidth=2.5,
        marker="s",
        vmin=0,
        vmax=station_errors["mean_abs_error"].quantile(0.95),
        label=f"Test Stations (n={len(test_stations)})",
        zorder=8
    )
    
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label("Mean Absolute Error", fontsize=12, fontweight="bold")
    
    # Highlight high-error stations with red circles
    high_error = station_errors[station_errors["is_high_error"]]
    if len(high_error) > 0:
        ax1.scatter(
            high_error["lon"],
            high_error["lat"],
            s=700,
            facecolors="none",
            edgecolors="red",
            linewidth=4,
            label=f"High Error (top {100-error_threshold_percentile}%, n={len(high_error)})",
            zorder=9
        )
        
        # Annotate worst 3 stations
        worst3 = high_error.nlargest(3, "mean_abs_error")
        for idx, (_, row) in enumerate(worst3.iterrows()):
            ax1.annotate(
                f"#{idx+1}: {row['station_id']}\n"
                f"Error: {row['mean_abs_error']:.2f}\n"
                f"Bias: {row['bias']:+.2f}",
                xy=(row["lon"], row["lat"]),
                xytext=(20, 20 - idx*15),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="yellow",
                    alpha=0.85,
                    edgecolor="red",
                    linewidth=2
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0.3",
                    color="red",
                    lw=2
                ),
                zorder=11
            )
    
    ax1.set_xlabel("Longitude", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Latitude", fontsize=13, fontweight="bold")
    ax1.set_title(
        f"Fold {station_errors['fold'].iloc[0]}: Spatial Distribution with Error Magnitude\n"
        f"Test: {len(test_stations)} | Train: {len(train_stations)} | High Error: {len(high_error)}",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    visualise_singapore_outline(ax=ax1)
    visualise_with_basemap(ax=ax1)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="best", fontsize=9, framealpha=0.95, ncol=2)
    
    # === RIGHT PLOT: Error Categories + Bias ===
    ax2 = axes[1]
    
    # Plot clusters background
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = station_coords[cluster_mask]
        ax2.scatter(
            cluster_coords[:, 0],
            cluster_coords[:, 1],
            c=[colors[cluster_id]],
            s=80,
            alpha=0.25,
            edgecolors="gray",
            linewidth=0.5
        )
    
    # Plot centroids
    ax2.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=300,
        alpha=0.8,
        marker="X",
        edgecolors="white",
        linewidth=2,
        zorder=10
    )
    
    # Plot train stations
    ax2.scatter(
        train_coords[:, 0],
        train_coords[:, 1],
        c="lightgray",
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
        marker="o",
        zorder=5
    )
    
    # Categorize errors into tertiles
    station_errors["error_category"] = pd.cut(
        station_errors["mean_abs_error"],
        bins=[0,
              station_errors["mean_abs_error"].quantile(0.33),
              station_errors["mean_abs_error"].quantile(0.67),
              station_errors["mean_abs_error"].max()],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )
    
    # Plot test stations colored by error category
    category_colors = {"Low": "green", "Medium": "orange", "High": "red"}
    category_markers = {"Low": "o", "Medium": "s", "High": "D"}
    
    for category in ["Low", "Medium", "High"]:
        subset = station_errors[station_errors["error_category"] == category]
        if len(subset) > 0:
            ax2.scatter(
                subset["lon"],
                subset["lat"],
                c=category_colors[category],
                s=300,
                edgecolors="black",
                linewidth=2.5,
                alpha=0.9,
                marker=category_markers[category],
                label=f"{category} Error (n={len(subset)})",
                zorder=8
            )
    
    # Add bias indicators (triangles for over/under prediction)
    over_pred = station_errors[station_errors["bias"] > 1.0]
    under_pred = station_errors[station_errors["bias"] < -1.0]
    
    if len(over_pred) > 0:
        ax2.scatter(
            over_pred["lon"],
            over_pred["lat"],
            marker="^",
            s=150,
            c="blue",
            alpha=0.7,
            edgecolors="white",
            linewidth=1.5,
            label=f"Over-predict (bias>1, n={len(over_pred)})",
            zorder=9
        )
    
    if len(under_pred) > 0:
        ax2.scatter(
            under_pred["lon"],
            under_pred["lat"],
            marker="v",
            s=150,
            c="purple",
            alpha=0.7,
            edgecolors="white",
            linewidth=1.5,
            label=f"Under-predict (bias<-1, n={len(under_pred)})",
            zorder=9
        )
    
    # Annotate stations with extreme bias
    extreme_bias_candidates = station_errors[abs(station_errors["bias"]) > 2.0].copy()
    if len(extreme_bias_candidates) > 0:
        extreme_bias_candidates["abs_bias"] = abs(extreme_bias_candidates["bias"])
        extreme_bias = extreme_bias_candidates.nlargest(min(2, len(extreme_bias_candidates)), "abs_bias")
    else:
        extreme_bias = pd.DataFrame()
    
    for _, row in extreme_bias.iterrows():
        ax2.annotate(
            f"{row['station_id']}\nBias: {row['bias']:+.2f}",
            xy=(row["lon"], row["lat"]),
            xytext=(15, -15),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="lightblue" if row["bias"] > 0 else "plum",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5
            ),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=-0.3",
                color="black",
                lw=1.5
            ),
            zorder=11
        )
    
    ax2.set_xlabel("Longitude", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Latitude", fontsize=13, fontweight="bold")
    ax2.set_title(
        f"Fold {station_errors['fold'].iloc[0]}: Error Categories & Prediction Bias\n"
        f"Over-prediction: {len(over_pred)} | Under-prediction: {len(under_pred)}",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    visualise_singapore_outline(ax=ax2)
    visualise_with_basemap(ax=ax2)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="best", fontsize=9, framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n✓ Saved error visualization to '{output_path}'")
    
    # Print cluster-wise statistics
    print("\n" + "="*80)
    print("ERROR STATISTICS BY CLUSTER")
    print("="*80)
    cluster_stats = station_errors.groupby("cluster").agg({
        "mean_abs_error": ["mean", "max"],
        "bias": "mean",
        "station_id": "count"
    }).round(3)
    cluster_stats.columns = ["Mean Error", "Max Error", "Mean Bias", "N Stations"]
    print(cluster_stats.to_string())
    
    return station_errors


# === SIMPLIFIED USAGE FUNCTION ===
def quick_visualize_fold(
    experiment_name,
    fold_idx,
    fold_result,
    station_dict,
    n_clusters=8,
    error_threshold_percentile=75,
    seed=42
):
    """
    Quick one-line visualization for a single fold.
    
    Parameters
    ----------
    experiment_name : str
        Name of experiment directory
    fold_idx : int
        Fold index (0, 1, 2, ...)
    fold_result : dict
        Single fold result from stratified_spatial_kfold_dual()
    station_dict : dict
        {station_id: [lat, lon]}
    n_clusters : int
        Number of clusters
    error_threshold_percentile : float
        Percentile for high error threshold
    seed : int
        Random seed
        
    Returns
    -------
    station_errors : pd.DataFrame
    """
    
    station_error_csv = f"experiments/{experiment_name}/predictions/station_errors_fold{fold_idx}.csv"
    output_path = f"experiments/{experiment_name}/error_map_fold{fold_idx}.png"
    
    return visualize_single_fold_errors(
        station_error_csv=station_error_csv,
        fold_result=fold_result,
        station_dict=station_dict,
        n_clusters=n_clusters,
        error_threshold_percentile=error_threshold_percentile,
        seed=seed,
        output_path=output_path
    )

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # After running K-Fold CV and testing all folds:
    weather_station_locations = get_station_coordinate_mappings()
    # Run K-Fold CV
    fold_results = stratified_spatial_kfold_dual(
        weather_station_locations,
        n_splits=5,
        validation_percent=20,
        n_clusters=8,
        seed=123,
        plot=False,
    )

    fold_result = fold_results[2]  # Get fold 2
    station_errors = visualize_single_fold_errors(
        station_error_csv="experiments/20251112_161807_new/predictions/station_errors_fold2.csv",
        fold_result=fold_result,
        station_dict=weather_station_locations,
        n_clusters=8,
        error_threshold_percentile=75,
        seed=123,
        output_path="experiments/20251112_161807_new/error_map_fold2.png"
    )
    
    # Example 2: Quick visualization
    
    station_errors = quick_visualize_fold(
        experiment_name="20251112_161807_new",
        fold_idx=2,
        fold_result=fold_results[2],
        station_dict=weather_station_locations
    )
    
    # Analyze errors and create visualizations
    # station_avg_errors = analyze_and_visualize_station_errors(
    #     fold_results=fold_results,
    #     station_dict=weather_station_locations,
    #     experiment_name="my_experiment",
    #     n_clusters=8,
    #     error_threshold_percentile=75,
    #     seed=42
    # )
    