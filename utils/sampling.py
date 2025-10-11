import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
import contextily as cx
from utils.visualisation import visualise_singapore_outline, visualise_with_basemap

def stratified_spatial_sampling_dual(station_dict, test_percent=10, validation_percent=20,
                                     n_clusters=8, random_state=42, plot=True):
    """
    Perform stratified spatial sampling using K-means clustering for both statistical and ML methods.
    
    The 10% test set is shared between both methods:
    - Statistical method: 90% train, 10% test
    - ML method: 70% train, 20% validation, 10% test (same as statistical)
    
    Parameters:
    -----------
    station_dict : dict
        Dictionary with structure {station_id: [lat, lon]}
    test_percent : float
        Percentage of stations to use for testing (default: 10)
        This test set is shared between both methods
    validation_percent : float
        Percentage of stations for ML validation (default: 20)
        Only used for ML method
    n_clusters : int
        Number of spatial clusters to create (default: 8)
    random_state : int
        Random seed for reproducibility (default: 42)
    plot : bool
        Whether to create visualization (default: True)
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'statistical': {'train': station_ids, 'test': station_ids}
        - 'ml': {'train': station_ids, 'validation': station_ids, 'test': station_ids}
        - 'test_stations': shared test station IDs
        - 'cluster_labels': cluster assignment for each station
    """
    
    # Extract station IDs and coordinates
    station_ids = np.array(list(station_dict.keys()))
    # Convert [lat, lon] to [lon, lat] for standard coordinate convention
    station_coords = np.array([[lon, lat] for lat, lon in station_dict.values()])
    
    n_stations = len(station_ids)
    
    # Validate input
    if n_clusters > n_stations:
        n_clusters = n_stations
        print(f"Warning: n_clusters reduced to {n_stations} (total number of stations)")
    
    # Validate percentages
    if test_percent + validation_percent >= 100:
        raise ValueError("test_percent + validation_percent must be less than 100")
    
    print(f"="*70)
    print(f"K-MEANS STRATIFIED SPATIAL SAMPLING")
    print(f"="*70)
    print(f"Total stations: {n_stations}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Random seed: {random_state}")
    print(f"\nSplit configuration:")
    print(f"  - Test (shared): {test_percent}%")
    print(f"  - Statistical train: {100-test_percent}%")
    print(f"  - ML train: {100-test_percent-validation_percent}%")
    print(f"  - ML validation: {validation_percent}%")
    
    # Step 1: Perform K-means clustering
    print(f"\n{'='*70}")
    print(f"STEP 1: K-means Clustering")
    print(f"{'='*70}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(station_coords)
    centroids = kmeans.cluster_centers_
    
    print(f"Clustering complete. Cluster centers:")
    for i, centroid in enumerate(centroids):
        n_in_cluster = np.sum(cluster_labels == i)
        print(f"  Cluster {i}: ({centroid[0]:.4f}, {centroid[1]:.4f}) - {n_in_cluster} stations")
    
    # Step 2: Stratified sampling - first split out test set
    print(f"\n{'='*70}")
    print(f"STEP 2: Creating Shared Test Set ({test_percent}%)")
    print(f"{'='*70}")
    
    test_indices = []
    remaining_indices = []
    
    for cluster_id in range(n_clusters):
        # Get indices of stations in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        n_cluster = len(cluster_indices)
        
        # Shuffle indices with cluster-specific seed for reproducibility
        np.random.seed(random_state + cluster_id)
        shuffled_indices = cluster_indices.copy()
        np.random.shuffle(shuffled_indices)
        
        # Calculate test size for this cluster (minimum 1 if cluster is large enough)
        n_test = max(1, int(n_cluster * test_percent / 100)) if n_cluster > 2 else 0
        
        # Split: test vs remaining
        cluster_test = shuffled_indices[:n_test]
        cluster_remaining = shuffled_indices[n_test:]
        
        test_indices.extend(cluster_test)
        remaining_indices.extend(cluster_remaining)
        
        print(f"  Cluster {cluster_id}: {n_cluster} stations → {n_test} test, {len(cluster_remaining)} remaining")
    
    test_indices = np.array(test_indices)
    remaining_indices = np.array(remaining_indices)
    
    test_stations = station_ids[test_indices]
    
    print(f"\nTest set created: {len(test_stations)} stations ({len(test_stations)/n_stations*100:.1f}%)")
    
    # Step 3: Statistical method split (remaining → train)
    print(f"\n{'='*70}")
    print(f"STEP 3: Statistical Method Split")
    print(f"{'='*70}")
    
    statistical_train_stations = station_ids[remaining_indices]
    
    print(f"  Training: {len(statistical_train_stations)} stations ({len(statistical_train_stations)/n_stations*100:.1f}%)")
    print(f"  Test: {len(test_stations)} stations ({len(test_stations)/n_stations*100:.1f}%)")
    
    # Step 4: ML method split (remaining → train + validation)
    print(f"\n{'='*70}")
    print(f"STEP 4: Machine Learning Method Split")
    print(f"{'='*70}")
    
    ml_train_indices = []
    ml_validation_indices = []
    
    # Calculate validation percentage from remaining data
    remaining_percent = 100 - test_percent
    validation_from_remaining = (validation_percent / remaining_percent) * 100
    
    for cluster_id in range(n_clusters):
        # Get remaining indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Filter to only remaining indices (not test)
        cluster_remaining = [idx for idx in cluster_indices if idx in remaining_indices]
        n_remaining = len(cluster_remaining)
        
        if n_remaining == 0:
            continue
        
        # Shuffle with different seed for train/val split
        np.random.seed(random_state + cluster_id + 100)
        shuffled_remaining = np.array(cluster_remaining).copy()
        np.random.shuffle(shuffled_remaining)
        
        # Split remaining into train and validation
        n_validation = max(1, int(n_remaining * validation_from_remaining / 100)) if n_remaining > 1 else 0
        
        cluster_train = shuffled_remaining[n_validation:]
        cluster_validation = shuffled_remaining[:n_validation]
        
        ml_train_indices.extend(cluster_train)
        ml_validation_indices.extend(cluster_validation)
        
        print(f"  Cluster {cluster_id}: {n_remaining} remaining → {len(cluster_train)} train, {len(cluster_validation)} validation")
    
    ml_train_indices = np.array(ml_train_indices)
    ml_validation_indices = np.array(ml_validation_indices)
    
    ml_train_stations = station_ids[ml_train_indices]
    ml_validation_stations = station_ids[ml_validation_indices]
    
    print(f"\nML split summary:")
    print(f"  Training: {len(ml_train_stations)} stations ({len(ml_train_stations)/n_stations*100:.1f}%)")
    print(f"  Validation: {len(ml_validation_stations)} stations ({len(ml_validation_stations)/n_stations*100:.1f}%)")
    print(f"  Test: {len(test_stations)} stations ({len(test_stations)/n_stations*100:.1f}%)")
    
    # Prepare results dictionary
    results = {
        'statistical': {
            'train': statistical_train_stations,
            'test': test_stations
        },
        'ml': {
            'train': ml_train_stations,
            'validation': ml_validation_stations,
            'test': test_stations
        },
        'test_stations': test_stations,
        'cluster_labels': cluster_labels,
    }
    
    # Create visualization if requested
    if plot:
        create_dual_sampling_visualization(results, station_coords, station_ids, 
                                          cluster_labels, centroids)
    
    return results


def create_dual_sampling_visualization(results, station_coords, station_ids, 
                                      cluster_labels, centroids):
    """Create comprehensive visualization of the dual sampling strategy."""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Create lookup dictionaries for quick checking
    stat_train_set = set(results['statistical']['train'])
    ml_train_set = set(results['ml']['train'])
    ml_val_set = set(results['ml']['validation'])
    test_set = set(results['test_stations'])
    
    # Plot 1: Clusters (top-left)
    ax1 = axes[0, 0]
    for cluster_id in range(len(centroids)):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = station_coords[cluster_mask]
        
        ax1.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                   c=[colors[cluster_id]], s=120, alpha=0.6, 
                   label=f'Cluster {cluster_id}', edgecolors='black', linewidth=0.5)
    
    # Plot centroids
    ax1.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', s=400, alpha=0.5, marker='X', 
               edgecolors='darkred', linewidth=2, label='Centroids', zorder=5)
    
    ax1.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax1.set_title('K-means Clustering Results', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    visualise_singapore_outline(ax=ax1)    
    visualise_with_basemap(ax=ax1)
    
    # Plot 2: Statistical Method Split (top-right)
    ax2 = axes[0, 1]
    for i, station_id in enumerate(station_ids):
        cluster_id = cluster_labels[i]
        lon, lat = station_coords[i]
        
        if station_id in test_set:
            marker, size, label_text = 's', 180, 'Test'
            edgecolor, linewidth = 'red', 2
        else:  # training
            marker, size, label_text = 'o', 120, 'Train'
            edgecolor, linewidth = 'black', 0.5
        
        ax2.scatter(lon, lat, c=[colors[cluster_id]], s=size, marker=marker,
                   alpha=0.7, edgecolors=edgecolor, linewidth=linewidth)
    
    ax2.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax2.set_title('Statistical Method Split (90% Train / 10% Test)', 
                 fontsize=14, fontweight='bold', pad=20)
    visualise_singapore_outline(ax=ax2)
    visualise_with_basemap(ax=ax2)
    
    # Custom legend for statistical
    train_patch = mpatches.Patch(color='gray', label=f'Training: {len(stat_train_set)} stations')
    test_patch = mpatches.Patch(color='red', label=f'Test: {len(test_set)} stations')
    ax2.legend(handles=[train_patch, test_patch], loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ML Method Split (bottom-left)
    ax3 = axes[1, 0]
    for i, station_id in enumerate(station_ids):
        cluster_id = cluster_labels[i]
        lon, lat = station_coords[i]
        
        if station_id in test_set:
            marker, size, label_text = 's', 180, 'Test'
            edgecolor, linewidth = 'red', 2
        elif station_id in ml_val_set:
            marker, size, label_text = '^', 150, 'Validation'
            edgecolor, linewidth = 'blue', 2
        else:  # training
            marker, size, label_text = 'o', 120, 'Train'
            edgecolor, linewidth = 'black', 0.5
        
        ax3.scatter(lon, lat, c=[colors[cluster_id]], s=size, marker=marker,
                   alpha=0.7, edgecolors=edgecolor, linewidth=linewidth)
    
    ax3.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax3.set_title('ML Method Split (70% Train / 20% Val / 10% Test)', 
                 fontsize=14, fontweight='bold', pad=20)
    visualise_singapore_outline(ax=ax3)
    visualise_with_basemap(ax=ax3)
    
    # Custom legend for ML
    train_patch = mpatches.Patch(color='gray', label=f'Training: {len(ml_train_set)} stations')
    val_patch = mpatches.Patch(color='blue', label=f'Validation: {len(ml_val_set)} stations')
    test_patch = mpatches.Patch(color='red', label=f'Test: {len(test_set)} stations (shared)')
    ax3.legend(handles=[train_patch, val_patch, test_patch], loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dual_sampling_results.png', dpi=300, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Visualization saved as 'dual_sampling_results.png'")
    print(f"{'='*70}")
    plt.show()


def save_sampling_results(results, filename='sampling_results.txt'):
    """Save sampling results to a text file."""
    
    with open(filename, 'w') as f:
        f.write("K-MEANS STRATIFIED SPATIAL SAMPLING RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("SHARED TEST SET\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of stations: {len(results['test_stations'])}\n")
        f.write(f"Station IDs:\n")
        for station_id in sorted(results['test_stations']):
            f.write(f"  {station_id}\n")
        
        f.write("\n" + "="*70 + "\n\n")
        
        f.write("STATISTICAL METHOD (90% Train / 10% Test)\n")
        f.write("-"*70 + "\n")
        f.write(f"Training stations: {len(results['statistical']['train'])}\n")
        f.write(f"Test stations: {len(results['statistical']['test'])}\n\n")
        
        f.write("Training Station IDs:\n")
        for station_id in sorted(results['statistical']['train']):
            f.write(f"  {station_id}\n")
        
        f.write("\n" + "="*70 + "\n\n")
        
        f.write("MACHINE LEARNING METHOD (70% Train / 20% Val / 10% Test)\n")
        f.write("-"*70 + "\n")
        f.write(f"Training stations: {len(results['ml']['train'])}\n")
        f.write(f"Validation stations: {len(results['ml']['validation'])}\n")
        f.write(f"Test stations: {len(results['ml']['test'])}\n\n")
        
        f.write("Training Station IDs:\n")
        for station_id in sorted(results['ml']['train']):
            f.write(f"  {station_id}\n")
        
        f.write("\nValidation Station IDs:\n")
        for station_id in sorted(results['ml']['validation']):
            f.write(f"  {station_id}\n")
    
    print(f"Results saved to '{filename}'")
