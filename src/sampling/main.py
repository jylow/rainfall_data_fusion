import numpy as np
from sklearn.cluster import KMeans

from src.visualization.sampling import create_dual_sampling_visualization


def stratified_spatial_sampling_dual(
    station_dict,
    test_percent=10,
    validation_percent=20,
    n_clusters=8,
    seed=42,
    plot=True,
    output_path="dual_sampling_results.png",
):
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
    seed : int
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

    print("=" * 70)
    print("K-MEANS STRATIFIED SPATIAL SAMPLING")
    print("=" * 70)
    print(f"Total stations: {n_stations}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Random seed: {seed}")
    print("\nSplit configuration:")
    print(f"  - Test (shared): {test_percent}%")
    print(f"  - Statistical train: {100 - test_percent}%")
    print(f"  - ML train: {100 - test_percent - validation_percent}%")
    print(f"  - ML validation: {validation_percent}%")

    # Step 1: Perform K-means clustering
    print(f"\n{'=' * 70}")
    print("STEP 1: K-means Clustering")
    print(f"{'=' * 70}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(station_coords)
    centroids = kmeans.cluster_centers_

    print("Clustering complete. Cluster centers:")
    for i, centroid in enumerate(centroids):
        n_in_cluster = np.sum(cluster_labels == i)
        print(
            f"  Cluster {i}: ({centroid[0]:.4f}, {centroid[1]:.4f}) - {n_in_cluster} stations"
        )

    # Step 2: Stratified sampling - first split out test set
    print(f"\n{'=' * 70}")
    print(f"STEP 2: Creating Shared Test Set ({test_percent}%)")
    print(f"{'=' * 70}")

    test_indices = []
    remaining_indices = []

    for cluster_id in range(n_clusters):
        # Get indices of stations in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        n_cluster = len(cluster_indices)

        # Shuffle indices with cluster-specific seed for reproducibility
        np.random.seed(seed + cluster_id)
        shuffled_indices = cluster_indices.copy()
        np.random.shuffle(shuffled_indices)

        # Calculate test size for this cluster (minimum 1 if cluster is large enough)
        n_test = max(1, int(n_cluster * test_percent / 100)) if n_cluster > 2 else 0

        # Split: test vs remaining
        cluster_test = shuffled_indices[:n_test]
        cluster_remaining = shuffled_indices[n_test:]

        test_indices.extend(cluster_test)
        remaining_indices.extend(cluster_remaining)

        print(
            f"  Cluster {cluster_id}: {n_cluster} stations → {n_test} test, {len(cluster_remaining)} remaining"
        )

    test_indices = np.array(test_indices)
    remaining_indices = np.array(remaining_indices)

    test_stations = station_ids[test_indices]

    print(
        f"\nTest set created: {len(test_stations)} stations ({len(test_stations) / n_stations * 100:.1f}%)"
    )

    # Step 3: Statistical method split (remaining → train)
    print(f"\n{'=' * 70}")
    print("STEP 3: Statistical Method Split")
    print(f"{'=' * 70}")

    statistical_train_stations = station_ids[remaining_indices]

    print(
        f"  Training: {len(statistical_train_stations)} stations ({len(statistical_train_stations) / n_stations * 100:.1f}%)"
    )
    print(
        f"  Test: {len(test_stations)} stations ({len(test_stations) / n_stations * 100:.1f}%)"
    )

    # Step 4: ML method split (remaining → train + validation)
    print(f"\n{'=' * 70}")
    print("STEP 4: Machine Learning Method Split")
    print(f"{'=' * 70}")

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
        np.random.seed(seed + cluster_id + 100)
        shuffled_remaining = np.array(cluster_remaining).copy()
        np.random.shuffle(shuffled_remaining)

        # Split remaining into train and validation
        n_validation = (
            max(1, int(n_remaining * validation_from_remaining / 100))
            if n_remaining > 1
            else 0
        )

        cluster_train = shuffled_remaining[n_validation:]
        cluster_validation = shuffled_remaining[:n_validation]

        ml_train_indices.extend(cluster_train)
        ml_validation_indices.extend(cluster_validation)

        print(
            f"  Cluster {cluster_id}: {n_remaining} remaining → {len(cluster_train)} train, {len(cluster_validation)} validation"
        )

    ml_train_indices = np.array(ml_train_indices)
    ml_validation_indices = np.array(ml_validation_indices)

    ml_train_stations = station_ids[ml_train_indices]
    ml_validation_stations = station_ids[ml_validation_indices]

    print("\nML split summary:")
    print(
        f"  Training: {len(ml_train_stations)} stations ({len(ml_train_stations) / n_stations * 100:.1f}%)"
    )
    print(
        f"  Validation: {len(ml_validation_stations)} stations ({len(ml_validation_stations) / n_stations * 100:.1f}%)"
    )
    print(
        f"  Test: {len(test_stations)} stations ({len(test_stations) / n_stations * 100:.1f}%)"
    )

    # Prepare results dictionary
    results = {
        "statistical": {"train": statistical_train_stations, "test": test_stations},
        "ml": {
            "train": ml_train_stations,
            "validation": ml_validation_stations,
            "test": test_stations,
        },
        "test_stations": test_stations,
        "cluster_labels": cluster_labels,
    }

    # Create visualization if requested
    if plot:
        create_dual_sampling_visualization(
            results, station_coords, station_ids, cluster_labels, centroids, output_path=output_path
        )

    return results

def stratified_spatial_kfold_dual(
    station_dict,
    n_splits=5,
    validation_percent=20,
    n_clusters=8,
    seed=42,
    plot=False,
):
    """
    Perform Spatial K-Fold Cross-Validation with stratified sampling using K-means clustering.
    
    Key feature: Ensures EXACTLY ONE test station per cluster per fold (when cluster size allows).
    
    Each fold has:
      - Statistical method: train/test split (no overlap)
      - ML method: train/validation/test split (same test as statistical)
    The test sets across folds are disjoint (true K-fold CV).

    Parameters
    ----------
    station_dict : dict
        {station_id: [lat, lon]}
    n_splits : int
        Number of K-Fold splits (e.g., 5)
    validation_percent : float
        Percent of *training data* used for ML validation
    n_clusters : int
        Number of spatial clusters
    seed : int
        Random seed for reproducibility
    plot : bool
        Whether to visualize each fold

    Returns
    -------
    folds : list of dict
        Each dict contains:
          {
            'fold_index': int,
            'statistical': {'train': [...], 'test': [...]},
            'ml': {'train': [...], 'validation': [...], 'test': [...]},
            'cluster_labels': [...],
          }
    """

    # --- Step 1: Prepare data
    station_ids = np.array(list(station_dict.keys()))
    station_coords = np.array([[lon, lat] for lat, lon in station_dict.values()])
    n_stations = len(station_ids)

    print("=" * 80)
    print("SPATIAL K-FOLD STRATIFIED SAMPLING (Fixed)")
    print("=" * 80)
    print(f"Total stations: {n_stations}")
    print(f"Clusters: {n_clusters}")
    print(f"Folds: {n_splits}")
    print(f"Random seed: {seed}")
    print("-" * 80)

    # --- Step 2: K-Means clustering (done ONCE for consistency)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(station_coords)

    # --- Step 3: Create rotation scheme for each cluster
    # Ensures exactly 1 test station per cluster per fold (when possible)
    np.random.seed(seed)
    cluster_rotations = {}
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        n_cluster = len(cluster_indices)
        
        # Shuffle stations in this cluster
        shuffled_indices = cluster_indices.copy()
        np.random.shuffle(shuffled_indices)
        
        # Create rotation: each fold gets exactly 1 test station (if cluster has >= n_splits stations)
        # Remaining stations go to training pool
        if n_cluster < n_splits:
            print(f"WARNING: Cluster {cluster_id} has only {n_cluster} stations (< {n_splits} folds)")
            # Pad with None for folds that won't get a test station from this cluster
            test_rotation = list(shuffled_indices) + [None] * (n_splits - n_cluster)
            train_pool = []
        else:
            # Each fold gets 1 test station, rest go to train pool
            test_rotation = shuffled_indices[:n_splits].tolist()
            train_pool = shuffled_indices[n_splits:].tolist()
        
        cluster_rotations[cluster_id] = {
            'test_rotation': test_rotation,  # Length = n_splits
            'train_pool': train_pool          # Stations never used for testing
        }
        
        print(f"Cluster {cluster_id}: {n_cluster} stations → "
              f"{len([x for x in test_rotation if x is not None])} test assignments, "
              f"{len(train_pool)} always-train")

    # --- Step 4: Build K-Fold splits
    folds_results = []

    for fold_idx in range(n_splits):
        print(f"\n{'='*80}")
        print(f"Creating Fold {fold_idx+1}/{n_splits}")
        print(f"{'='*80}")

        test_indices = []
        train_indices = []

        # Collect data for this fold
        for cluster_id in range(n_clusters):
            rotation = cluster_rotations[cluster_id]
            
            # Get the test station for this fold from this cluster
            test_station = rotation['test_rotation'][fold_idx]
            if test_station is not None:
                test_indices.append(test_station)
            
            # Training pool: all stations NOT testing in this fold
            for other_fold_idx in range(n_splits):
                if other_fold_idx != fold_idx:
                    other_test_station = rotation['test_rotation'][other_fold_idx]
                    if other_test_station is not None:
                        train_indices.append(other_test_station)
            
            # Add stations from train_pool (never used for testing)
            train_indices.extend(rotation['train_pool'])

        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)

        print(f"Test stations: {len(test_indices)} (should be ~{n_clusters})")
        print(f"Train pool: {len(train_indices)}")

        # --- Step 5: ML validation split (from training pool)
        ml_train_indices = []
        ml_val_indices = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_train_indices = [idx for idx in train_indices if cluster_mask[idx]]
            n_cluster_train = len(cluster_train_indices)

            if n_cluster_train == 0:
                continue

            # Shuffle and split
            np.random.seed(seed + 100 + fold_idx + cluster_id)
            shuffled = np.random.permutation(cluster_train_indices)
            n_val = max(1, int(len(shuffled) * validation_percent / 100)) if n_cluster_train > 1 else 0
            cluster_val = shuffled[:n_val]
            cluster_ml_train = shuffled[n_val:]

            ml_train_indices.extend(cluster_ml_train)
            ml_val_indices.extend(cluster_val)

        # --- Step 6: Convert to station IDs
        test_stations = station_ids[test_indices]
        stat_train_stations = station_ids[train_indices]
        ml_train_stations = station_ids[ml_train_indices]
        ml_val_stations = station_ids[ml_val_indices]

        # --- Step 7: Store results
        fold_result = {
            "fold_index": fold_idx,
            "statistical": {
                "train": stat_train_stations,
                "test": test_stations,
            },
            "ml": {
                "train": ml_train_stations,
                "validation": ml_val_stations,
                "test": test_stations,
            },
            "test_stations": test_stations,
            "cluster_labels": cluster_labels,
        }

        folds_results.append(fold_result)

        print(f"Fold {fold_idx+1} summary:")
        print(f"  Statistical train: {len(stat_train_stations)} "
              f"({len(stat_train_stations)/n_stations*100:.1f}%)")
        print(f"  Test: {len(test_stations)} ({len(test_stations)/n_stations*100:.1f}%)")
        print(f"  ML train: {len(ml_train_stations)} ({len(ml_train_stations)/n_stations*100:.1f}%)")
        print(f"  ML validation: {len(ml_val_stations)} ({len(ml_val_stations)/n_stations*100:.1f}%)")

        if plot:
            create_dual_sampling_visualization(
                fold_result, station_coords, station_ids, cluster_labels, 
                kmeans.cluster_centers_, 
                output_path=f"fold_{fold_idx+1}_sampling_results.png"
            )

    # --- Verification: Ensure no station appears in multiple test sets
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")
    all_test_stations = set()
    for fold in folds_results:
        fold_test = set(fold['test_stations'])
        overlap = all_test_stations & fold_test
        if overlap:
            print(f"ERROR: Fold {fold['fold_index']+1} has overlapping test stations: {overlap}")
        all_test_stations.update(fold_test)
    
    print(f"Total unique test stations across all folds: {len(all_test_stations)}")
    print(f"Expected: {min(n_clusters * n_splits, n_stations)}")
    print("✓ K-Fold validation complete!")

    return folds_results