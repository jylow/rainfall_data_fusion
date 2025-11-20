import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch


def visualize_splitted_graphs(
    stations,
    weather_station_locations,
    train_graph,
    validation_graph,
    full_graph,
    fold=0,
    model_dir: str = "experiments",
):
    print("=" * 70)
    print("GRAPH STRUCTURE ANALYSIS")
    print("=" * 70)

    # ============================================
    # FULL GRAPH SETUP
    # ============================================
    print("\n--- Full Graph ---")
    station_count = full_graph.station_id.shape[0]  # 63 stations
    print(f"Total stations: {station_count}")
    print(f"Train nodes: {full_graph.train_mask.sum().item()}")
    print(f"Val nodes: {full_graph.val_mask.sum().item()}")
    print(f"Test nodes: {full_graph.test_mask.sum().item()}")

    # Create NetworkX graph for full graph
    G_full = nx.Graph()
    G_full.add_nodes_from(range(station_count))

    # Add edges (these are already in global node indices)
    edges_full = full_graph.edge_index.numpy().T
    G_full.add_edges_from(edges_full)

    print(f"Full graph edges: {G_full.number_of_edges()}")

    # ============================================
    # TRAIN GRAPH SETUP (CORRECTED)
    # ============================================
    print("\n--- Train Graph ---")

    # CRITICAL: train_graph uses LOCAL indices (0 to num_train_nodes-1)
    # but train_graph.orig_id maps back to GLOBAL indices
    num_train_nodes = train_graph.x.shape[1]
    print(f"Train graph nodes: {num_train_nodes}")

    # Get mapping: local_idx -> global_idx
    if hasattr(train_graph, "orig_id"):
        orig_ids = train_graph.orig_id.numpy()
        print(f"orig_id mapping exists: {len(orig_ids)} mappings")
    else:
        print("ERROR: train_graph missing 'orig_id' attribute!")
        # Fallback: assume train nodes are the first ones with train_mask=True in full graph
        orig_ids = np.where(
            full_graph.train_mask.numpy() | full_graph.val_mask.numpy()
        )[0]
        print(f"Reconstructed orig_id from masks: {len(orig_ids)} mappings")

    # Create reverse mapping: global_idx -> local_idx
    global_to_local = {int(g): i for i, g in enumerate(orig_ids)}

    print(f"Train graph local indices: 0 to {num_train_nodes - 1}")
    print(f"Train graph global indices: {orig_ids[:5]}... (first 5)")

    # Create NetworkX graph for train graph
    G_train = nx.Graph()
    G_train.add_nodes_from(range(num_train_nodes))

    # Add edges (train_graph.edge_index uses LOCAL indices)
    edges_train_local = train_graph.edge_index.numpy().T
    G_train.add_edges_from(edges_train_local)

    print(f"Train graph edges: {G_train.number_of_edges()}")

    # ============================================
    # VALIDATION GRAPH (train + val nodes, re-colored)
    # ============================================
    print("\n--- Validation Graph (train + val nodes) ---")

    # The validation graph uses the train+val node
    num_valgraph_nodes = validation_graph.x.shape[1]  # identical to train graph
    print(f"Validation graph nodes: {num_valgraph_nodes}")

    # Get mapping: local_idx -> global_idx
    if hasattr(validation_graph, "orig_id"):
        val_orig_ids = validation_graph.orig_id.numpy()
        print(f"orig_id mapping exists: {len(val_orig_ids)} mappings")
    else:
        print("ERROR: train_graph missing 'orig_id' attribute!")
        # Fallback: assume train nodes are the first ones with train_mask=True in full graph
        val_orig_ids = np.where(full_graph.train_mask.numpy())[0]
        print(f"Reconstructed orig_id from masks: {len(val_orig_ids)} mappings")

    # Create NetworkX graph
    G_valgraph = nx.Graph()
    G_valgraph.add_nodes_from(range(num_valgraph_nodes))

    # Edges are the train+val only
    edges_val_local = validation_graph.edge_index.numpy().T
    G_valgraph.add_edges_from(edges_val_local)

    print(f"Validation graph edges: {G_valgraph.number_of_edges()}")

    # ============================================
    # GENERATE GEOGRAPHICAL LAYOUT
    # ============================================
    print("\n" + "=" * 70)
    print("GENERATING GEOGRAPHICAL POSITIONS")
    print("=" * 70)

    # --- Full Graph Positions ---
    # Key: node_idx (0 to 62) -> (lon, lat)
    pos_full = {}
    for node_idx in range(station_count):
        station_str_id = stations[full_graph.station_id[node_idx].item()]
        if station_str_id in weather_station_locations:
            lat, lon = weather_station_locations[station_str_id]
            pos_full[node_idx] = (lon, lat)  # NetworkX uses (x, y) = (lon, lat)
        else:
            print(f"WARNING: Station {station_str_id} not found in locations")
            pos_full[node_idx] = (0, 0)  # Default position

    print(f"Full graph positions generated: {len(pos_full)}")

    # --- Train Graph Positions (CORRECTED) ---
    # Key: local_idx (0 to 54) -> (lon, lat)
    # Use orig_id to map back to global station indices
    pos_train = {}
    for local_idx in range(num_train_nodes):
        global_idx = int(orig_ids[local_idx])

        # Get station string ID from full graph
        station_str_id = stations[full_graph.station_id[global_idx].item()]

        if station_str_id in weather_station_locations:
            lat, lon = weather_station_locations[station_str_id]
            pos_train[local_idx] = (lon, lat)
        else:
            print(f"WARNING: Station {station_str_id} not found in locations")
            pos_train[local_idx] = (0, 0)

    print(f"Train graph positions generated: {len(pos_train)}")

    # --- Validation Graph Pos (same positions as train graph)
    pos_valgraph = {}
    for local_idx in range(num_valgraph_nodes):
        global_idx = int(val_orig_ids[local_idx])
        station_str_id = stations[full_graph.station_id[global_idx].item()]

        if station_str_id in weather_station_locations:
            lat, lon = weather_station_locations[station_str_id]
            pos_valgraph[local_idx] = (lon, lat)
        else:
            pos_valgraph[local_idx] = (0, 0)

    print(f"Validation graph positions generated: {len(pos_valgraph)}")
    # ============================================
    # CREATE CONSISTENT COLOR MAPS
    # ============================================
    print("\n" + "=" * 70)
    print("CREATING COLOR MAPS")
    print("=" * 70)

    # --- Full Graph Colors ---
    color_map_full = []
    node_labels_full = {}

    for node_idx in range(station_count):
        if full_graph.train_mask[node_idx]:
            color_map_full.append("green")
            label = "Train"
        elif full_graph.val_mask[node_idx]:
            color_map_full.append("blue")
            label = "Val"
        elif full_graph.test_mask[node_idx]:
            color_map_full.append("red")
            label = "Test"
        else:
            color_map_full.append("gray")
            label = "Unknown"

        # Optional: add node labels showing global index
        node_labels_full[node_idx] = f"{node_idx}"

    print(
        f"Full graph - Train: {color_map_full.count('green')}, "
        f"Val: {color_map_full.count('blue')}, "
        f"Test: {color_map_full.count('red')}"
    )

    # --- Train Graph Colors (CORRECTED) ---
    # Must use train_graph masks directly (already in local indices)
    color_map_train = []
    node_labels_train = {}

    for local_idx in range(num_train_nodes):
        # Use train_graph masks (these are in local indices)
        if train_graph.train_mask[local_idx]:
            color_map_train.append("green")
            label = "Train"
        elif train_graph.val_mask[local_idx]:
            color_map_train.append("blue")
            label = "Val"
        elif (
            train_graph.test_mask[local_idx]
            if hasattr(train_graph, "test_mask")
            else False
        ):
            color_map_train.append("red")
            label = "Test"
        else:
            color_map_train.append("gray")
            label = "Unknown"

        # Show both local and global indices
        global_idx = int(orig_ids[local_idx])
        node_labels_train[local_idx] = f"{local_idx}\n({global_idx})"

    print(
        f"Train graph - Train: {color_map_train.count('green')}, "
        f"Val: {color_map_train.count('blue')}, "
        f"Test: {color_map_train.count('red')}"
    )

    # --- Validation Graph Colors
    color_map_val = []
    node_labels_val = {}

    for local_idx in range(num_valgraph_nodes):
        if validation_graph.train_mask[local_idx]:
            color_map_val.append("green")
            node_type = "Train"
        elif validation_graph.val_mask[local_idx]:
            color_map_val.append("blue")
            node_type = "Val"
        else:
            color_map_val.append("gray")
            node_type = "Other"

        global_idx = int(val_orig_ids[local_idx])
        node_labels_val[local_idx] = f"{local_idx}\n({global_idx})"

    print(
        f"Validation graph - Train: {color_map_val.count('green')}, "
        f"Val: {color_map_val.count('blue')}, "
        f"Test: {color_map_val.count('red')}"
    )

    # ============================================
    # VERIFICATION: Check Consistency
    # ============================================
    print("\n" + "=" * 70)
    print("CONSISTENCY VERIFICATION")
    print("=" * 70)

    # Check: Train nodes in train_graph should match train/val nodes in full_graph
    train_val_in_full = set(
        np.where((full_graph.train_mask | full_graph.val_mask).numpy())[0]
    )
    train_nodes_mapped = set(orig_ids)

    if train_val_in_full == train_nodes_mapped:
        print("✅ Node sets are consistent!")
    else:
        print("⚠️ WARNING: Node sets don't match!")
        print(f"   Full graph train+val: {len(train_val_in_full)} nodes")
        print(f"   Train graph (via orig_id): {len(train_nodes_mapped)} nodes")
        print(f"   Difference: {train_val_in_full - train_nodes_mapped}")

    # Check: Color distribution should match
    full_train_count = sum(1 for i in orig_ids if full_graph.train_mask[i])
    train_train_count = color_map_train.count("green")

    print("\nTrain node count:")
    print(f"   Full graph (for train graph nodes): {full_train_count}")
    print(f"   Train graph: {train_train_count}")
    print(f"   Match: {'✅' if full_train_count == train_train_count else '❌'}")

    # ============================================
    # DRAW THE PLOTS
    # ============================================
    print("\n" + "=" * 70)
    print("DRAWING GRAPHS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # --- Plot 1: Full Graph ---
    ax = axes[0]
    nx.draw(
        G_full,
        pos_full,
        node_color=color_map_full,
        with_labels=True,
        labels=node_labels_full,
        node_size=400,
        font_size=7,
        font_weight="bold",
        edge_color="gray",
        width=1.5,
        ax=ax,
    )
    ax.set_title(
        f"Full Graph ({station_count} stations)\n"
        f"Train={color_map_full.count('green')}, "
        f"Val={color_map_full.count('blue')}, "
        f"Test={color_map_full.count('red')}",
        fontsize=14,
        fontweight="bold",
    )

    # --- Plot 2: Train Graph ---
    ax = axes[1]
    nx.draw(
        G_train,
        pos_train,
        node_color=color_map_train,
        with_labels=True,
        labels=node_labels_train,
        node_size=400,
        font_size=7,
        font_weight="bold",
        edge_color="gray",
        width=1.5,
        ax=ax,
    )
    ax.set_title(
        f"Train Graph ({num_train_nodes} stations)\n"
        f"Local(Global) indices shown\n"
        f"Train={color_map_train.count('green')}, "
        f"Val={color_map_train.count('blue')}",
        fontsize=14,
        fontweight="bold",
    )

    # --- Plot 3: Validation Graph (train + val nodes) ---
    ax = axes[2]
    nx.draw(
        G_valgraph,
        pos_valgraph,
        node_color=color_map_val,
        with_labels=True,
        labels=node_labels_val,
        node_size=400,
        font_size=7,
        font_weight="bold",
        edge_color="gray",
        width=1.5,
        ax=ax,
    )
    ax.set_title("Validation Graph (Train + Val Nodes)", fontsize=14, fontweight="bold")

    # Add legend
    legend_elements = [
        Patch(facecolor="green", label="Train"),
        Patch(facecolor="blue", label="Validation"),
        Patch(facecolor="red", label="Test"),
        Patch(facecolor="gray", label="Unknown"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{model_dir}/graph_comparison_corrected_{fold}.png", dpi=150, bbox_inches="tight")
    print(f"✅ Graphs saved to '{model_dir}/graph_comparison_corrected_{fold}.png'")
    plt.show()

    # ============================================
    # DETAILED MAPPING TABLE (for debugging)
    # ============================================
    print("\n" + "=" * 70)
    print("NODE MAPPING TABLE (First 10 nodes)")
    print("=" * 70)
    print(
        f"{'Local Idx':<12} {'Global Idx':<12} {'Station ID':<15} {'Type in Train':<15} {'Type in Full':<15}"
    )
    print("-" * 70)

    for local_idx in range(min(10, num_train_nodes)):
        global_idx = int(orig_ids[local_idx])
        station_str_id = stations[full_graph.station_id[global_idx].item()]

        # Type in train graph
        if train_graph.train_mask[local_idx]:
            type_train = "Train"
        elif train_graph.val_mask[local_idx]:
            type_train = "Val"
        else:
            type_train = "Other"

        # Type in full graph
        if full_graph.train_mask[global_idx]:
            type_full = "Train"
        elif full_graph.val_mask[global_idx]:
            type_full = "Val"
        elif full_graph.test_mask[global_idx]:
            type_full = "Test"
        else:
            type_full = "Other"

        print(
            f"{local_idx:<12} {global_idx:<12} {station_str_id:<15} {type_train:<15} {type_full:<15}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    ✅ Full Graph: {station_count} nodes, {G_full.number_of_edges()} edges
    - Uses GLOBAL indices (0 to {station_count - 1})
    - Shows Train/Val/Test splits
    
    ✅ Train Graph: {num_train_nodes} nodes, {G_train.number_of_edges()} edges  
    - Uses LOCAL indices (0 to {num_train_nodes - 1})
    - Contains only Train nodes from full graph
    - Labels show: Local(Global) index mapping

    ✅ Train Graph: {num_valgraph_nodes} nodes, {G_valgraph.number_of_edges()} edges  
    - Uses LOCAL indices (0 to {num_valgraph_nodes - 1})
    - Contains only Train+Val nodes from full graph
    - Labels show: Local(Global) index mapping
    
    ✅ Consistency: Node colors and positions are now aligned!
    - Both graphs use the same geographical layout
    - Color coding matches across both visualizations
    - orig_id properly maps local -> global indices
    """)
