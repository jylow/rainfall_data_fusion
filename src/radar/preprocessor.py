import pandas as pd
from pathlib import Path
import rasterio
from rasterio.windows import Window
from datetime import datetime
import tqdm
from typing import Tuple, List
import numpy as np
from scipy.spatial import cKDTree

from src.raingauge.utils import load_weather_station_dataset

BOUNDS_SINGAPORE = {"left": 103.6, "right": 104.1, "top": 1.5, "bottom": 1.188}


class RadarPreprocessor:
    def __init__(
        self,
        radar_base_path: str = "database/sg_radar_data",
        output_path: str = "database/sg_radar_data_cropped",
        weather_station_df: pd.DataFrame = None,
    ):
        """
        Initialize radar preprocessor.

        Args:
            radar_base_path: Base path to radar data
            output_path: Path to save cropped radar images
            weather_station_df: Weather station dataframe with time_sgt index
        """
        self.radar_base_path = Path(radar_base_path)
        self.output_path = Path(output_path)
        self.weather_station_df = weather_station_df

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    def parse_radar_filename(self, filename: str) -> datetime:
        """
        Parse radar filename to extract timestamp.

        Example: spacesync_99_202502010015_area_23.tif -> 2025-02-01 00:15:00

        Args:
            filename: Radar filename

        Returns:
            datetime object
        """
        try:
            # Extract datetime string from filename
            parts = filename.split("_")
            datetime_str = parts[2]  # 202502010015

            # Parse: YYYYMMDDHHMMSS format
            year = int(datetime_str[0:4])
            month = int(datetime_str[4:6])
            day = int(datetime_str[6:8])
            hour = int(datetime_str[8:10])
            minute = int(datetime_str[10:12])

            return datetime(year, month, day, hour, minute)
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return None

    def get_all_radar_files(self) -> List[Tuple[datetime, Path]]:
        """
        Scan radar directory and get all radar files with their timestamps.

        Returns:
            List of (timestamp, filepath) tuples
        """
        radar_files = []

        # Scan all date directories
        for date_dir in self.radar_base_path.iterdir():
            if not date_dir.is_dir():
                continue

            # Scan all .tif files in date directory
            for radar_file in date_dir.glob("spacesync_99_*.tif"):
                timestamp = self.parse_radar_filename(radar_file.name)
                if timestamp:
                    radar_files.append((timestamp, radar_file))

        # Sort by timestamp
        radar_files.sort(key=lambda x: x[0])

        return radar_files

    def match_with_weather_data(
        self,
        radar_files: List[Tuple[datetime, Path]],
        unmatched_radar_csv_path: str = "database/unmatched_radar_files.csv",
        unmatched_weather_csv_path: str = "database/unmatched_weather_files.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Match radar files with weather station timestamps.
        Also identifies unmatched radar files and saves them to a CSV.

        Args:
            radar_files: List of (timestamp, filepath) tuples.
            unmatched_radar_csv_path: Path to save the CSV of unmatched radar files.
            unmatched_weather_csv_path: Path to save the CSV of unmatched weather files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - matched_weather_df: The weather_station_df filtered to matched timestamps.
                - matched_radar_df: DataFrame with matched radar timestamps and filepaths.
        """
        if self.weather_station_df is None:
            raise ValueError("Weather station dataframe not provided")

        # Get weather station timestamps and set up for merge
        weather_timestamps = pd.to_datetime(self.weather_station_df.index)
        weather_ts_df = pd.DataFrame(
            {
                "timestamp": weather_timestamps,
                "weather_idx": range(len(weather_timestamps)),
            }
        )

        # Create dataframe of radar files
        radar_df = pd.DataFrame(radar_files, columns=["timestamp", "filepath"])
        radar_df["timestamp"] = pd.to_datetime(radar_df["timestamp"])

        # Find matching timestamps (inner join)
        # This join_df contains matched timestamps, filepaths, and weather indices
        join_df = pd.merge(
            weather_ts_df,
            radar_df,
            on="timestamp",
            how="inner",
        )

        # --- New: Find, display, and save unmatched files ---

        # Find radar files whose timestamps are NOT in the successful join_df
        unmatched_radar_df = radar_df[~radar_df["timestamp"].isin(join_df["timestamp"])]
        unmatched_weather_df = weather_ts_df[
            ~weather_ts_df["timestamp"].isin(join_df["timestamp"])
        ]

        print("\n=== Unmatched Radar Files ===")
        if unmatched_radar_df.empty:
            print("All radar files were matched.")
        else:
            print(f"Found {len(unmatched_radar_df)} unmatched radar files:")
            # Display the unmatched files
            print(unmatched_radar_df.to_string())

            # Save to CSV
            try:
                print(
                    f"\nSaving unmatched radar files to {unmatched_radar_csv_path}..."
                )
                unmatched_radar_df.to_csv(unmatched_radar_csv_path, index=False)
                print(f"Successfully saved to {unmatched_radar_csv_path}.")
            except Exception as e:
                print(f"Error saving unmatched files to CSV: {e}")

        print("\n=== Unmatched Weather Files ===")
        if unmatched_weather_df.empty:
            print("All weather files were matched.")
        else:
            print(f"Found {len(unmatched_weather_df)} unmatched weather files:")
            # Display the unmatched files
            print(unmatched_weather_df.to_string())

            # Save to CSV
            try:
                print(
                    f"\nSaving unmatched radar files to {unmatched_weather_csv_path}..."
                )
                unmatched_weather_df.to_csv(unmatched_weather_csv_path, index=False)
                print(f"Successfully saved to {unmatched_weather_csv_path}.")
            except Exception as e:
                print(f"Error saving unmatched files to CSV: {e}")

        # --- Original Statistics ---
        print("\n=== Matching Statistics ===")
        print(f"Weather station timestamps: {len(weather_timestamps)}")
        print(f"Radar files found: {len(radar_df)}")
        print(f"Matched timestamps: {len(join_df)}")

        match_rate = 0.0
        if len(weather_timestamps) > 0:
            match_rate = len(join_df) / len(weather_timestamps) * 100
            print(f"Match rate (against weather data): {match_rate:.2f}%")
        else:
            print("Match rate (against weather data): N/A (no weather data)")

        # --- New: Prepare and return matched DataFrames ---

        # Get the matched weather data using the original indices
        matched_weather_df = self.weather_station_df.iloc[join_df["weather_idx"]]

        # Get the matched radar data
        matched_radar_df = join_df

        return matched_radar_df, matched_weather_df

    def crop_radar_image(self, input_path: Path, output_path: Path) -> bool:
        """
        Crop radar image to Singapore bounds.

        Args:
            input_path: Path to input radar .tif file
            output_path: Path to save cropped .tif file

        Returns:
            True if successful, False otherwise
        """
        try:
            with rasterio.open(input_path) as src:
                # Get original bounds
                bounds = src.bounds
                transform = src.transform

                # Convert geographic coordinates to pixel coordinates
                left_col, top_row = ~transform * (
                    BOUNDS_SINGAPORE["left"],
                    BOUNDS_SINGAPORE["top"],
                )
                right_col, bottom_row = ~transform * (
                    BOUNDS_SINGAPORE["right"],
                    BOUNDS_SINGAPORE["bottom"],
                )

                # Ensure integers and proper ordering
                left_col = int(max(0, left_col))
                right_col = int(min(src.width, right_col))
                top_row = int(max(0, top_row))
                bottom_row = int(min(src.height, bottom_row))

                # Read the windowed data
                window = Window.from_slices(
                    (top_row, bottom_row), (left_col, right_col)
                )

                data = src.read(1, window=window)

                # Update transform for cropped image
                window_transform = src.window_transform(window)

                # Write cropped image
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs=src.crs,
                    transform=window_transform,
                    compress="lzw",
                ) as dst:
                    dst.write(data, 1)

                return True

        except Exception as e:
            print(f"Error cropping {input_path}: {e}")
            return False

    def process_all_matched_files(self, matched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and crop all matched radar files.

        Args:
            matched_df: DataFrame with matched timestamps and filepaths

        Returns:
            DataFrame with processed file information
        """
        results = []

        print(f"\n=== Processing {len(matched_df)} radar images ===")

        for idx, row in tqdm.tqdm(
            matched_df.iterrows(), total=len(matched_df), desc="Cropping radar images"
        ):
            timestamp = row["timestamp"]
            input_path = row["filepath"]

            # Create output path maintaining date directory structure
            date_str = timestamp.strftime("%Y%m%d")
            output_filename = f"cropped_{input_path.name}"
            output_path = self.output_path / date_str / output_filename

            # Crop image
            success = self.crop_radar_image(input_path, output_path)

            results.append(
                {
                    "timestamp": timestamp,
                    "weather_idx": row["weather_idx"],
                    "input_path": str(input_path),
                    "output_path": str(output_path) if success else None,
                    "success": success,
                }
            )

        results_df = pd.DataFrame(results)

        # Print summary
        success_count = results_df["success"].sum()
        print("\n=== Processing Complete ===")
        print(f"Successfully processed: {success_count}/{len(results_df)}")
        print(f"Failed: {len(results_df) - success_count}")

        return results_df

    def save_metadata(
        self,
        results_df: pd.DataFrame,
        filename: str = "radar_preprocessing_metadata.csv",
    ):
        """
        Save preprocessing metadata to CSV.

        Args:
            results_df: Results dataframe
            filename: Output filename
        """
        output_file = self.output_path / filename
        results_df.to_csv(output_file, index=False)
        print(f"\nMetadata saved to: {output_file}")

    def create_radar_grid_nodes(self, radar_df, row_idx=0):
        """
        Convert radar data to grid nodes with coordinates.

        Parameters:
        -----------
        radar_df : pd.DataFrame
            DataFrame containing radar data with 'data', 'bounds', and 'transform' columns
        row_idx : int
            Index of the radar frame to process

        Returns:
        --------
        grid_features : np.ndarray
            Array of shape (n_nodes, feature_dim) containing radar intensities
        grid_coords : np.ndarray
            Array of shape (n_nodes, 2) containing (lat, lon) coordinates
        grid_shape : tuple
            Original (height, width) of the radar grid
        """
        radar_data = radar_df.iloc[row_idx]["data"]
        bounds = radar_df.iloc[row_idx]["bounds"]
        transform = radar_df.iloc[row_idx]["transform"]

        height, width = radar_data.shape

        # Create coordinate arrays for each grid point
        # bounds typically: (min_x, min_y, max_x, max_y) or (left, bottom, right, top)
        if isinstance(bounds, (list, tuple)):
            min_x, min_y, max_x, max_y = bounds
        else:
            # If bounds is from rasterio, extract values
            min_x, min_y, max_x, max_y = (
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )

        # Create grid of coordinates
        x_coords = np.linspace(min_x, max_x, width)
        y_coords = np.linspace(min_y, max_y, height)

        # Create meshgrid
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Flatten to create node list
        grid_coords = np.stack([yy.flatten(), xx.flatten()], axis=1)  # (lat, lon)
        grid_features = radar_data.flatten().reshape(
            -1, 1
        )  # Reshape for feature matrix

        return grid_features, grid_coords, (height, width)

    def prepare_radar_features_temporal(self, radar_df, weather_station_df_pivot):
        """
        Prepare radar grid features aligned with weather station temporal resolution.

        Parameters:
        -----------
        radar_df : pd.DataFrame
            DataFrame with radar data indexed by time_sgt
        weather_station_df_pivot : pd.DataFrame
            Pivoted weather station data with datetime index

        Returns:
        --------
        radar_features : np.ndarray
            Array of shape (n_timesteps, n_grid_nodes, n_features)
        grid_coords : np.ndarray
            Array of shape (n_grid_nodes, 2) with (lat, lon)
        grid_shape : tuple
            Original (height, width) of radar grid
        """
        # Ensure radar_df is sorted by time
        radar_df = radar_df.sort_values("time_sgt").reset_index(drop=True)

        # Get the time index from weather stations
        time_index = weather_station_df_pivot.index

        # Initialize list to store radar data for each timestep
        radar_temporal_data = []

        # Get grid coordinates from first radar image
        first_radar = radar_df.iloc[0]
        bounds = first_radar["bounds"]
        radar_data = first_radar["data"]
        height, width = radar_data.shape

        # Extract bounds
        if isinstance(bounds, (list, tuple)):
            min_x, min_y, max_x, max_y = bounds
        else:
            min_x, min_y, max_x, max_y = (
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )

        # Create coordinate grid
        x_coords = np.linspace(min_x, max_x, width)
        y_coords = np.linspace(min_y, max_y, height)
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_coords = np.stack([yy.flatten(), xx.flatten()], axis=1)

        # Align radar data with weather station timestamps
        for timestamp in time_index:
            # Find closest radar image
            time_diffs = np.abs((radar_df["time_sgt"] - timestamp).dt.total_seconds())
            closest_idx = time_diffs.argmin()

            # Only use if within reasonable time window (e.g., 15 minutes)
            if time_diffs.iloc[closest_idx] <= 900:  # 15 minutes in seconds
                radar_image = radar_df.iloc[closest_idx]["data"]
            else:
                # Use NaN or zeros if no close match
                radar_image = np.zeros_like(radar_df.iloc[0]["data"])

            # Flatten and reshape
            radar_flat = radar_image.flatten().reshape(-1, 1)
            radar_temporal_data.append(radar_flat)

        # Stack to create (n_timesteps, n_nodes, n_features)
        radar_features = np.stack(radar_temporal_data, axis=0)
        radar_features_normalized = (radar_features - np.min(radar_features)) / (
            np.max(radar_features) - np.min(radar_features)
        )

        return radar_features_normalized, grid_coords, (height, width)

    def connect_radar_to_stations(self, grid_coords, station_coords, radius_km=1.0):
        """
        Create edges between radar grid nodes and weather stations within a radius.

        Parameters:
        -----------
        grid_coords : np.ndarray
            Array of shape (n_grid_nodes, 2) with (lat, lon) of radar grid points
        station_coords : np.ndarray
            Array of shape (n_stations, 2) with (lat, lon) of weather stations
        radius_km : float
            Connection radius in kilometers

        Returns:
        --------
        edge_index : np.ndarray
            Array of shape (2, n_edges) for PyG format
        edge_distances : np.ndarray
            Array of shape (n_edges,) containing distances in km
        """
        # Build KD-tree for efficient spatial queries
        # Convert lat/lon to approximate distances (rough conversion)
        # For more accuracy, use proper geodesic distance
        tree = cKDTree(station_coords)

        # Query for all grid points within radius
        # Convert km to degrees (approximate: 1 degree ≈ 111 km at equator)
        radius_deg = radius_km / 111.0

        # Find all station neighbors for each grid point
        neighbors = tree.query_ball_point(grid_coords, r=radius_deg)

        # Build edge list
        radar_indices = []
        station_indices = []
        distances = []

        for grid_idx, station_list in enumerate(neighbors):
            for station_idx in station_list:
                # Calculate actual distance
                dist = np.linalg.norm(
                    (grid_coords[grid_idx] - station_coords[station_idx]) * 111.0
                )
                if dist <= radius_km:
                    radar_indices.append(grid_idx)
                    station_indices.append(station_idx)
                    distances.append(dist)

        # Convert to PyG edge_index format [2, num_edges]
        edge_index = np.array([radar_indices, station_indices])
        edge_distances = np.array(distances)

        return edge_index, edge_distances

    def haversine_distance(self, coord1, coord2):
        """
        Calculate haversine distance between coordinates (more accurate for lat/lon).

        Parameters:
        -----------
        coord1, coord2 : np.ndarray
            Arrays of shape (n, 2) with (lat, lon) in degrees

        Returns:
        --------
        distances : np.ndarray
            Distances in kilometers
        """
        lat1, lon1 = np.radians(coord1[:, 0]), np.radians(coord1[:, 1])
        lat2, lon2 = np.radians(coord2[:, 0]), np.radians(coord2[:, 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in km
        r = 6371

        return c * r

    def create_grid_edges_radius(self, grid_coords, radius_km):
        """
        Create edges between radar grid cells within a specified radius.

        Parameters:
        -----------
        grid_coords : np.ndarray
            Array of shape (n_nodes, 2) with (lat, lon) coordinates
        radius_km : float
            Connection radius in kilometers

        Returns:
        --------
        edges : np.ndarray
            Edge index array of shape (2, n_edges)
        """
        # Converts degree to km, approximately 111km per degree of latitudes
        radius_deg = radius_km / 111.0

        # Build KD-tree for efficient spatial queries
        tree = cKDTree(grid_coords)

        edges_src = []
        edges_dst = []

        # Find neighbors within radius for each grid node
        neighbors_list = tree.query_ball_point(grid_coords, r=radius_deg)

        for node_idx, neighbor_indices in enumerate(neighbors_list):
            for neighbor_idx in neighbor_indices:
                # Skip self-loops
                if neighbor_idx == node_idx:
                    continue

                # Calculate actual distance
                dist = np.linalg.norm(
                    (grid_coords[node_idx] - grid_coords[neighbor_idx]) * 111.0
                )

                if dist <= radius_km:
                    edges_src.append(node_idx)
                    edges_dst.append(neighbor_idx)

        # Return edges or empty array if no edges found
        if len(edges_src) == 0:
            return np.array([[], []], dtype=np.int64)

        return np.array([edges_src, edges_dst], dtype=np.int64)


def main():
    """
    Main preprocessing pipeline.
    """
    print("=== Radar Image Preprocessing ===\n")

    # Load weather station data
    # Assuming weather_station_df_pivot is available from your main script
    # You'll need to load this from your saved data or regenerate it
    try:
        weather_station_data = load_weather_station_dataset("weather_station_data.csv")

        cols = list(weather_station_data.columns)
        cols.remove("time_sgt")
        cols.remove("gid")

        weather_station_df_pivot = (
            pd.pivot(
                data=weather_station_data, index="time_sgt", columns="gid", values=cols
            )
            .resample("15min")
            .first()
        )

        print(
            f"Loaded weather station data: {weather_station_df_pivot.shape[0]} timestamps"
        )

    except Exception as e:
        print(f"Error loading weather station data: {e}")
        print("Please ensure weather_station_data.csv is available")
        return

    # Initialize preprocessor
    preprocessor = RadarPreprocessor(
        radar_base_path="database/sg_radar_data",
        output_path="database/sg_radar_data_cropped",
        weather_station_df=weather_station_df_pivot,
    )

    # Step 1: Get all radar files
    print("\nStep 1: Scanning radar files...")
    radar_files = preprocessor.get_all_radar_files()

    if len(radar_files) == 0:
        print("No radar files found! Please check the radar_base_path.")
        return

    print(f"Found {len(radar_files)} radar files")
    print(f"Date range: {radar_files[0][0]} to {radar_files[-1][0]}")

    # Step 2: Match with weather data
    print("\nStep 2: Matching with weather station data...")
    matched_df = preprocessor.match_with_weather_data(radar_files)

    if len(matched_df) == 0:
        print("No matching timestamps found!")
        return

    # Step 3: Process and crop all matched files
    print("\nStep 3: Cropping radar images to Singapore bounds...")
    results_df = preprocessor.process_all_matched_files(matched_df)

    # Step 4: Save metadata
    preprocessor.save_metadata(results_df)

    print("\n=== Preprocessing Complete ===")
    print(f"Cropped radar images saved to: {preprocessor.output_path}")


if __name__ == "__main__":
    main()
