def save_sampling_results(results, filename="sampling_results.txt"):
    """Save sampling results to a text file."""

    with open(filename, "w") as f:
        f.write("K-MEANS STRATIFIED SPATIAL SAMPLING RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("SHARED TEST SET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of stations: {len(results['test_stations'])}\n")
        f.write("Station IDs:\n")
        for station_id in sorted(results["test_stations"]):
            f.write(f"  {station_id}\n")

        f.write("\n" + "=" * 70 + "\n\n")

        f.write("STATISTICAL METHOD (90% Train / 10% Test)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Training stations: {len(results['statistical']['train'])}\n")
        f.write(f"Test stations: {len(results['statistical']['test'])}\n\n")

        f.write("Training Station IDs:\n")
        for station_id in sorted(results["statistical"]["train"]):
            f.write(f"  {station_id}\n")

        f.write("\n" + "=" * 70 + "\n\n")

        f.write("MACHINE LEARNING METHOD (70% Train / 20% Val / 10% Test)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Training stations: {len(results['ml']['train'])}\n")
        f.write(f"Validation stations: {len(results['ml']['validation'])}\n")
        f.write(f"Test stations: {len(results['ml']['test'])}\n\n")

        f.write("Training Station IDs:\n")
        for station_id in sorted(results["ml"]["train"]):
            f.write(f"  {station_id}\n")

        f.write("\nValidation Station IDs:\n")
        for station_id in sorted(results["ml"]["validation"]):
            f.write(f"  {station_id}\n")

    print(f"Results saved to '{filename}'")
