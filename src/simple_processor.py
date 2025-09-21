import numpy as np
import rasterio
from rasterio.windows import Window
import os
import pickle
from pathlib import Path


def check_files():
    """Check what files are available in the data directory."""
    print("Checking data directory...")
    data_dir = Path("data")

    if not data_dir.exists():
        print("Data directory doesn't exist. Creating it...")
        data_dir.mkdir()
        return []

    files = list(data_dir.glob("*"))
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    return files


def inspect_single_file(file_path):
    """Inspect a single GeoTIFF file without using too many resources."""
    print(f"\nInspecting: {file_path}")

    try:
        with rasterio.open(file_path) as src:
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Bands: {src.count}")
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  CRS: {src.crs}")

            # Read a tiny sample
            sample_size = min(50, src.width, src.height)
            sample = src.read(window=Window(0, 0, sample_size, sample_size))

            print(f"  Sample shape: {sample.shape}")
            print(
                f"  Sample min/max: {np.nanmin(sample):.3f} / {np.nanmax(sample):.3f}"
            )
            print(f"  Has NaN: {np.isnan(sample).any()}")

            return True
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False


def create_single_chip_test(stack_path, labels_path):
    """Create just one chip as a test."""
    print(f"\nTesting with one chip from:")
    print(f"  Stack: {stack_path}")
    print(f"  Labels: {labels_path}")

    try:
        with rasterio.open(stack_path) as stack_src, rasterio.open(
            labels_path
        ) as labels_src:

            # Read a 256x256 chip from the center
            center_x = stack_src.width // 2
            center_y = stack_src.height // 2

            window = Window(center_x - 128, center_y - 128, 256, 256)

            print(f"Reading window: {window}")

            chip_data = stack_src.read(window=window)
            label_data = labels_src.read(window=window)

            print(f"Chip shape: {chip_data.shape}")
            print(f"Label shape: {label_data.shape}")
            print(f"Chip data type: {chip_data.dtype}")
            print(f"Label data type: {label_data.dtype}")

            # Check for valid data
            print(f"Chip has NaN: {np.isnan(chip_data).any()}")
            print(f"Label has NaN: {np.isnan(label_data).any()}")
            print(
                f"Chip range: {np.nanmin(chip_data):.3f} to {np.nanmax(chip_data):.3f}"
            )
            print(
                f"Label range: {np.nanmin(label_data):.3f} to {np.nanmax(label_data):.3f}"
            )

            # Check if there's any deforestation in this chip
            has_defor = np.sum(label_data > 0) > 0
            print(f"Has deforestation: {has_defor}")

            # Save test chip
            os.makedirs("data/test", exist_ok=True)
            np.save("data/test/test_chip.npy", chip_data.astype(np.float32))
            np.save("data/test/test_label.npy", label_data.astype(np.float32))

            print("Test chip saved to data/test/")
            return True

    except Exception as e:
        print(f"Error creating test chip: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to run step by step."""
    print("=== Simple GeoTIFF Processor ===\n")

    # Step 1: Check what files we have
    files = check_files()

    if not files:
        print("\nNo files found. Please:")
        print("1. Go to your Google Earth Engine tasks")
        print("2. Download the exported files")
        print("3. Place them in the 'data' directory")
        return

    # Step 2: Look for GeoTIFF files
    tif_files = [f for f in files if f.suffix.lower() in [".tif", ".tiff"]]

    if not tif_files:
        print("\nNo GeoTIFF files found. Available files:")
        for f in files:
            print(f"  {f.name}")
        return

    print(f"\nFound {len(tif_files)} GeoTIFF files:")
    for f in tif_files:
        print(f"  {f.name}")

    # Step 3: Inspect each file
    valid_files = []
    for tif_file in tif_files:
        if inspect_single_file(tif_file):
            valid_files.append(tif_file)

    if len(valid_files) < 2:
        print(
            f"\nNeed at least 2 valid files (stack + labels), found {len(valid_files)}"
        )
        return

    # Step 4: Try to identify stack and labels
    # Usually the larger file is the stack (more bands)
    print(
        f"\nTrying to identify stack and labels from {len(valid_files)} valid files..."
    )

    # For now, let's assume:
    # - The first file is the stack
    # - The second file is the labels (or vice versa)

    if len(valid_files) >= 2:
        stack_path = valid_files[0]
        labels_path = valid_files[1]

        print(f"Assuming:")
        print(f"  Stack: {stack_path.name}")
        print(f"  Labels: {labels_path.name}")

        # Test with one chip
        success = create_single_chip_test(stack_path, labels_path)

        if success:
            print("\n✅ Test successful! You can now run the full processor.")
            print("Edit the file paths in the main processor and run it.")
        else:
            print("\n❌ Test failed. Check the file paths and formats.")


if __name__ == "__main__":
    main()
