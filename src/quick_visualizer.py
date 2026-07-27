import numpy as np
import matplotlib.pyplot as plt
import pickle


def visualize_chips():
    """Quick visualization of processed chips."""

    print("=== Quick Data Visualizer ===\n")

    # Load metadata
    with open("data/processed/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"Total chips: {len(metadata)}")

    # Load first few chips
    n_to_show = min(3, len(metadata))

    fig, axes = plt.subplots(n_to_show, 4, figsize=(16, 4 * n_to_show))
    if n_to_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_to_show):
        chip_path = f"data/processed/{metadata[i]['chip_path']}"
        mask_path = f"data/processed/{metadata[i]['mask_path']}"

        chip = np.load(chip_path)  # Shape: (18, 256, 256)
        mask = np.load(mask_path)  # Shape: (1, 256, 256)

        # Find indices for visualization (canonical band names; order varies
        # per metadata file, so always look up by name, never by position)
        band_names = metadata[i]["band_names"]

        # RGB composite (using post-period bands: B4_post, B3_post, B2_post)
        b4_idx = band_names.index("B4_post")  # Red
        b3_idx = band_names.index("B3_post")  # Green
        b2_idx = band_names.index("B2_post")  # Blue

        rgb = np.stack([chip[b4_idx], chip[b3_idx], chip[b2_idx]], axis=2)
        rgb = np.clip(rgb / 3000, 0, 1)  # Normalize for display

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Chip {i+1}: RGB (Post-2021)")
        axes[i, 0].axis("off")

        # NDVI post
        ndvi_post_idx = band_names.index("NDVI_post")
        ndvi_post = chip[ndvi_post_idx]

        im1 = axes[i, 1].imshow(ndvi_post, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 1].set_title("NDVI (Post)")
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        # Change index (dNBR)
        try:
            dnbr_idx = band_names.index("dNBR")
            dnbr = chip[dnbr_idx]
        except ValueError:
            dnbr_idx = 10  # Fallback
            dnbr = chip[dnbr_idx]

        im2 = axes[i, 2].imshow(dnbr, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        axes[i, 2].set_title("dNBR (Change)")
        axes[i, 2].axis("off")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        # Deforestation mask
        im3 = axes[i, 3].imshow(mask[0], cmap="Reds", vmin=0, vmax=1)
        axes[i, 3].set_title("Deforestation Mask")
        axes[i, 3].axis("off")
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

        # Print statistics
        print(f"Chip {i+1}:")
        print(f"  NDVI range: {ndvi_post.min():.3f} - {ndvi_post.max():.3f}")
        print(f"  dNBR range: {dnbr.min():.3f} - {dnbr.max():.3f}")
        print(f"  Mask sum: {mask.sum():.1f} (deforestation pixels)")
        print(f"  Has deforestation: {metadata[i]['has_deforestation']}")

    plt.tight_layout()
    plt.savefig("data/processed/sample_chips.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Check for any subtle deforestation signals
    print(f"\n=== Deforestation Analysis ===")

    # Load several chips and check change indices
    n_check = min(20, len(metadata))
    dnbr_values = []
    dndvi_values = []

    for i in range(n_check):
        chip = np.load(f"data/processed/{metadata[i]['chip_path']}")
        band_names = metadata[i]["band_names"]

        try:
            dnbr_idx = band_names.index("dNBR")
            dndvi_idx = band_names.index("dNDVI")

            dnbr = chip[dnbr_idx]
            dndvi = chip[dndvi_idx]

            dnbr_values.extend(dnbr.flatten())
            dndvi_values.extend(dndvi.flatten())

        except ValueError:
            continue

    if dnbr_values:
        dnbr_values = np.array(dnbr_values)
        dndvi_values = np.array(dndvi_values)

        print(f"Change index statistics (from {n_check} chips):")
        print(f"dNBR - Mean: {dnbr_values.mean():.4f}, Std: {dnbr_values.std():.4f}")
        print(f"       Range: [{dnbr_values.min():.4f}, {dnbr_values.max():.4f}]")
        print(f"dNDVI - Mean: {dndvi_values.mean():.4f}, Std: {dndvi_values.std():.4f}")
        print(f"        Range: [{dndvi_values.min():.4f}, {dndvi_values.max():.4f}]")

        # Check for potential deforestation signals
        # Typically dNBR < -0.1 indicates disturbance
        potential_defor_dnbr = (dnbr_values < -0.1).sum()
        potential_defor_dndvi = (dndvi_values < -0.2).sum()

        print(f"\nPotential deforestation signals:")
        print(
            f"Pixels with dNBR < -0.1: {potential_defor_dnbr} ({potential_defor_dnbr/len(dnbr_values)*100:.2f}%)"
        )
        print(
            f"Pixels with dNDVI < -0.2: {potential_defor_dndvi} ({potential_defor_dndvi/len(dndvi_values)*100:.2f}%)"
        )

        if potential_defor_dnbr == 0 and potential_defor_dndvi == 0:
            print("\nObservation: Very little forest change detected in this area.")
            print("This could mean:")
            print("1. The area was stable in 2020-2021 (good!)")
            print("2. The area chosen doesn't have clear-cut deforestation")
            print("3. You might want to try a different area or time period")

    print(f"\nSample visualization saved to: data/processed/sample_chips.png")


if __name__ == "__main__":
    visualize_chips()
