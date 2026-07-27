# src/visualize_predictions.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import DeforestationDataset
import segmentation_models_pytorch as smp

# Load best model
model = smp.Unet(encoder_name="resnet34", in_channels=18, classes=1)
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth"))
model.eval()

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model = model.to(device)

# Load validation data
val_dataset = DeforestationDataset(
    "data/processed",
    "data/processed/val_metadata.pkl",
    "data/processed/normalization_stats.pkl",
    augment=False,
)

# Visualize 5 predictions
fig, axes = plt.subplots(5, 4, figsize=(16, 20))

for i in range(5):
    chip, true_mask = val_dataset[i]

    # Get prediction
    with torch.no_grad():
        chip_batch = chip.unsqueeze(0).to(device)
        pred_logits = model(chip_batch)
        pred_prob = torch.sigmoid(pred_logits).cpu().numpy()[0, 0]
        pred_mask = (pred_prob > 0.5).astype(float)

    # Find RGB bands for visualization (post-period true color)
    band_names = val_dataset.metadata[i]["band_names"]
    b4_idx = band_names.index("B4_post")
    b3_idx = band_names.index("B3_post")
    b2_idx = band_names.index("B2_post")
    rgb = chip[[b4_idx, b3_idx, b2_idx]].permute(1, 2, 0).numpy()
    rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)

    # Plot
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"RGB Image {i}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(true_mask[0], cmap="Reds", vmin=0, vmax=1)
    axes[i, 1].set_title("True Deforestation")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred_mask, cmap="Reds", vmin=0, vmax=1)
    axes[i, 2].set_title("Predicted Deforestation")
    axes[i, 2].axis("off")

    # Show prediction confidence
    axes[i, 3].imshow(pred_prob, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[i, 3].set_title("Prediction Confidence")
    axes[i, 3].axis("off")

    # Print statistics
    print(f"\nSample {i}:")
    print(
        f"True deforestation: {true_mask.sum().item():.0f} pixels ({true_mask.mean()*100:.2f}%)"
    )
    print(
        f"Predicted deforestation: {pred_mask.sum():.0f} pixels ({pred_mask.mean()*100:.2f}%)"
    )
    print(f"Mean confidence: {pred_prob.mean():.3f}")

plt.tight_layout()
plt.savefig("outputs/predictions_visualization.png", dpi=150, bbox_inches="tight")
print("\nVisualization saved to outputs/predictions_visualization.png")
