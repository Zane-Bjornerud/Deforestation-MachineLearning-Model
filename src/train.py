import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import DeforestationDataset
import numpy as np
import os
import segmentation_models_pytorch as smp

# Check for MPS (Metal Performance Shaders) on M1 Mac
# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
#     print("Using MPS device")
# elif torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
#     print("Using CUDA device")
# else:
#     DEVICE = torch.device("cpu")
#     print("Using CPU device")

# Force CPU for compatibility with Focal Loss on Mac M1
DEVICE = torch.device("cpu")
print("Using CPU device (MPS has compatibility issues with some loss functions)")

# Model configuration
IN_CH = 18  # Your data has 18 bands
OUT_CH = 1

# Loss functions
# bce = nn.BCEWithLogitsLoss()
bce = smp.losses.FocalLoss(mode="binary", alpha=0.75, gamma=2.0)


def dice_loss(logits, y, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum(dim=(1, 2, 3))
    den = (p + y).sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


# Training loop
def train_model(train_loader, val_loader, epochs=50):
    best_iou = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            try:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = bce(logits, yb) + dice_loss(logits, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Validation
        model.eval()
        inter = union = val_loss = pred_pos = actual_pos = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                try:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = bce(logits, yb) + dice_loss(logits, yb)
                    val_loss += loss.item()

                    pb = torch.sigmoid(logits) > 0.5
                    yb_bool = yb.bool()
                    inter += (pb & yb_bool).sum().item()
                    union += (pb | yb_bool).sum().item()
                    pred_pos += pb.sum().item()
                    actual_pos += yb_bool.sum().item()
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        iou = inter / max(1, union)
        f1 = 2 * inter / max(1, pred_pos + actual_pos)
        precision = inter / max(1, pred_pos)
        recall = inter / max(1, actual_pos)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, IoU={iou:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}"
        )

        # Save best model
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pth")
            print(f"  New best IoU: {best_iou:.3f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), f"outputs/checkpoints/model_epoch_{epoch}.pth"
            )

    print(f"Training complete! Best IoU: {best_iou:.3f}")


if __name__ == "__main__":
    # Create output directories
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # Paths
    data_dir = "data/processed"
    train_metadata = f"{data_dir}/train_metadata.pkl"
    val_metadata = f"{data_dir}/val_metadata.pkl"
    norm_stats = f"{data_dir}/normalization_stats.pkl"

    # Check if files exist
    if not os.path.exists(train_metadata):
        print(f"Train metadata not found: {train_metadata}")
        print("Please run: python src/split_data.py first")
        exit(1)

    # Create datasets
    print("Loading datasets...")
    train_dataset = DeforestationDataset(
        data_dir, train_metadata, norm_stats, augment=True
    )
    val_dataset = DeforestationDataset(
        data_dir, val_metadata, norm_stats, augment=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("No training samples found!")
        exit(1)

    # Check actual input channels from data
    sample_x, sample_y = train_dataset[0]
    actual_channels = sample_x.shape[0]
    print(f"Actual input channels: {actual_channels}")
    print(f"Sample chip shape: {sample_x.shape}")
    print(f"Sample mask shape: {sample_y.shape}")

    # Update model if needed
    if actual_channels != IN_CH:
        print(f"Updating model input channels from {IN_CH} to {actual_channels}")
        IN_CH = actual_channels

    # Create model and optimizer
    model = smp.Unet(encoder_name="resnet34", in_channels=IN_CH, classes=OUT_CH).to(
        DEVICE
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  # reduced from 1e-3

    # Create data loaders with smaller batch size due to 256x256 images
    try:
        train_loader = DataLoader(
            train_dataset, batch_size=2, shuffle=True, num_workers=0
        )  # num_workers=0 for Mac
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

        # Test loading one batch
        print("Testing data loading...")
        test_batch = next(iter(train_loader))
        print(
            f"Batch loaded successfully: {test_batch[0].shape}, {test_batch[1].shape}"
        )

        # Train model
        print("Starting training...")
        train_model(train_loader, val_loader, epochs=20)  # Start with fewer epochs

    except Exception as e:
        print(f"Error during training setup: {e}")
        import traceback

        traceback.print_exc()
