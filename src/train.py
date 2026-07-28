import argparse
import os

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import yaml
from torch.utils.data import DataLoader

from dataset import DeforestationDataset
from dataset_contract import load_contract

EXPERIMENT_DIR = "configs/experiments"


def load_experiment_config(experiment_id_or_path):
    """Load an experiment config: which dataset contract to train on, where
    to write checkpoints, and hyperparameters. Keeping this separate from the
    dataset contract is what lets hansen_loss and change_based be run and
    compared as two distinct experiments against the same code."""
    if os.path.sep in experiment_id_or_path or experiment_id_or_path.endswith(
        (".yaml", ".yml")
    ):
        path = experiment_id_or_path
    else:
        path = os.path.join(EXPERIMENT_DIR, f"{experiment_id_or_path}.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No experiment config found at {path}. Add one under "
            f"{EXPERIMENT_DIR}/ (see hansen_loss_v1.yaml / change_based_v1.yaml)."
        )

    with open(path) as f:
        return yaml.safe_load(f)


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
IN_CH = 18  # data has 18 bands
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
def train_model(train_loader, val_loader, checkpoint_dir, epochs=50):
    best_iou = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            try:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                if epoch == 0 and batch_idx == 0:
                    print(type(xb))
                    print("Image shape:", xb.shape)
                    print("Image dtype:", xb.dtype)
                    print("Image range:", xb.min().item(), xb.max().item())

                    print("Mask shape:", yb.shape)
                    print("Mask dtype:", yb.dtype)
                    print("Mask values:", torch.unique(yb))

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
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print(f"  New best IoU: {best_iou:.3f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch}.pth"
            )

    print(f"Training complete! Best IoU: {best_iou:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the deforestation model on a specific experiment"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment id under configs/experiments/ (e.g. hansen_loss_v1 "
        "or change_based_v1), or an explicit path to an experiment yaml. "
        "Each experiment pins a dataset contract, checkpoint dir, and "
        "hyperparameters, so hansen_loss and change_based runs stay "
        "separate and comparable.",
    )
    args = parser.parse_args()

    experiment = load_experiment_config(args.experiment)
    contract = load_contract(experiment["dataset_id"])

    print(f"=== Experiment: {experiment['experiment_id']} ===")
    print(f"Dataset: {contract.dataset_id} (label_mode={contract.label_mode})")
    print(f"Processed data: {contract.processed_path}")

    # Create output directories
    checkpoint_dir = experiment["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Paths
    data_dir = contract.processed_path
    train_metadata = f"{data_dir}/train_metadata.pkl"
    val_metadata = f"{data_dir}/val_metadata.pkl"
    norm_stats = f"{data_dir}/normalization_stats.pkl"

    # Check if files exist
    if not os.path.exists(train_metadata):
        print(f"Train metadata not found: {train_metadata}")
        print(
            f"Please run: python src/split_data.py --dataset-id {contract.dataset_id} first"
        )
        exit(1)

    # Create datasets
    print("Loading datasets...")
    train_dataset = DeforestationDataset(
        data_dir, train_metadata, norm_stats, contract, augment=True
    )
    val_dataset = DeforestationDataset(
        data_dir, val_metadata, norm_stats, contract, augment=False
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
    opt = torch.optim.AdamW(
        model.parameters(), lr=experiment.get("learning_rate", 1e-4)
    )

    # Create data loaders with smaller batch size due to 256x256 images
    batch_size = experiment.get("batch_size", 2)
    try:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )  # num_workers=0 for Mac
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Test loading one batch
        print("Testing data loading...")
        test_batch = next(iter(train_loader))
        print(
            f"Batch loaded successfully: {test_batch[0].shape}, {test_batch[1].shape}"
        )

        # Train model
        print("Starting training...")
        train_model(
            train_loader,
            val_loader,
            checkpoint_dir,
            epochs=experiment.get("epochs", 20),
        )

    except Exception as e:
        print(f"Error during training setup: {e}")
        import traceback

        traceback.print_exc()
