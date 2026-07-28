#!/usr/bin/env bash
# scripts/setup_gdrive_rclone.sh
#
# Configure an rclone remote for Google Drive and sync GEE export shards
# from Drive into this repo's data/raw/<dataset_id>/ directory. Meant to
# replace manually downloading/unzipping shards from the Drive web UI,
# especially for large exports like gee_full_gfc_v1.
#
# Usage:
#   scripts/setup_gdrive_rclone.sh <dataset_id> [drive_folder]
#
# <dataset_id> must have a contract at configs/datasets/<dataset_id>.yaml
#   (supplies raw_path and target_year, which combine into the export's
#   file prefix -- see FILE_PREFIX in scripts/gee_export_chips.py).
# [drive_folder] defaults to "deforest_export", matching DRIVE_FOLDER in
#   scripts/gee_export_chips.py.

set -euo pipefail

REMOTE_NAME="gdrive"
DATASET_ID="${1:?Usage: $0 <dataset_id> [drive_folder]}"
DRIVE_FOLDER="${2:-deforest_export}"

CONTRACT="configs/datasets/${DATASET_ID}.yaml"
if [[ ! -f "$CONTRACT" ]]; then
  echo "No contract found at $CONTRACT" >&2
  exit 1
fi

RAW_PATH=$(grep '^raw_path:' "$CONTRACT" | sed 's/^raw_path:[[:space:]]*//')
TARGET_YEAR=$(grep '^target_year:' "$CONTRACT" | sed 's/^target_year:[[:space:]]*//')
FILE_PREFIX="${DATASET_ID}_${TARGET_YEAR}"

echo "Dataset:      $DATASET_ID"
echo "Raw path:     $RAW_PATH"
echo "Drive folder: $DRIVE_FOLDER"
echo "File prefix:  $FILE_PREFIX"
echo

# 1. Install rclone if missing
if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found."
  if command -v brew >/dev/null 2>&1; then
    echo "Installing via Homebrew..."
    brew install rclone
  else
    echo "Homebrew not found -- install rclone manually: https://rclone.org/install/"
    exit 1
  fi
fi

# 2. Configure the Google Drive remote (interactive: opens a browser for OAuth)
if ! rclone listremotes | grep -q "^${REMOTE_NAME}:$"; then
  echo "No '$REMOTE_NAME' remote configured yet."
  echo "This launches rclone's interactive setup. It will open a browser"
  echo "window for you to log into the Google account that owns the Drive"
  echo "folder and grant rclone access. When prompted:"
  echo "  - storage type: search for 'drive' (Google Drive), pick its number"
  echo "  - client_id / client_secret: leave blank (press Enter) for rclone's defaults"
  echo "  - scope: 1 (full access) or 2 (read-only) -- 2 is enough for downloading only"
  echo "  - root_folder_id / service_account_file: leave blank"
  echo "  - Edit advanced config: n"
  echo "  - Use auto config: y (opens browser)"
  echo "  - Configure as team drive: n (unless the folder is actually a Shared Drive)"
  echo
  rclone config create "$REMOTE_NAME" drive
else
  echo "Remote '$REMOTE_NAME:' already configured, skipping setup."
fi

# 3. Copy matching shards down
mkdir -p "$RAW_PATH"
echo
echo "Copying ${FILE_PREFIX}* from ${REMOTE_NAME}:${DRIVE_FOLDER}/ into ${RAW_PATH}/ ..."
rclone copy "${REMOTE_NAME}:${DRIVE_FOLDER}" "$RAW_PATH" \
  --include "${FILE_PREFIX}*" \
  --progress

echo
echo "Done. Files in ${RAW_PATH}:"
ls -la "$RAW_PATH"
