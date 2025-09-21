#!/usr/bin/env bash
# scripts/download_data.sh
# Download datasets listed in data/manifest.tsv into data/
# Supports any direct URL (S3, Hugging Face, public HTTP links).
# Optional 3rd column lets you verify SHA256 checksums.

set -euo pipefail

MANIFEST="data/manifest.tsv"
DEST_DIR="data"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest $MANIFEST not found."
  echo "Create it with lines like: <filename>\\t<url>\\t<optional_sha256>"
  echo "Example:"
  echo -e "rondonia_deforest_chips_2021-00001.tfrecord\thttps://example.com/path/file1.tfrecord\t<sha256 optional>"
  exit 1
fi

mkdir -p "$DEST_DIR"

download_file () {
  local fname="$1"
  local url="$2"

  echo "→ Downloading $fname"
  # -L follows redirects; -C - resumes partial downloads if present
  curl -L --fail --retry 3 --retry-delay 3 -C - -o "$DEST_DIR/$fname" "$url"
}

verify_sha256 () {
  local fname="$1"
  local expected="$2"

  if [[ -z "$expected" ]]; then
    return 0
  fi

  if command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$DEST_DIR/$fname" | awk '{print $1}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$DEST_DIR/$fname" | awk '{print $1}')"
  else
    echo "WARN: No sha256 tool found; skipping checksum for $fname"
    return 0
  fi

  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: SHA256 mismatch for $fname"
    echo "Expected: $expected"
    echo "Actual:   $actual"
    exit 2
  else
    echo "✓ SHA256 verified for $fname"
  fi
}

echo "Reading $MANIFEST ..."
# Expect tab-separated: filename<TAB>url<TAB>sha256(optional)
# Skip blank lines and comments starting with '#'
while IFS=$'\t' read -r fname url sha256 || [[ -n "${fname:-}" ]]; do
  # Trim spaces
  fname="${fname//[$'\r\n']}"
  url="${url//[$'\r\n']}"
  sha256="${sha256//[$'\r\n']}"

  [[ -z "$fname" ]] && continue
  [[ "${fname:0:1}" == "#" ]] && continue

  if [[ -f "$DEST_DIR/$fname" ]]; then
    echo "• Found existing $fname — skipping download."
    [[ -n "$sha256" ]] && verify_sha256 "$fname" "$sha256"
    continue
  fi

  download_file "$fname" "$url"
  [[ -n "$sha256" ]] && verify_sha256 "$fname" "$sha256"
done < "$MANIFEST"

echo
echo "All done. Current data/ contents:"
du -h -d 1 "$DEST_DIR" || true
