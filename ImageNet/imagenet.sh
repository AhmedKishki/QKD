#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Define download and extraction directories
DOWNLOAD_DIR="$SCRIPT_DIR"
EXTRACT_DIR="$SCRIPT_DIR"

# Create directories if they don’t exist
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR/train"
mkdir -p "$EXTRACT_DIR/valid"

# Function to download files with progress bar
download_file() {
    local url=$1
    local output_file=$2

    echo "Downloading $output_file..."
    curl -L --progress-bar -o "$DOWNLOAD_DIR/$output_file" "$url"
    echo "$output_file download complete!"
}

# ImageNet URLs
BASE_URL="https://www.kaggle.com/api/v1/datasets/download/sautkin"
FILES=("imagenet1k0" "imagenet1k1" "imagenet1k2" "imagenet1k3" "imagenet1kvalid")

# Download all files
for file in "${FILES[@]}"; do
    download_file "$BASE_URL/$file" "$file.zip"
done

# Function to extract ZIP files
extract_zip() {
    local zip_file=$1
    local target_dir=$2

    echo "Extracting $zip_file..."
    unzip -q "$DOWNLOAD_DIR/$zip_file" -d "$target_dir"
    echo "$zip_file extracted!"
}

# Extract and merge train files
for file in "imagenet1k0" "imagenet1k1" "imagenet1k2" "imagenet1k3"; do
    extract_zip "$file.zip" "$EXTRACT_DIR/train"
done

# Extract valid separately
extract_zip "imagenet1kvalid.zip" "$EXTRACT_DIR/valid"

echo "ImageNet dataset fully downloaded and extracted at: $EXTRACT_DIR"
