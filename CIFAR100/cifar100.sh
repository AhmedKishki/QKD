#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Define directories
DOWNLOAD_DIR="$SCRIPT_DIR/downloads"
EXTRACT_DIR="$SCRIPT_DIR/extracted"
CIFAR100_TRAIN_DIR="$SCRIPT_DIR/train"
CIFAR100_VALID_DIR="$SCRIPT_DIR/valid"
CIFAR100_URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_ARCHIVE="$DOWNLOAD_DIR/cifar-100-python.tar.gz"

# Create necessary directories
mkdir -p "$DOWNLOAD_DIR"

# Function to download dataset if not already present
download_file() {
    local url=$1
    local output_file=$2

    if [ -f "$output_file" ]; then
        echo "$output_file already exists, skipping download."
    else
        echo "Downloading $output_file..."
        curl -L --progress-bar -o "$output_file" "$url"
        if [ $? -ne 0 ]; then
            echo "Error downloading $output_file"
            exit 1
        fi
        echo "$output_file download complete!"
    fi
}

# Function to extract tar.gz file
extract_tar() {
    local tar_file=$1
    local target_dir=$2

    echo "Extracting $tar_file..."
    mkdir -p "$target_dir"
    tar -xzf "$tar_file" -C "$target_dir"
    if [ $? -ne 0 ]; then
        echo "Error extracting $tar_file"
        exit 1
    fi
    echo "$tar_file extracted!"
}

# Download and extract CIFAR-100
download_file "$CIFAR100_URL" "$CIFAR100_ARCHIVE"
extract_tar "$CIFAR100_ARCHIVE" "$EXTRACT_DIR"
mv "$EXTRACT_DIR/cifar-100-python" "$EXTRACT_DIR/cifar100_data"

# Convert CIFAR-100 dataset to JPEG format
echo "Converting CIFAR-100 dataset to JPEG format..."

python3 - <<EOF
import os
import pickle
import numpy as np
from PIL import Image

data_dir = os.path.join("$EXTRACT_DIR", "cifar100_data")
train_dir = "$CIFAR100_TRAIN_DIR"
valid_dir = "$CIFAR100_VALID_DIR"

# Load labels
with open(os.path.join(data_dir, "meta"), "rb") as f:
    meta = pickle.load(f, encoding="bytes")
    labels = [label.decode("utf-8") for label in meta[b"fine_label_names"]]

# Create label directories
for label in labels:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, label), exist_ok=True)

# Function to process CIFAR batch
def load_cifar_batch(file):
    with open(file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels_list = batch[b"fine_labels"]
    return images, labels_list

# Process training data
images, image_labels = load_cifar_batch(os.path.join(data_dir, "train"))
for i, (img, label) in enumerate(zip(images, image_labels)):
    label_name = labels[label]
    img = Image.fromarray(img)
    img.save(os.path.join(train_dir, label_name, f"train_{i}.jpg"))

# Process test data (valid set)
test_images, test_labels = load_cifar_batch(os.path.join(data_dir, "test"))
for i, (img, label) in enumerate(zip(test_images, test_labels)):
    label_name = labels[label]
    img = Image.fromarray(img)
    img.save(os.path.join(valid_dir, label_name, f"test_{i}.jpg"))

print("CIFAR-100 dataset converted to JPEG and organized into", train_dir, "and", valid_dir)
EOF

# Remove extracted folder
echo "Cleaning up extracted folder..."
rm -rf "$EXTRACT_DIR"

echo "Dataset is ready:"
echo "  Training data: $CIFAR100_TRAIN_DIR"
echo "  Validation data: $CIFAR100_VALID_DIR"
