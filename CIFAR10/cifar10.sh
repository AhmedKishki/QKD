#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Define download and extraction directories
DOWNLOAD_DIR="$SCRIPT_DIR/downloads"
EXTRACT_DIR="$SCRIPT_DIR/extracted"
TRAIN_DIR="$SCRIPT_DIR/train"
VALID_DIR="$SCRIPT_DIR/valid"

# CIFAR-10 Python version URL
CIFAR10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE="$DOWNLOAD_DIR/cifar-10-python.tar.gz"

# Create necessary directories
mkdir -p "$DOWNLOAD_DIR" "$TRAIN_DIR" "$VALID_DIR"

# Function to download CIFAR-10 if not already present
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

# Download and extract CIFAR-10
download_file "$CIFAR10_URL" "$CIFAR10_ARCHIVE"
extract_tar "$CIFAR10_ARCHIVE" "$EXTRACT_DIR"

# Move extracted dataset to a known folder
mv "$EXTRACT_DIR/cifar-10-batches-py" "$EXTRACT_DIR/cifar10_data"

echo "CIFAR-10 dataset downloaded and extracted."
echo "Converting dataset to JPEG format..."

# Run Python script for conversion
python3 - <<EOF
import os
import pickle
import numpy as np
from PIL import Image

# Define paths
data_dir = os.path.join("$EXTRACT_DIR", "cifar10_data")
train_dir = "$TRAIN_DIR"
valid_dir = "$VALID_DIR"

# CIFAR-10 label names
labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Function to load CIFAR-10 batch
def load_cifar_batch(file):
    with open(file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels_list = batch[b"labels"]
    return images, labels_list

# Create label directories in train/ and valid/
for label in labels:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, label), exist_ok=True)

# Load training batches and save images
train_files = [f"data_batch_{i}" for i in range(1, 6)]

for file in train_files:
    images, image_labels = load_cifar_batch(os.path.join(data_dir, file))
    
    for i, (img, label) in enumerate(zip(images, image_labels)):
        label_name = labels[label]
        img = Image.fromarray(img)
        img.save(os.path.join(train_dir, label_name, f"{file}_{i}.jpg"))

# Load test batch and save images to valid/
test_images, test_labels = load_cifar_batch(os.path.join(data_dir, "test_batch"))

for i, (img, label) in enumerate(zip(test_images, test_labels)):
    label_name = labels[label]
    img = Image.fromarray(img)
    img.save(os.path.join(valid_dir, label_name, f"test_{i}.jpg"))

print("CIFAR-10 dataset converted to JPEG and organized into train/ and valid/ folders.")
EOF

# Remove extracted folder
echo "Cleaning up extracted folder..."
rm -rf "$EXTRACT_DIR"

echo "Dataset is ready in:"
echo "  Training data: $TRAIN_DIR"
echo "  Validation data: $VALID_DIR"
