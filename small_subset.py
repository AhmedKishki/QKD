import os
import shutil
import random

# Paths
cwd = os.getcwd()
imagenet_val_dir = os.path.join(cwd, "ImageNet/valid")  # Path to ImageNet validation dataset
train_small_dir = os.path.join(cwd, "ImageNet/train_small")
val_small_dir = os.path.join(cwd, "ImageNet/valid_small")

# Ensure output directories exist
os.makedirs(train_small_dir, exist_ok=True)
os.makedirs(val_small_dir, exist_ok=True)

# Seed for reproducibility
random.seed(42)

# Iterate over class folders
for class_name in os.listdir(imagenet_val_dir):
    class_path = os.path.join(imagenet_val_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # Skip non-directory files

    # Create class subdirectories in output folders
    train_class_dir = os.path.join(train_small_dir, class_name)
    val_class_dir = os.path.join(val_small_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # List all images in the class
    images = os.listdir(class_path)
    random.shuffle(images)  # Shuffle to ensure random selection

    # Compute split sizes
    split_idx = int(len(images) * 0.8)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    # Copy images to respective folders
    for file in train_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(train_class_dir, file))
    
    for file in val_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(val_class_dir, file))

print("Dataset split complete. Train_small and Val_small created.")
