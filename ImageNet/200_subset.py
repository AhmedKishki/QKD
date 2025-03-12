import os
import shutil
import random

# Paths
cwd = os.getcwd()
train_dir = os.path.join(cwd, "ImageNet/train")
train200_dir = os.path.join(cwd, "ImageNet/train200")

# Ensure output directory exists
os.makedirs(train200_dir, exist_ok=True)

# Seed for reproducibility
random.seed(42)

# Iterate over class folders
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # Skip non-directory files

    # Create class subdirectory in train200
    train200_class_dir = os.path.join(train200_dir, class_name)
    os.makedirs(train200_class_dir, exist_ok=True)

    # List all .jpg images in the class
    images = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]

    # Select 200 random .jpg images (or all if there are fewer than 200)
    selected_images = random.sample(images, min(200, len(images)))

    # Copy selected .jpg images to train200
    for file in selected_images:
        shutil.copy(os.path.join(class_path, file), os.path.join(train200_class_dir, file))

print("200 .jpg files per class copied to train200.")
