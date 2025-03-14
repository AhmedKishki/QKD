import os
import csv
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd

def get_model(model_name, pretrained=True):
    """
    Returns a model instance pre-trained on ImageNet.
    """
    model_name = model_name.lower()

    if model_name == 'alexnet':
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.alexnet(weights=weights)
    elif model_name == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif model_name == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
        
    return model
############################################################################################
class NewModel(torch.nn.Module):
    """
    Wraps a pretrained model with a new classification head.

    Args:
        pretrained_model (nn.Module): The original pretrained model.
        num_classes (int): Number of output classes.
    """
    def __init__(self, pretrained_model, num_classes):
        super(NewModel, self).__init__()
        self.model = pretrained_model
        self.fc = torch.nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
############################################################################################
def fine_tune_model(pretrained_model, num_classes, train_loader, device, freeze=True, lr=1e-3, epochs=10):
    """
    Fine-tunes a pretrained model by freezing its layers and training a new classification head.

    Args:
        pretrained_model (nn.Module): The original pretrained model.
        num_classes (int): Number of output classes.
        train_loader (DataLoader): Dataloader for training.
        device (torch.device): Device to train on.
        freeze (bool): If True, freezes the pretrained model's weights. Default is True.
        lr (float): Learning rate for fine-tuning. Default is 1e-3.
        epochs (int): Number of training epochs. Default is 10.

    Returns:
        nn.Module: The fine-tuned model.
    """
    model = NewModel(pretrained_model, num_classes).to(device)

    if freeze:
        for param in model.model.parameters():
            param.requires_grad = False  # Freeze the pretrained model

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model
############################################################################################
def get_data_loaders(train_dir, val_dir, batch_size, num_workers=4):
    """
    Prepares and returns the train and validation DataLoaders.
    
    Optionally, if train_samples_per_class is provided,
    only that many random samples per class will be selected for the training set.
    The validation set will remain the full dataset.
    
    Args:
        train_dir (str): Directory path for training data.
        val_dir (str): Directory path for validation data.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        train_samples_per_class (int, optional): Number of samples to select per class for the training set.
    
    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    # Define transforms for training and validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the full datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Total train samples: {len(train_dataset)}")
    print(f"Total val samples: {len(val_dataset)}")
    print(f"Train steps per epoch: {len(train_loader)}")
    print(f"Validation steps per epoch: {len(val_loader)}")

    return train_loader, val_loader
############################################################################################
def save_results_csv(csv_path,
                          teacher_model,
                          student_model,
                          alpha,
                          temperature,
                          epochs,
                          student_qkd_acc):
    """
    Saves the results to a CSV file. If the file exists, it appends a new row; otherwise, it creates the file.
    """
    headers = ['Teacher', 'Student', 'Alpha', 'Temperature', 'Epochs', 'QKD Accuracy']
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Teacher': teacher_model,
            'Student': student_model,
            'Alpha': alpha,
            'Temperature': f'{temperature:.1f}',
            'Epochs': epochs,
            'QKD Accuracy': student_qkd_acc
        })
    
    print(f"Saved results to {csv_path}")
############################################################################################   
def check_config_csv(csv_path, teacher_model_name, student_model_name, alpha_teacher, alpha_student, temperature, num_epochs):
    """
    Checks if a given experiment result already exists in the CSV file.
    """
    num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring = num_epochs
    
    print(f"\n[INFO] Checking existing experiments for:")
    print(f"       Teacher: {teacher_model_name}, Student: {student_model_name}, Alpha: t:{alpha_teacher:.1f}, s:{alpha_student:.1f}, Temp: {temperature:.1f}")

    if not os.path.exists(csv_path):
        print("[INFO] CSV file does not exist. Proceeding with experiment.")
        return False

    with open(csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row:
                continue  # Skip empty rows
            if (row['Teacher'] == teacher_model_name and
                row['Student'] == student_model_name and
                row['Alpha'] == f't:{alpha_teacher:.1f}, s:{alpha_student:.1f}' and
                row['Temperature'] == f'{temperature:.1f}' and
                row['Epochs'] == f"{num_epochs_selfstudying}-{num_epochs_costudying}-{num_epochs_tutoring}"):
                print("[WARNING] Experiment already exists. Skipping...")
                return True

    print("[INFO] Experiment does not exist. Proceeding with training.")
    return False
############################################################################################