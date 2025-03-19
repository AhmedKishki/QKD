import os
import torch
import csv
import statistics
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from helper import NewModel, fine_tune_model
from helper2 import get_model, get_data_loaders

# Constants
retrials = 5
epochs = 100
cwd = os.getcwd()
models = ['mobilenet_v3_small', 'resnet18']
dataset = 'CIFAR100'
num_classes = 100
train_dir = os.path.join(cwd, "CIFAR100/train")
test_dir = os.path.join(cwd, "CIFAR100/valid")
batch_size = 32
num_workers = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size, num_workers, dataset)

# CSV file setup
csv_filename = "base_model_results.csv"
header = ["Model", "Avg Top-1 Acc", "Std Top-1 Acc", "Avg Top-5 Acc", "Std Top-5 Acc"]

# Write the header only if the file does not exist
file_exists = os.path.exists(csv_filename)

if not file_exists:
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

for model_name in models:
    top1_accuracies = []
    top5_accuracies = []

    for _ in range(retrials):
        pretrained_model = get_model(model_name, pretrained=True)
        model = NewModel(pretrained_model, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = torch.nn.CrossEntropyLoss()

        print(f'\n=============FINE TUNING {model_name}=============\n')

        for epoch in range(epochs):
            total_train_loss = 0
            model.train()

            progress_bar = tqdm(train_loader, desc=f"Training - Epoch {epoch+1}/{epochs}", unit="batch")
            for step, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                if (step + 1) % 100 == 0:
                    progress_bar.set_postfix(loss=loss.item())

            scheduler.step()

        # Final evaluation on test set
        model.eval()
        correct_top1, correct_top5, total = 0, 0, 0
        progress_bar_test = tqdm(test_loader, desc=f"Testing {model_name}", unit="batch")
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(progress_bar_test):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                _, predicted_top1 = torch.max(outputs, 1)
                _, predicted_top5 = torch.topk(outputs, 5, dim=1)
                total += labels.size(0)
                correct_top1 += (predicted_top1 == labels).sum().item()
                correct_top5 += sum([labels[i] in predicted_top5[i] for i in range(labels.size(0))])

        top1_accuracy = 100 * correct_top1 / total
        top5_accuracy = 100 * correct_top5 / total

        print(f"Test Results - {model_name}: Top-1 Accuracy: {top1_accuracy:.2f}%, Top-5 Accuracy: {top5_accuracy:.2f}%")

        top1_accuracies.append(top1_accuracy)
        top5_accuracies.append(top5_accuracy)

    # Compute mean and standard deviation
    avg_top1 = statistics.mean(top1_accuracies)
    std_top1 = statistics.stdev(top1_accuracies)
    avg_top5 = statistics.mean(top5_accuracies)
    std_top5 = statistics.stdev(top5_accuracies)

    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, avg_top1, std_top1, avg_top5, std_top5])

print(f"Test results appended to {csv_filename}")
