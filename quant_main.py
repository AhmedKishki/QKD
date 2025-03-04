import os
import csv
import copy
import torch
import torch.optim as optim

from helper2 import (
    post_training_quantization,
    quantization_aware_training,
    evaluate_model_quantized,
    get_model,
    get_data_loaders,
    save_results_csv_quant
)

def check_if_experiment_exists(csv_filename, model_name):
    """
    Checks if a given model result already exists in the CSV file.
    """
    print(f"\n[INFO] Checking existing experiments for Model: {model_name}")
    
    if not os.path.exists(csv_filename):
        print("[INFO] CSV file does not exist. Proceeding with experiment.")
        return False
    
    with open(csv_filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get('Model', '').strip() == model_name:
                print(f"[WARNING] Experiment already exists for {model_name}. Skipping...")
                return True  # Experiment already recorded

    print("[INFO] No previous experiment found. Proceeding with training.")
    return False

def train_quantization(
    model_name, 
    train_loader, 
    val_loader,
    num_epochs_qat=10,
    device='cuda',
    max_lr=1e-3,
    min_lr=1e-6,
    csv_filename="/home/ida01/ew2218/QKD/quant_results.csv"
): 
    """
    Trains a model using quantization techniques (QAT and PTQ) only if results do not already exist.
    """
    # Check if experiment should run
    if check_if_experiment_exists(csv_filename, model_name):
        return

    print(f"\n[START] Running Quantization Benchmark for Model: {model_name}")

    # Load and prepare the model
    model = get_model(model_name, pretrained=True)
    model_ptq = copy.deepcopy(model)
    model_qat = copy.deepcopy(model)

    # -----------------------------------
    # Base Model Evaluation
    # -----------------------------------
    print("\n[EVALUATION] Evaluating Base Model before Quantization...")
    model_init_acc = evaluate_model_quantized(model, val_loader)
    print(f"[RESULT] Base Model Accuracy: {model_init_acc:.2f}%")

    # -----------------------------------
    # Quantization-Aware Training (QAT)
    # -----------------------------------
    print("\n[TRAINING] Starting Quantization-Aware Training (QAT)...")
    print(f"       Model: {model_name}, Epochs: {num_epochs_qat}, Device: {device}")
    print(f"       Learning Rate: Max={max_lr:.2e}, Min={min_lr:.2e}")

    model_qat = quantization_aware_training(model_qat, train_loader, device, num_epochs_qat, max_lr, min_lr)

    print("\n[EVALUATION] Evaluating QAT Model...")
    model_qat_acc = evaluate_model_quantized(model_qat, val_loader)
    print(f"[RESULT] QAT Model Accuracy: {model_qat_acc:.2f}%")

    # -----------------------------------
    # Post-Training Quantization (PTQ)
    # -----------------------------------
    print("\n[QUANTIZATION] Applying Post-Training Quantization (PTQ)...")
    model_ptq = post_training_quantization(model_ptq, train_loader, device)

    print("\n[EVALUATION] Evaluating PTQ Model...")
    model_ptq_acc = evaluate_model_quantized(model_ptq, val_loader)
    print(f"[RESULT] PTQ Model Accuracy: {model_ptq_acc:.2f}%")

    # -----------------------------------
    # Save results to CSV
    # -----------------------------------
    print("[INFO] Saving experiment results to CSV...")
    save_results_csv_quant(
        csv_filename,
        model_name,
        model_init_acc,
        model_ptq_acc,
        model_qat_acc
    )
    print("[SUCCESS] Results saved successfully.")

def main():
    # ------------------------------
    # Global Configuration
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = "/home/ida01/ew2218/QKD/ImageNet/train200"
    val_dir = "/home/ida01/ew2218/QKD/ImageNet/valid"
    batch_size = 64
    num_workers = 16
    
    # ------------------------------
    # Hyperparameters
    # ------------------------------
    max_lr = 1e-3
    min_lr = 1e-6
    num_epochs_qat = 50
    
    # ------------------------------
    # Data Loaders
    # ------------------------------
    print("\n[INFO] Loading data loaders...")
    train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size, num_workers)
    print("[SUCCESS] Data loaded successfully.")

    # ------------------------------
    # Define Models for Quantization
    # ------------------------------
    models = [
        'mobilenet_v3_small', 'efficientnet_v2_s', 
        'resnet18', 'alexnet', 'resnet50', 
        'mobilenet_v3_large', 'efficientnet_v2_l'
    ]

    for model_name in models:
        print(f"\n[INFO] Benchmarking Quantization Performance for {model_name}...")
        train_quantization(
            model_name,
            train_loader,
            val_loader,
            num_epochs_qat=num_epochs_qat,
            device=device,
            max_lr=max_lr,
            min_lr=min_lr
        )

if __name__ == "__main__":
    main()
