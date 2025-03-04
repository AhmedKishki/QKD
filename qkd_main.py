import os
import csv
import torch

from helper2 import (
    quantization_knowledge_distillation,
    evaluate_model_quantized,
    save_results_csv_qkd,
    get_model,
    get_data_loaders
)

def check_if_experiment_exists(csv_filename, teacher_model_name, student_model_name, alpha_teacher, alpha_student, temperature, num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring):
    """
    Checks if a given experiment result already exists in the CSV file.
    """
    print(f"\n[INFO] Checking existing experiments for:")
    print(f"       Teacher: {teacher_model_name}, Student: {student_model_name}, Alpha: t:{alpha_teacher:.1f}, s:{alpha_student:.1f}, Temp: {temperature:.1f}")

    if not os.path.exists(csv_filename):
        print("[INFO] CSV file does not exist. Proceeding with experiment.")
        return False

    with open(csv_filename, mode='r', newline='') as file:
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

def train_quantized_student_with_teacher(
    teacher, 
    student,
    teacher_model_name, 
    student_model_name, 
    train_loader, 
    val_loader,
    alpha_student,
    alpha_teacher,
    temperature,
    num_epochs_selfstudying=5,
    num_epochs_costudying=5,
    num_epochs_tutoring=5,
    device='cuda',
    max_lr=1e-3,
    min_lr=1e-6,
    teacher_lr=1e-6
):
    """
    Trains a student model using quantized knowledge distillation from a teacher model.
    """
    csv_filename = "/home/ida01/ew2218/QKD/qkd_results_200.csv"

    # Check if experiment already exists
    if check_if_experiment_exists(csv_filename, teacher_model_name, student_model_name, alpha_teacher, alpha_student, temperature, num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring):
        print(f"[SKIP] Experiment {teacher_model_name} -> {student_model_name} (alpha={alpha_teacher:.1f}, temp={temperature:.1f}) already completed.")
        return
    
    print("\n[START] Quantization Knowledge Distillation (QKD) Training")
    print(f"       Teacher: {teacher_model_name}, Student: {student_model_name}")
    print(f"       Alpha Student: {alpha_student:.1f}, Alpha Teacher: {alpha_teacher:.1f}, Temperature: {temperature:.1f}")
    print(f"       Training Details: Self-Studying: {num_epochs_selfstudying} epochs, Co-Studying: {num_epochs_costudying} epochs, Tutoring: {num_epochs_tutoring} epochs")
    print(f"       Device: {device}, Max LR: {max_lr:.2e}, Min LR: {min_lr:.2e}, Teacher LR: {teacher_lr:.2e}")

    # Move models to device
    teacher.to(device)
    student.to(device)

    # Train with quantization knowledge distillation
    student = quantization_knowledge_distillation(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        num_epochs_selfstudying=num_epochs_selfstudying,
        num_epochs_costudying=num_epochs_costudying,
        num_epochs_tutoring=num_epochs_tutoring,
        device=device,
        max_lr=max_lr,
        min_lr=min_lr,
        teacher_lr=teacher_lr,
        alpha_s=alpha_student,
        alpha_t=alpha_teacher,
        temperature=temperature,
        log_interval=100
    )

    print(f"\n[EVALUATION] Evaluating Quantized Student Model ({student_model_name})...")
    student_qkd_acc = evaluate_model_quantized(student, val_loader)
    print(f"[RESULT] QKD Accuracy: {student_qkd_acc}")

    # Save results to CSV
    print("[INFO] Saving results to CSV...")
    save_results_csv_qkd(
        csv_filename,
        teacher_model_name,
        student_model_name,
        f't:{alpha_teacher:.1f}, s:{alpha_student:.1f}',
        temperature,
        f"{num_epochs_selfstudying}-{num_epochs_costudying}-{num_epochs_tutoring}",
        student_qkd_acc
    )
    print("[SUCCESS] Experiment results saved.")

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
    # KD Hyperparameters
    # ------------------------------
    alpha_st_pairs = [(0.5,0.5), (1.0,1.0), (1.0,0.5)]
    temperatures = [6.0]
    num_epochs = [(5,5,5), (20,10,20), (30,0,20)]
    max_lr = 1e-3
    min_lr = 1e-6
    teacher_lr = 1e-6

    # ------------------------------
    # Data Loaders
    # ------------------------------
    print("[INFO] Loading data...")
    train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size, num_workers)
    print("[SUCCESS] Data loaded successfully.")

    # ------------------------------
    # Define Teacher and Student Models
    # ------------------------------
    teacher_student_pairs = [
        ('mobilenet_v3_small', 'mobilenet_v3_small'),
        ('resnet18', 'resnet18'),
        ('mobilenet_v3_large', 'mobilenet_v3_small')
    ]
    
    for teacher_model_name, student_model_name in teacher_student_pairs:
        print(f"\n[MODEL SETUP] Teacher: {teacher_model_name}, Student: {student_model_name}")
        teacher = get_model(teacher_model_name, pretrained=True)
        student = get_model(student_model_name, pretrained=True)
        
        for num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring in num_epochs:
            for alpha_s, alpha_t in alpha_st_pairs:
                for temp in temperatures:
                    print(f"\n[TRAINING SETUP] Alpha Teacher: {alpha_t:.1f}, Alpha Student: {alpha_s:.1f}, Temperature: {temp:.1f}")
                    train_quantized_student_with_teacher(
                        teacher,
                        student,
                        teacher_model_name,
                        student_model_name,
                        train_loader,
                        val_loader,
                        alpha_s,
                        alpha_t,
                        temp,
                        num_epochs_selfstudying,
                        num_epochs_costudying,
                        num_epochs_tutoring,
                        device,
                        max_lr,
                        min_lr,
                        teacher_lr
                    )

if __name__ == "__main__":
    main()
