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

cwd = os.getcwd()

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
    kd_loss,
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
    csv_filename = os.path.join(cwd, f"results_qkd_{kd_loss}_small.csv")

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
        kd_loss=kd_loss,
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
    train_dir = os.path.join(cwd, "ImageNet/train_small")
    val_dir = os.path.join(cwd, "ImageNet/valid_small")
    batch_size = 32
    num_workers = 16
    
    # ------------------------------
    # Experiment 1 Hyperparameters
    # ------------------------------
    kd_loss_labels = ['CS', 'KL', 'JS', 'TV']
    alpha_st_pairs = [(0.5,0.5),(0.7,0.3),(1.0,0.5)]
    temperatures = [6.0]
    num_epochs = [(4,4,4),(6,3,3),(3,6,3),(3,3,6),(6,6,0),(0,6,6),(0,12,0)]
    max_lr = 1e-3
    min_lr = 1e-6
    teacher_lr = 1e-6
    
    # ------------------------------
    # Experiment 2 Hyperparameters
    # ------------------------------
    # kd_loss_labels = ['CS', 'KL', 'JS', 'TV']
    # alpha_st_pairs = [(0.7,0.3)]
    # temperatures = [6.0]
    # num_epochs = [(20,0,0),(0,20,0),(0,0,20),(0,15,5),(5,0,15),(15,0,5),(0,5,15)]
    # max_lr = 1e-3
    # min_lr = 1e-6
    # teacher_lr = 1e-6
    
    # ------------------------------
    # Experiment 3 Hyperparameters
    # ------------------------------
    
    # alpha_st_pairs = []
    # temperatures = [6.0]
    # num_epochs = []
    # max_lr = 1e-3
    # min_lr = 1e-6
    # teacher_lr = 1e-6

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
        ('mobilenet_v3_small', 'mobilenet_v3_small')
    ]
    
    for teacher_model_name, student_model_name in teacher_student_pairs:
        print(f"\n[MODEL SETUP] Teacher: {teacher_model_name}, Student: {student_model_name}")
        teacher = get_model(teacher_model_name, pretrained=True)
        student = get_model(student_model_name, pretrained=True)
        for kd_loss in kd_loss_labels:
            for num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring in num_epochs:
                for alpha_s, alpha_t in alpha_st_pairs:
                    for temp in temperatures:
                        print(f"\n[TRAINING SETUP] Alpha Teacher: {alpha_t:.1f}, Alpha Student: {alpha_s:.1f}, Temperature: {temp:.1f}")
                        train_quantized_student_with_teacher(
                            kd_loss,
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
