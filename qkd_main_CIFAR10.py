import os
import csv
import copy
import torch

from helper2 import (
    quantization_knowledge_distillation,
    evaluate_model_quantized,
    save_results_csv_qkd,
    get_model,
    get_data_loaders,
    check_if_experiment_exists
)

from helper import NewModel, fine_tune_model

cwd = os.getcwd()

def train_quantized_student_with_teacher(
    csv_file_name,
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
    student_lr=1e-2,
    teacher_lr=1e-6,
    retrials=1,
    dataset='50'
):
    """
    Trains a student model using quantized knowledge distillation from a teacher model.
    """
    # Check if experiment already exists
    if check_if_experiment_exists(csv_file_name, teacher_model_name, student_model_name, alpha_teacher, alpha_student, temperature, num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring, retrials):
        print(f"[SKIP] Experiment {teacher_model_name} -> {student_model_name} (alpha={alpha_teacher:.1f}, temp={temperature:.1f}) already completed.")
        return
    
    print("\n[START] Quantization Knowledge Distillation (QKD) Training")
    print(f"       Teacher: {teacher_model_name}, Student: {student_model_name}")
    print(f"       Alpha Student: {alpha_student:.1f}, Alpha Teacher: {alpha_teacher:.1f}, Temperature: {temperature:.1f}")
    print(f"       Training Details: Self-Studying: {num_epochs_selfstudying} epochs, Co-Studying: {num_epochs_costudying} epochs, Tutoring: {num_epochs_tutoring} epochs")
    
    # Move models to device
    teacher.to(device)
    student_qkd = student.to(device)

    # Fine tuning
    teacher = fine_tune_model(teacher, 10, train_loader, val_loader, device)
    student_qkd = fine_tune_model(student_qkd, 10, train_loader, val_loader, device)

    # Train with quantization knowledge distillation
    student_qkd = quantization_knowledge_distillation(
        student=student_qkd,
        teacher=teacher,
        train_loader=train_loader,
        kd_loss=kd_loss,
        num_epochs_selfstudying=num_epochs_selfstudying,
        num_epochs_costudying=num_epochs_costudying,
        num_epochs_tutoring=num_epochs_tutoring,
        device=device,
        student_lr=student_lr,
        teacher_lr=teacher_lr,
        alpha_s=alpha_student,
        alpha_t=alpha_teacher,
        temperature=temperature,
        log_interval=100,
        dataset=dataset
    )

    print(f"\n[EVALUATION] Evaluating Quantized Student Model ({student_model_name})...")
    student_qkd_acc = evaluate_model_quantized(student_qkd.cpu(), val_loader)
    print(f"[RESULT] QKD Accuracy: {student_qkd_acc}")

    # Save results to CSV
    print("[INFO] Saving results to CSV...")
    save_results_csv_qkd(
        csv_file_name,
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
    name = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retrials = 1
    
    # ------------------------------
    # Define Teacher and Student Models
    # ------------------------------
    teacher_student_pairs = [
        ('mobilenet_v3_small', 'mobilenet_v3_small')
    ]
    
    # ------------------------------
    # Data Loaders
    # ------------------------------
    dataset = 'CIFAR10'
    train_dir = os.path.join(cwd, "CIFAR10/train")
    val_dir = os.path.join(cwd, "CIFAR10/valid")
    batch_size = 32
    num_workers = 16
    print("[INFO] Loading data...")
    train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size, num_workers, dataset)
    print("[SUCCESS] Data loaded successfully.")
    
    # # ------------------------------
    # # Experiment Hyperparameters
    # # ------------------------------
    retrials = 3
    kd_loss_labels = ['KL', 'CS']
    alpha_st_pairs = [(0.7,0.3)]
    temperatures = [6.0]
    student_lr = 1e-4
    teacher_lr = 1e-6
    num_epochs = [(30,20,50)]
    names = ["_".join(f"{x:02d}" for x in t) for t in num_epochs]
    
    # # ------------------------------
    # # Running Experiment
    # # ------------------------------
    for teacher_model_name, student_model_name in teacher_student_pairs:
        for kd_loss in kd_loss_labels:
            for i, (num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring) in enumerate(num_epochs):
                for alpha_s, alpha_t in alpha_st_pairs:
                    for temp in temperatures:
                        for _ in range(retrials):
                            csv_filename = os.path.join(cwd, f"results_qkd_{kd_loss}_{names[i]}_{dataset}.csv")
                            print(f"\n[MODEL SETUP] Teacher: {teacher_model_name}, Student: {student_model_name}")
                            teacher = get_model(teacher_model_name, pretrained=True)
                            student = get_model(student_model_name, pretrained=True)
                            print(f"\n[TRAINING SETUP] Alpha Teacher: {alpha_t:.1f}, Alpha Student: {alpha_s:.1f}, Temperature: {temp:.1f}")
                            train_quantized_student_with_teacher(
                                csv_filename,
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
                                student_lr,
                                teacher_lr,
                                retrials,
                                dataset
                            )

if __name__ == "__main__":
    main()
