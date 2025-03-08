import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qat_qconfig_mapping
from tqdm import tqdm

from loss_functions import kl_loss, cs_loss, ms_loss


def phase_training(phase: str,
                   num_epochs: int,
                   student: torch.nn.Module,
                   teacher: torch.nn.Module,
                   train_loader,
                   device,
                   optimizer_student: optim.Optimizer,
                   optimizer_teacher: optim.Optimizer,
                   scheduler_student: torch.optim.lr_scheduler.LambdaLR,
                   kd_loss_fn,
                   alpha_s: float,
                   alpha_t: float,
                   temperature: float,
                   log_interval: int):
    """
    Trains for a given number of epochs using a mode determined by `phase`:
      - 'selfstudying': Only the student is updated using cross-entropy loss.
      - 'costudying': Both student and teacher are updated with cross-entropy and KD losses.
      - 'tutoring': Only the student is updated (teacher in eval mode) using KD loss.
    """
    if phase == 'selfstudying':
        student.train()
        teacher.eval()  # Teacher is not used in self-studying
    elif phase == 'costudying':
        student.train()
        teacher.train()
    elif phase == 'tutoring':
        student.train()
        teacher.eval()
    else:
        raise ValueError("Invalid phase. Choose 'selfstudying', 'costudying', or 'tutoring'.")

    for epoch in range(num_epochs):
        running_s_loss = 0.0
        running_t_loss = 0.0  # Only used during co-studying

        progress_bar = tqdm(train_loader, desc=f"{phase.capitalize()} Phase - Epoch {epoch+1}/{num_epochs}", unit="batch")
        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if phase == 'costudying':
                # Zero gradients for both student and teacher
                optimizer_student.zero_grad()
                optimizer_teacher.zero_grad()
                
                teacher_outputs = teacher(inputs)
                student_outputs = student(inputs)

                # Student losses
                s_ce_loss = F.cross_entropy(student_outputs, labels)
                s_kd_loss = kd_loss_fn(student_outputs, teacher_outputs.detach(), temperature)
                student_loss = alpha_s * s_kd_loss + (1 - alpha_s) * s_ce_loss

                # Teacher losses
                t_ce_loss = F.cross_entropy(teacher_outputs, labels)
                t_kd_loss = kd_loss_fn(teacher_outputs, student_outputs.detach(), temperature)
                teacher_loss = alpha_t * t_kd_loss + (1 - alpha_t) * t_ce_loss

                student_loss.backward()
                optimizer_student.step()
                
                teacher_loss.backward()
                optimizer_teacher.step()

                running_s_loss += student_loss.item()
                running_t_loss += teacher_loss.item()

                if (step + 1) % log_interval == 0:
                    progress_bar.set_postfix({
                        "S_loss": f"{running_s_loss / (step + 1):.4f}",
                        "T_loss": f"{running_t_loss / (step + 1):.4f}"
                    })
            elif phase == 'tutoring':
                optimizer_student.zero_grad()

                # Teacher in evaluation mode (no gradients)
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                student_outputs = student(inputs)

                s_ce_loss = F.cross_entropy(student_outputs, labels)
                s_kd_loss = kd_loss_fn(student_outputs, teacher_outputs.detach(), temperature)
                student_loss = alpha_s * s_kd_loss + (1 - alpha_s) * s_ce_loss

                student_loss.backward()
                optimizer_student.step()

                running_s_loss += student_loss.item()

                if (step + 1) % log_interval == 0:
                    progress_bar.set_postfix({
                        "S_loss": f"{running_s_loss / (step + 1):.4f}"
                    })
            elif phase == 'selfstudying':
                optimizer_student.zero_grad()
                student_outputs = student(inputs)
                s_ce_loss = F.cross_entropy(student_outputs, labels)
                s_ce_loss.backward()
                optimizer_student.step()

                running_s_loss += s_ce_loss.item()

                if (step + 1) % log_interval == 0:
                    progress_bar.set_postfix({
                        "S_CE_loss": f"{running_s_loss / (step + 1):.4f}"
                    })

        scheduler_student.step()

        if phase == 'costudying':
            print(f"{phase.capitalize()} - Epoch [{epoch+1}/{num_epochs}] - S_loss: {running_s_loss / len(train_loader):.4f} | T_loss: {running_t_loss / len(train_loader):.4f}")
        elif phase in ['tutoring', 'selfstudying']:
            print(f"{phase.capitalize()} - Epoch [{epoch+1}/{num_epochs}] - S_loss: {running_s_loss / len(train_loader):.4f}")


def quantization_knowledge_distillation(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    train_loader,
    device,
    kd_loss='KL',
    num_epochs_selfstudying=5,
    num_epochs_costudying=5,
    num_epochs_tutoring=5,
    max_lr=1e-3,
    min_lr=1e-6,
    teacher_lr=1e-6,
    alpha_s=0.5,
    alpha_t=0.5,
    temperature=6.0,
    log_interval=100
):
    """
    Implements Quantization-Aware Knowledge Distillation (QKD) by sequentially running
    self-studying, co-studying, and tutoring phases using a common training function.
    """
    # Select KD loss function
    if kd_loss == 'KL':
        kd_loss_fn = kl_loss
        kd_loss_label = "Kullback-Leibler (KL) Divergence"
    elif kd_loss == 'CS':
        kd_loss_fn = cs_loss
        kd_loss_label = "Cosine Similarity (CS)"
    elif kd_loss == 'MS':
        kd_loss_fn = ms_loss
        kd_loss_label = "Mean Squared Error (MS)"
    else:
        raise ValueError(f"Invalid KD loss function: {kd_loss}")

    print("\nStarting Quantization-Aware Knowledge Distillation (QKD) training.")
    print(f"\nKD Loss: {kd_loss_label}")

    # Set up QAT configuration using FX Graph Mode
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    example_inputs = next(iter(train_loader))[0]  # Sample input for tracing
    student = prepare_qat_fx(student, qconfig_mapping, example_inputs)

    student.to(device)
    teacher.to(device)

    optimizer_student = optim.AdamW(student.parameters(), lr=max_lr)
    optimizer_teacher = optim.AdamW(teacher.parameters(), lr=teacher_lr)

    total_epochs = num_epochs_selfstudying + num_epochs_costudying + num_epochs_tutoring
    scheduler_student = torch.optim.lr_scheduler.LambdaLR(
        optimizer_student,
        lr_lambda=lambda epoch: (min_lr / max_lr) ** (epoch / (total_epochs - 1))
    )

    # Execute all phases sequentially using the common phase_training function
    phase_training('selfstudying', num_epochs_selfstudying, student, teacher, train_loader, device,
                   optimizer_student, optimizer_teacher, scheduler_student, kd_loss_fn,
                   alpha_s, alpha_t, temperature, log_interval)
    
    phase_training('costudying', num_epochs_costudying, student, teacher, train_loader, device,
                   optimizer_student, optimizer_teacher, scheduler_student, kd_loss_fn,
                   alpha_s, alpha_t, temperature, log_interval)
    
    phase_training('tutoring', num_epochs_tutoring, student, teacher, train_loader, device,
                   optimizer_student, optimizer_teacher, scheduler_student, kd_loss_fn,
                   alpha_s, alpha_t, temperature, log_interval)
        
    # Convert to quantized inference format
    student.to('cpu')
    student.eval()
    student = convert_fx(student)

    return student
