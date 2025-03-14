import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qat_qconfig_mapping
from tqdm import tqdm

from loss_functions import Loss, kl_loss, cs_loss, ms_loss, js_loss, tv_loss


def selfstudying_phase(num_epochs, student, train_loader, device,
                       optimizer_student, scheduler_student, log_interval, loss_handle):
    """
    Selfstudying phase:
    - Only the student is updated using cross-entropy loss.
    """
    student.train()
    for epoch in range(num_epochs):
        running_s_ce_loss = 0.0  # Initialize correctly

        progress_bar = tqdm(train_loader,
                            desc=f"Selfstudying Phase - Epoch {epoch+1}/{num_epochs}",
                            unit="batch")
        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_student.zero_grad()

            outputs = student(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer_student.step()

            running_s_ce_loss += loss.item()

            if (step + 1) % log_interval == 0:
                progress_bar.set_postfix({
                    "S_CE_loss": f"{running_s_ce_loss / (step + 1):.4f}"
                })

        loss_handle.student_loss.append(running_s_ce_loss / len(train_loader))
        loss_handle.student_ce_loss.append(running_s_ce_loss / len(train_loader))
        loss_handle.student_kd_loss.append(None)
        loss_handle.teacher_loss.append(None)
        loss_handle.teacher_ce_loss.append(None)
        loss_handle.teacher_kd_loss.append(None)

        loss_handle.print()
        loss_handle.save()

        scheduler_student.step()


def costudying_phase(num_epochs, student, teacher, train_loader, device,
                     optimizer_student, optimizer_teacher, scheduler_student,
                     kd_loss_fn, alpha_s, alpha_t, temperature, log_interval, loss_handle):
    """
    Co-studying phase:
    - Both student and teacher are updated.
    - Student and teacher losses are a combination of cross-entropy and KD losses.
    """
    student.train()
    teacher.train()
    for epoch in range(num_epochs):
        running_s_loss = 0.0
        running_t_loss = 0.0
        running_s_ce_loss = 0.0
        running_t_ce_loss = 0.0
        running_s_kd_loss = 0.0
        running_t_kd_loss = 0.0

        progress_bar = tqdm(train_loader,
                            desc=f"Costudying Phase - Epoch {epoch+1}/{num_epochs}",
                            unit="batch")
        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

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
            running_s_ce_loss += s_ce_loss.item()
            running_t_ce_loss += t_ce_loss.item()
            running_s_kd_loss += s_kd_loss.item()
            running_t_kd_loss += t_kd_loss.item()

            if (step + 1) % log_interval == 0:
                progress_bar.set_postfix({
                    "S_loss": f"{running_s_loss / (step + 1):.4f}",
                    "T_loss": f"{running_t_loss / (step + 1):.4f}"
                })

        loss_handle.student_loss.append(running_s_loss / len(train_loader))
        loss_handle.student_ce_loss.append(running_s_ce_loss / len(train_loader))
        loss_handle.student_kd_loss.append(running_s_kd_loss / len(train_loader))
        loss_handle.teacher_loss.append(running_t_loss / len(train_loader))
        loss_handle.teacher_ce_loss.append(running_t_ce_loss / len(train_loader))
        loss_handle.teacher_kd_loss.append(running_t_kd_loss / len(train_loader))

        loss_handle.print()
        loss_handle.save()
        
        scheduler_student.step()


def tutoring_phase(num_epochs, student, teacher, train_loader, device,
                   optimizer_student, scheduler_student, kd_loss_fn, alpha_s,
                   temperature, log_interval, loss_handle):
    """
    Tutoring phase:
    - Only the student is updated.
    - Teacher is in evaluation mode (no gradients).
    - Student is trained with KD loss and cross-entropy loss.
    """
    student.train()
    teacher.eval()
    for epoch in range(num_epochs):
        running_loss = 0.0

        progress_bar = tqdm(train_loader,
                            desc=f"Tutoring Phase - Epoch {epoch+1}/{num_epochs}",
                            unit="batch")
        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_student.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)

            s_ce_loss = F.cross_entropy(student_outputs, labels)
            s_kd_loss = kd_loss_fn(student_outputs, teacher_outputs.detach(), temperature)
            student_loss = alpha_s * s_kd_loss + (1 - alpha_s) * s_ce_loss

            student_loss.backward()
            optimizer_student.step()

            running_loss += student_loss.item()

            if (step + 1) % log_interval == 0:
                progress_bar.set_postfix({
                    "S_loss": f"{running_loss / (step + 1):.4f}"
                })

        loss_handle.student_loss.append(running_loss / len(train_loader))
        loss_handle.print()
        loss_handle.save()

        scheduler_student.step()

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
    Implements Quantization-Aware Knowledge Distillation (QKD) by running
    selfstudying, costudying, and tutoring phases sequentially.
    """
    # Select KD loss function
    if kd_loss == 'KL':
        kd_loss_fn = kl_loss
        kd_loss_label = "Kullback-Leibler Divergence (KL)"
    elif kd_loss == 'CS':
        kd_loss_fn = cs_loss
        kd_loss_label = "Cosine Similarity (CS)"
    elif kd_loss == 'MS':
        kd_loss_fn = ms_loss
        kd_loss_label = "Mean Squared Error (MS)"
    elif kd_loss == 'JS':
        kd_loss_fn = js_loss
        kd_loss_label = "Jensen-Shannon Divergence (JS)"
    elif kd_loss == 'TV':
        kd_loss_fn = tv_loss
        kd_loss_label = "Total Variation (TV)"
    else:
        raise ValueError(f"Invalid KD loss function: {kd_loss}")

    print("\nStarting Quantization-Aware Knowledge Distillation (QKD) training.")
    print(f"\nKD Loss: {kd_loss_label}")

    # Set up QAT configuration using FX Graph Mode
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    example_inputs = next(iter(train_loader))[0]
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
    
    loss_handle = Loss(dataset,kd_loss, num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring)
    
    ############# Pre-Evaluation #############
    student.eval()
    teacher.eval()
    with torch.no_grad():
        running_s_ce_loss = 0.0
        running_t_ce_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            student_outputs = student(inputs)
            teacher_outputs = teacher(inputs)
            s_ce_loss = F.cross_entropy(student_outputs, labels)
            t_ce_loss = F.cross_entropy(teacher_outputs, labels)
            running_s_ce_loss += s_ce_loss.item()
            running_t_ce_loss += t_ce_loss.item()
            
    loss_handle.student_loss.append(running_s_ce_loss / len(train_loader))
    loss_handle.student_ce_loss.append(running_s_ce_loss / len(train_loader))
    loss_handle.student_kd_loss.append(None)
    loss_handle.teacher_loss.append(running_t_ce_loss / len(train_loader))
    loss_handle.teacher_ce_loss.append(running_t_ce_loss / len(train_loader))
    loss_handle.teacher_kd_loss.append(None)
    loss_handle.print()
    loss_handle.save()
            
    ############# Self Studying #############
    selfstudying_phase(num_epochs_selfstudying, student, train_loader, device,
                       optimizer_student, scheduler_student, log_interval, loss_handle)

    ############# Co Studying #############
    costudying_phase(num_epochs_costudying, student, teacher, train_loader, device,
                     optimizer_student, optimizer_teacher, scheduler_student,
                     kd_loss_fn, alpha_s, alpha_t, temperature, log_interval, loss_handle)

    ############# Tutoring #############
    tutoring_phase(num_epochs_tutoring, student, teacher, train_loader, device,
                   optimizer_student, scheduler_student, kd_loss_fn, alpha_s, temperature, log_interval, loss_handle)

    student.to('cpu')
    student.eval()
    student = convert_fx(student)

    return student