import os
import csv
import math
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping, QConfigMapping
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

############################################################################################
def get_model(model_name, pretrained=True):
    """
    Returns a model instance pre-trained on ImageNet.
    """
    model_name = model_name.lower()
    
    if model_name == 'vgg16':
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
    elif model_name == 'alexnet':
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
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    elif model_name == 'resnet152':
        weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet152(weights=weights)
    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
    elif model_name == 'efficientnet_v2_s':
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
    elif model_name == 'efficientnet_v2_l':
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_l(weights=weights)
    elif model_name == 'efficientnet_b7':
        weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b7(weights=weights)
    elif model_name == 'shufflenet_v2':
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
    elif model_name == 'squeezenet':
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.squeezenet1_1(weights=weights)
    elif model_name == 'swin_v2':
        weights = models.Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_v2_b(weights=weights)
    elif model_name == 'maxvit_t':
        weights = models.MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.maxvit_t(weights=weights)
    elif model_name == 'convnext_large':
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_large(weights=weights)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
        
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
def distillation_loss(student_logits, teacher_logits, temperature):
    """Computes the KL divergence loss between the softened student and teacher logits."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss
############################################################################################
def post_training_quantization(model, data_loader, device, num_calibration_samples=100):
    """
    Post-training quantization using FX Graph Mode with limited calibration samples.

    Args:
        model (torch.nn.Module): The PyTorch model to quantize.
        data_loader (DataLoader): DataLoader for the calibration dataset.
        device (torch.device): Device on which the quantization happens (CPU or GPU).
        num_calibration_samples (int): Number of samples to use for calibration.

    Returns:
        torch.nn.Module: The FX quantized model.
    """
    model.to(device)
    model.eval()

    qconfig_mapping = get_default_qconfig_mapping("x86")

    # Prepare the model using FX Graph Mode
    example_inputs = next(iter(data_loader))[0].to(device)  # Get a sample input for tracing
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)

    # Calibration: run a forward pass over a subset of the calibration data
    sample_count = 0
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="Calibrating model"):
            if sample_count >= num_calibration_samples:
                break
            inputs = inputs.to(device)
            prepared_model(inputs)
            sample_count += inputs.size(0)  # Assuming inputs is a batch

    # Convert the calibrated model to a quantized version
    quantized_model = convert_fx(prepared_model.to('cpu'))

    print("Post-training quantization using FX Graph Mode completed.")
    return quantized_model
############################################################################################
def quantization_aware_training(model, train_loader, device, num_epochs, max_lr=1e-3, min_lr=1e-6):
    """
    Applies Quantization-Aware Training (QAT) using FX Graph Mode.

    Args:
        model (torch.nn.Module): The model to be trained with QAT.
        train_loader (DataLoader): DataLoader providing the training data.
        device (torch.device): Device for training (CPU or GPU).
        num_epochs (int): Number of epochs for QAT.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate after decay.
    
    Returns:
        torch.nn.Module: The fully quantized model after QAT.
    """
    model.to(device)
    model.eval()

    # Set up QAT configuration using FX Graph Mode
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    example_inputs = next(iter(train_loader))[0]  # Get a sample input for tracing
    model = prepare_qat_fx(model, qconfig_mapping, example_inputs)

    # Optimizer & Scheduler (same as QKD)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: (min_lr / max_lr) ** (epoch / (num_epochs - 1))
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop.
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"QAT - Epoch {epoch+1}/{num_epochs}", unit="batch")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{running_loss / (len(progress_bar) + 1):.4f}"})

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"QAT - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Convert the QAT model to a fully quantized model.
    model.eval()
    model_fx = convert_fx(model.to('cpu'))
    print("Quantization-Aware Training completed and model converted using FX Graph Mode.")

    return model_fx
############################################################################################
def alpha_scheduler(epoch, total_epochs, max_alpha=1.0, min_alpha=0.2):
    """Smooth transition from heavy KD reliance to balanced training."""
    return min_alpha + (max_alpha - min_alpha) * (0.5 * (1 + math.cos(math.pi * epoch / total_epochs)))
############################################################################################
def temperature_scheduler(epoch, total_epochs, max_T=10.0, min_T=2.0):
    """Soft teacher guidance early, sharper later."""
    decay_factor = (min_T / max_T) ** (epoch / total_epochs)
    return max_T * decay_factor
############################################################################################
def quantization_knowledge_distillation(
    student,
    teacher,
    train_loader,
    device,
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
    Implements Self-Studying, Co-Studying, and Tutoring for quantization-aware knowledge distillation.
    QKD: Quantization-aware Knowledge Distillation
    """
    num_epochs = num_epochs_selfstudying + num_epochs_costudying + num_epochs_tutoring
 
    # Set up QAT configuration using FX Graph Mode
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    example_inputs = next(iter(train_loader))[0]  # Get a sample input for tracing
    student = prepare_qat_fx(student, qconfig_mapping, example_inputs)

    student.to(device)
    teacher.to(device)

    optimizer_student = optim.AdamW(student.parameters(), lr=max_lr)
    optimizer_teacher = optim.AdamW(teacher.parameters(), lr=teacher_lr)

    scheduler_student = torch.optim.lr_scheduler.LambdaLR(
        optimizer_student, 
        lr_lambda=lambda epoch: (min_lr / max_lr) ** (epoch / (num_epochs - 1))
    )

    # === Self-Studying ===
    print("\n=== Self-Studying ===\n")
    student.train()
    for epoch in range(num_epochs_selfstudying):
        running_s_ce_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Self-Studying - Epoch {epoch+1}/{num_epochs_selfstudying}", unit="batch")
        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_student.zero_grad()
            student_outputs = student(inputs)

            s_ce_loss = F.cross_entropy(student_outputs, labels)

            s_ce_loss.backward()
            optimizer_student.step()

            running_s_ce_loss += s_ce_loss.item()

            if (step + 1) % log_interval == 0:
                progress_bar.set_postfix({
                    "S_CE_loss": f"{running_s_ce_loss / (step + 1):.4f}"
                })

        scheduler_student.step()
        print(f"Self-Studying - Epoch [{epoch+1}/{num_epochs_selfstudying}] - Loss: {running_s_ce_loss / len(train_loader):.4f}")

    # === Co-Studying ===
    print("\n=== Co-Studying ===\n")
    student.train()
    teacher.train()

    for epoch in range(num_epochs_costudying):
        running_s_loss, running_t_loss = 0.0, 0.0
        running_s_ce_loss, running_s_kd_loss = 0.0, 0.0
        running_t_ce_loss, running_t_kd_loss = 0.0, 0.0

        progress_bar = tqdm(train_loader, desc=f"Co-Studying - Epoch {epoch+1}/{num_epochs_costudying}", unit="batch")
        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_student.zero_grad()
            optimizer_teacher.zero_grad()

            teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)

            # Compute student losses
            s_ce_loss = F.cross_entropy(student_outputs, labels)
            s_kd_loss = F.kl_div(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs.detach() / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            student_loss = alpha_s * s_kd_loss + (1 - alpha_s) * s_ce_loss

            # Compute teacher losses
            t_ce_loss = F.cross_entropy(teacher_outputs, labels)
            t_kd_loss = F.kl_div(
                F.log_softmax(teacher_outputs / temperature, dim=1),
                F.softmax(student_outputs.detach() / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            teacher_loss = alpha_t * t_kd_loss + (1 - alpha_t) * t_ce_loss

            # Backpropagation
            student_loss.backward()
            optimizer_student.step()

            teacher_loss.backward()
            optimizer_teacher.step()

            # Accumulate loss values
            running_s_loss += student_loss.item()
            running_s_ce_loss += s_ce_loss.item()
            running_s_kd_loss += s_kd_loss.item()
            running_t_loss += teacher_loss.item()
            running_t_ce_loss += t_ce_loss.item()
            running_t_kd_loss += t_kd_loss.item()

            if (step + 1) % log_interval == 0:
                progress_bar.set_postfix({
                    "S_loss": f"{running_s_loss / (step + 1):.4f}",
                    "T_loss": f"{running_t_loss / (step + 1):.4f}",
                    "S_CE": f"{running_s_ce_loss / (step + 1):.4f}",
                    "T_CE": f"{running_t_ce_loss / (step + 1):.4f}",
                })

        scheduler_student.step()

        print(
            f"Co-Studying - Epoch [{epoch+1}/{num_epochs_costudying}] - "
            f"S_Loss: {running_s_loss / len(train_loader):.4f} | "
            f"T_Loss: {running_t_loss / len(train_loader):.4f} | "
            f"S_CE: {running_s_ce_loss / len(train_loader):.4f} | "
            f"T_CE: {running_t_ce_loss / len(train_loader):.4f}"
        )

    # === Tutoring ===
    print("\n=== Tutoring ===\n")
    student.train()
    teacher.eval()

    for epoch in range(num_epochs_tutoring):
        running_s_ce_loss, running_s_kd_loss = 0.0, 0.0

        progress_bar = tqdm(train_loader, desc=f"Tutoring - Epoch {epoch+1}/{num_epochs_tutoring}", unit="batch")
        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_student.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs = student(inputs)

            s_ce_loss = F.cross_entropy(student_outputs, labels)
            s_kd_loss = F.kl_div(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs.detach() / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            student_loss = alpha_s * s_kd_loss + (1 - alpha_s) * s_ce_loss

            student_loss.backward()
            optimizer_student.step()

            running_s_loss += student_loss.item()
            running_s_ce_loss += s_ce_loss.item()
            running_s_kd_loss += s_kd_loss.item()

            if (step + 1) % log_interval == 0:
                progress_bar.set_postfix({
                    "S_loss": f"{running_s_loss / (step + 1):.4f}",
                    "S_CE": f"{running_s_ce_loss / (step + 1):.4f}",
                    "S_KD": f"{running_s_kd_loss / (step + 1):.4f}",
                })

        scheduler_student.step()

        print(
            f"Tutoring - Epoch [{epoch+1}/{num_epochs_tutoring}] - "
            f"S_Loss: {running_s_loss / len(train_loader):.4f} | "
            f"S_CE: {running_s_ce_loss / len(train_loader):.4f}"
        )
        
    # Convert to quantized inference format
    student.to('cpu')
    student.eval()
    student = convert_fx(student)

    return student
############################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
############################################################################################
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
############################################################################################
def evaluate_model_quantized(model, data_loader, neval_batches=None, string_out=True):
    model.eval()
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    cnt = 0

    with torch.no_grad():
        for image, target in tqdm(data_loader, desc="Evaluating model"):
            output = model(image)
            cnt += 1

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].cpu().item(), image.size(0))
            top5.update(acc5[0].cpu().item(), image.size(0))

            if neval_batches is not None and cnt >= neval_batches:
                break

    if string_out:
        return f'top1: {top1.avg:.2f}, top5: {top5.avg:.2f}'
    else:
        return top1, top5
############################################################################################
def save_results_csv_quant(csv_path,
                     model,
                     base_acc,
                     ptq_acc,
                     qat_acc):
    """
    Saves the results to a CSV file. If the file exists, it appends a new row; otherwise, it creates the file.
    Headers: 'Teacher, Student, KD Algorithm, Alpha, Temperature, Teacher Accuracy, Student Accuracy, KD Student Accuracy'
    """
    headers = ['Model', 'Base Accuracy', 'PTQ Accuracy', 'QAT Accuracy']
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
   
        writer.writerow({
            'Model': model,
            'Base Accuracy': base_acc,
            'PTQ Accuracy': ptq_acc,
            'QAT Accuracy': qat_acc,
        })
    
    print(f"Saved results to {csv_path}")
############################################################################################
def save_results_csv_qkd(csv_path,
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