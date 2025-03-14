import os

import pandas as pd

import torch
import torch.nn.functional as F

from datetime import datetime

class Loss:
    def __init__(self, 
                 dataset,
                 kd_loss, 
                 num_epochs_selfstudying, 
                 num_epochs_costudying, 
                 num_epochs_tutoring,
                 directory="Losses"):
        
        self.directory = directory
        self.filename = f'{kd_loss}__{num_epochs_selfstudying}-{num_epochs_costudying}-{num_epochs_tutoring}_{datetime.now().strftime("%H%M%S")}_{dataset}.csv'
        os.makedirs(self.directory, exist_ok=True)
        
        self.student_loss = []
        self.teacher_loss = []
        self.student_ce_loss = []
        self.teacher_ce_loss = []
        self.student_kd_loss = []
        self.teacher_kd_loss = []

    def save(self):
        """
        Saves loss data into a CSV file.
        """
        df = pd.DataFrame({
            'epoch': range(len(self.student_loss)),
            'student_loss': self.student_loss,
            'teacher_loss': self.teacher_loss,
            'student_ce_loss': self.student_ce_loss,
            'teacher_ce_loss': self.teacher_ce_loss,
            'student_kd_loss': self.student_kd_loss,
            'teacher_kd_loss': self.teacher_kd_loss
        })
        
        filepath = os.path.join(self.directory, self.filename)
        df.to_csv(filepath, mode='w', header=True, index=False)
        
        print(f"Loss data saved to {filepath}")
        
    def print(self):
        """
        Prints the latest non-None loss values from non-empty lists.
        """
        losses = {
            "S_Loss": self.student_loss,
            "T_Loss": self.teacher_loss,
            "S_CE": self.student_ce_loss,
            "T_CE": self.teacher_ce_loss,
            "S_KD": self.student_kd_loss,
            "T_KD": self.teacher_kd_loss
        }

        # Extract only non-empty lists with a valid last value
        last_values = {key: lst[-1] for key, lst in losses.items() if lst and lst[-1] is not None}

        # If no valid values exist, print a message and return
        if not last_values:
            print("No valid loss values recorded.")
            return

        # Get the last recorded epoch number
        epoch = max(len(lst) for lst in losses.values() if lst) - 1

        # Format output dynamically
        formatted_values = " | ".join(f"{key}: {val:.4f}" for key, val in last_values.items())

        print(f"Epoch {epoch}: {formatted_values}")

def kl_loss(student_logits, teacher_logits, temperature):
    """
    Computes the Kullback-Leibler (KL) divergence loss between the softened
    logits of the student and teacher models. This loss helps the student model
    to mimic the teacher's output distribution (soft labels) during knowledge distillation.

    The logits are softened by scaling with a temperature parameter. A higher temperature
    produces a softer probability distribution over classes, which can carry more information
    about the inter-class similarities.

    Args:
        student_logits (Tensor): The output logits from the student model (before softmax).
        teacher_logits (Tensor): The output logits from the teacher model (before softmax).
        temperature (float): The temperature scaling parameter. Higher values lead to softer
                             probability distributions.

    Returns:
        Tensor: The KL divergence loss between the softened student and teacher outputs,
                scaled by the square of the temperature.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return loss * ( temperature ** 2 )

def cs_loss(student_logits, teacher_logits, temperature):
    """
    Computes the cosine similarity loss between the softmax outputs of student and teacher logits,
    applying temperature scaling to soften the distributions. The loss is defined as 1 minus the
    cosine similarity of the two probability distributions (averaged over the batch), and then scaled by T².

    Args:
        student_logits (Tensor): Logits output from the student model.
        teacher_logits (Tensor): Logits output from the teacher model.
        temperature (float): Temperature for scaling the logits in the softmax.

    Returns:
        Tensor: The computed cosine similarity loss.
    """
    student_prob = F.softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    loss = 1 - F.cosine_similarity(student_prob, teacher_prob, dim=1).mean()
    return loss * (temperature ** 2)

def ms_loss(student_logits, teacher_logits, temperature):
    """
    Computes a mean squared error (MSE) loss between attention maps derived from the
    student and teacher logits, using temperature scaling.

    Args:
        student_logits (Tensor): Logits from the student model with shape [B, C].
        teacher_logits (Tensor): Logits from the teacher model with shape [B, C].
        temperature (float): Temperature for scaling the logits.

    Returns:
        Tensor: The computed MSE loss.
    """
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits.detach()/ temperature
    student_attention = student_scaled ** 2
    teacher_attention = teacher_scaled ** 2
    student_attention = F.normalize(student_attention, p=2, dim=1)
    teacher_attention = F.normalize(teacher_attention, p=2, dim=1)
    loss = F.mse_loss(student_attention, teacher_attention)
    return loss * (temperature ** 2)

def js_loss(student_logits, teacher_logits, temperature):
    """
    Computes Jensen-Shannon Divergence (JS) loss between the student and teacher models.

    JSD measures the similarity between two probability distributions and is a symmetric,
    numerically stable alternative to KL divergence.

    Args:
        student_logits (Tensor): Logits from the student model with shape [B, C].
        teacher_logits (Tensor): Logits from the teacher model with shape [B, C].
        temperature (float): Temperature for scaling the logits.
        eps (float, optional): Small constant to prevent log(0) issues. Default is 1e-8.

    Returns:
        Tensor: The computed JSD loss.
    
    Notes:
        - The `0.5` factor is necessary because JSD is defined as the average of two KL divergences.
        - `teacher_logits` is detached to prevent unnecessary gradient computation.
        - `torch.clamp(mean_probs, min=eps)` prevents log(0) issues in KL divergence.
    """
    student_probs = F.softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    mean_probs = 0.5 * (student_probs + teacher_probs)
    mean_probs = torch.clamp(mean_probs, min=1e-9)
    loss = 0.5 * F.kl_div(F.log_softmax(student_logits / temperature, dim=1), mean_probs, reduction='batchmean') + \
           0.5 * F.kl_div(F.log_softmax(teacher_logits.detach() / temperature, dim=1), mean_probs, reduction='batchmean')
    return loss

def tv_loss(student_logits, teacher_logits, temperature):
    """
    Computes the Total Variation (TV) loss between the student and teacher model's
    softmax probability distributions.

    TV loss measures the absolute difference between two probability distributions.

    Args:
        student_logits (Tensor): Logits from the student model with shape [B, C].
        teacher_logits (Tensor): Logits from the teacher model with shape [B, C].
        temperature (float): Temperature for scaling the logits.

    Returns:
        Tensor: The computed TV loss.

    Notes:
        - Uses `softmax` to convert logits into probability distributions.
        - `teacher_logits` is detached to avoid unnecessary gradient computation.
        - Computes the mean absolute difference between the distributions.
    """
    student_probs = F.softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    loss = torch.mean(torch.abs(student_probs - teacher_probs))
    return loss

def combined_loss(kd_loss_fn, student_logits, teacher_logits, labels, alpha, temperature):
    """
    TV loss measures the difference in smoothness between the teacher and students probability distributions. 
    Its useful when you want the students distribution to match the teacher's overall shape rather than exact values.
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    if teacher_logits is None:
        return ce_loss
    kd_loss = kd_loss_fn(student_logits, teacher_logits.detach(), temperature)
    return (1 - alpha) * ce_loss + alpha * kd_loss
