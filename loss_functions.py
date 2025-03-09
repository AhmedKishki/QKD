import torch.nn.functional as F

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
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
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
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
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
    teacher_scaled = teacher_logits / temperature
    student_attention = student_scaled ** 2
    teacher_attention = teacher_scaled ** 2
    student_attention = F.normalize(student_attention, p=2, dim=1)
    teacher_attention = F.normalize(teacher_attention, p=2, dim=1)
    loss = F.mse_loss(student_attention, teacher_attention)
    return loss * (temperature ** 2)

def combined_loss(kd_loss_fn, student_logits, teacher_logits, labels, alpha, temperature):
    ce_loss = F.cross_entropy(student_logits, labels)
    if teacher_logits is None:
        return ce_loss
    kd_loss = kd_loss_fn(student_logits, teacher_logits.detach(), temperature)
    return (1 - alpha) * ce_loss + alpha * kd_loss
