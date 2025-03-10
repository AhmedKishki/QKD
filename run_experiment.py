import argparse
import yaml

from quantization_knowledge_distillation import quantization_knowledge_distillation
from evaluation import evaluate_model_quantized
from helper import get_model, get_data_loaders, check_config_csv, save_results_csv

def main():
    parser = argparse.ArgumentParser(description="Run Quantization Knowledge Distillation from YAML config")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Load YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    experiment = config.get('experiment1', {})
    
    kd_losses = experiment.get('kd_loss', [])
    alpha_st_pairs = experiment.get('alpha_st_pairs', [])
    student_teacher_pairs = experiment.get('student_teacher_pairs', [])
    temperatures = experiment.get('temperatures', [6.0])
    num_epochs_list = experiment.get('num_epochs', [])
    max_lr = experiment.get('max_lr', 1e-3)
    min_lr = experiment.get('min_lr', 1e-6)
    teacher_lr = experiment.get('teacher_lr', 1e-6)
    device = experiment.get('device', 'cuda')
    batch_size = experiment.get('batch_size', 64)
    num_workers = experiment.get('num_workers', 16)
    train_dir = experiment.get('traindir', 'imageNet/train200')
    val_dir = experiment.get('validdir', 'imageNet/valid')
    
    for student_name, teacher_name in student_teacher_pairs:
        student = get_model(student_name)
        teacher = get_model(teacher_name)
        for kd_loss in kd_losses:
            for alpha_s, alpha_t in alpha_st_pairs:
                for temperature in temperatures:
                    for num_epochs in num_epochs_list:
                        num_epochs_selfstudying, num_epochs_costudying, num_epochs_tutoring = num_epochs
                        
                        if check_config_csv(csv_path, teacher_name, student_name, alpha_t, alpha_s, temperature, num_epochs):
                            print(f"Skipping {student_name} -> {teacher_name}, Alpha: ({alpha_s}, {alpha_t}), Temp: {temperature}")
                            continue
                        
                        csv_path = f"results_qkd_{kd_loss}.csv"
                        train_loader, valid_loader = get_data_loaders(train_dir, val_dir, batch_size, num_workers)

                        print(f"Running KD with {student_name} -> {teacher_name}, KD Loss: {kd_loss}, Alpha: ({alpha_s}, {alpha_t}), Temp: {temperature}")
                        
                        student = quantization_knowledge_distillation(
                            student=student,
                            teacher=teacher,
                            train_loader=train_loader,
                            device=device,
                            kd_loss=kd_loss,
                            num_epochs_selfstudying=num_epochs_selfstudying,
                            num_epochs_costudying=num_epochs_costudying,
                            num_epochs_tutoring=num_epochs_tutoring,
                            max_lr=max_lr,
                            min_lr=min_lr,
                            teacher_lr=teacher_lr,
                            alpha_s=alpha_s,
                            alpha_t=alpha_t,
                            temperature=temperature,
                            log_interval=100
                        )
                        
                        accuracy = evaluate_model_quantized(student, valid_loader)
                        save_results_csv(csv_path, teacher_name, student_name, alpha_s, alpha_t, temperature, num_epochs, accuracy)

if __name__ == "__main__":
    main()
