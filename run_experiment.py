#!/usr/bin/env python

import sys
import yaml

def process_experiment(experiment_config):
    # Obtain the list of kd_loss values
    kd_loss = experiment_config.get("kd_loss", [])
    
    # Process alpha_st_pairs: split into alpha_s and alpha_t
    alpha_st_pairs = experiment_config.get("alpha_st_pairs", [])
    
    # Preserve student_teacher_pairs as intact pairs
    student_teacher_pairs = experiment_config.get("student_teacher_pairs", [])
    
    # Get temperatures list
    temperatures = experiment_config.get("temperatures", [])
    
    # Keep num_epochs as triples without splitting
    num_epochs = experiment_config.get("num_epochs", [])
    
    # Other hyperparameters
    max_lr = experiment_config.get("max_lr")
    min_lr = experiment_config.get("min_lr")
    teacher_lr = experiment_config.get("teacher_lr")
    
    # Print out the extracted parameters
    print("kd_loss:", kd_loss)
    print("alpha_st_pairs:", alpha_st_pairs)
    print("student_teacher_pairs:", student_teacher_pairs)
    print("temperatures:", temperatures)
    print("num_epochs_triples:", num_epochs)
    print("max_lr:", max_lr)
    print("min_lr:", min_lr)
    print("teacher_lr:", teacher_lr)
    print()

def main(yaml_file):
    try:
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        sys.exit(1)
    
    # Iterate over each experiment in the configuration
    for experiment_name, experiment_config in config.items():
        print(f"--- Running {experiment_name} ---")
        process_experiment(experiment_config)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_experiments.py <path_to_yaml_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    main(yaml_file)
