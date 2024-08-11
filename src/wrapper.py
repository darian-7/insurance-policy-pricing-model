import subprocess
import os

def run_script(script_path):
    try:
        print(f"Running {script_path}...")
        subprocess.run(['python3', script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")

if __name__ == "__main__":
    # Absolute paths to the scripts
    base_dir = "/Users/darian/Desktop/C5i docs/C5i Code/insurance-policy-pricing-model/src"
    preprocessing_script = os.path.join(base_dir, 'preprocessing.py')
    training_script = os.path.join(base_dir, 'sagemaker_training.py')
    
    # Run preprocessing.py
    run_script(preprocessing_script)
    
    # Run sagemaker_training.py
    run_script(training_script)