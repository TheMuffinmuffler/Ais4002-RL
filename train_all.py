import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*20}")
    print(f"Starting {script_name}")
    print(f"{'='*20}\n")
    
    process = subprocess.Popen([sys.executable, script_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    if process.returncode != 0:
        print(f"Error: {script_name} failed with return code {process.returncode}")
    else:
        print(f"Finished {script_name} successfully")

def main():
    os.makedirs("logs", exist_ok=True)
    # PPO runs in parallel (on CPU)
    print("\nStarting PPO training in parallel (CPU)...")
    ppo_proc = subprocess.Popen([sys.executable, "train_rl.py"], stdout=open("logs/ppo_train.log", "w"), stderr=subprocess.STDOUT)
    
    # TD3 and SAC run sequentially (on GPU if available)
    sequential_scripts = ["train_rl_TD3.py", "train_rl_SAC.py"]
    
    for script in sequential_scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Warning: {script} not found, skipping.")
            
    print("\nSequential training finished. Waiting for PPO to complete...")
    ppo_proc.wait()
    print("All training completed!")

if __name__ == "__main__":
    main()
