
import os
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from qube_env import QubeEnv
from compat import apply_compat_shims

def test_diag():
    print("Step 1: Applying shims...")
    apply_compat_shims()
    
    print("Step 2: Testing PyTorch/CUDA...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    print("Step 3: Testing Environment Initialization...")
    try:
        env = QubeEnv(domain_randomization=True)
        obs, _ = env.reset()
        print(f"Env reset successful. Obs shape: {obs.shape}")
    except Exception as e:
        print(f"Env init failed: {e}")
        return

    print("Step 4: Testing Model Loading...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "best_sac/best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, "qube_sac_final.zip")
        
    if os.path.exists(model_path):
        try:
            # Test loading on CPU first to see if it's a CUDA crash
            model = SAC.load(model_path, env=env, device="cpu")
            print("Model loaded successfully on CPU.")
            
            if torch.cuda.is_available():
                model = SAC.load(model_path, env=env, device="cuda")
                print("Model loaded successfully on CUDA.")
        except Exception as e:
            print(f"Model loading failed: {e}")
    else:
        print("No model found to test loading.")

    print("Step 5: Testing a single step...")
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"Step successful. Reward: {reward}")
    except Exception as e:
        print(f"Env step failed: {e}")

if __name__ == "__main__":
    test_diag()
