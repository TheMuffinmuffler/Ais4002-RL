import os
import sys
from stable_baselines3 import SAC, TD3
from compat import apply_compat_shims

apply_compat_shims()

def inspect_sac(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    print(f"\n--- Inspecting SAC: {path} ---")
    try:
        model = SAC.load(path, device="cpu")
        print(f"  Observation Space: {model.observation_space}")
        print(f"  Ent Coef: {model.ent_coef}")
        if hasattr(model, "log_ent_coef"):
            import torch
            print(f"  Log Ent Coef: {model.log_ent_coef.item()}")
            print(f"  Actual Ent Coef: {torch.exp(model.log_ent_coef).item():.6f}")
        print(f"  Learning Rate: {model.learning_rate}")
    except Exception as e:
        print(f"  Error: {e}")

def inspect_td3(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    print(f"\n--- Inspecting TD3: {path} ---")
    try:
        model = TD3.load(path, device="cpu")
        print(f"  Observation Space: {model.observation_space}")
        print(f"  Learning Rate: {model.learning_rate}")
        print(f"  Action Noise: {model.action_noise}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    models_dir = "FromYousseff/models"
    inspect_sac(os.path.join(models_dir, "qube_sac_final.zip"))
    inspect_sac(os.path.join(models_dir, "best_sac/best_model.zip"))
    inspect_td3(os.path.join(models_dir, "qube_td3_final.zip"))
