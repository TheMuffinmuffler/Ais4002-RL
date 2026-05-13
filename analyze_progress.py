import numpy as np
from qube_env import QubeEnv
from stable_baselines3 import SAC
import glob
import os

import sys

def analyze_best_checkpoint(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoints = glob.glob("models/qube_sac_fresh_*.zip")
        if not checkpoints:
            print("No checkpoints found yet.")
            return
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    else:
        latest_checkpoint = checkpoint_path
        
    print(f"Analyzing {latest_checkpoint}...")
    
    env = QubeEnv(domain_randomization=False)
    model = SAC.load(latest_checkpoint)
    
    for mode in ["Deterministic", "Stochastic"]:
        print(f"\n--- {mode} Evaluation ---")
        obs, _ = env.reset()
        history = []
        total_reward = 0
        
        is_det = (mode == "Deterministic")
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=is_det)
            obs, reward, terminated, truncated, info = env.step(action)
            theta, alpha, th_dot, al_dot = env.state
            history.append([theta, alpha, th_dot, al_dot, reward])
            total_reward += reward
                
        history = np.array(history)
        max_alpha_reached = np.rad2deg(np.min(np.abs(history[:, 1])))
        max_al_dot = np.max(np.abs(history[:, 3]))
        
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Min |Alpha|: {max_alpha_reached:.2f} degrees")
        print(f"Max |Alpha Dot|: {max_al_dot:.2f} rad/s")
    
    # Behavior Deduction
    if max_al_dot > 5.0:
        print("Behavior: Agent is AGGRESSIVELY SWINGING.")
    elif max_al_dot > 1.0:
        print("Behavior: Agent is GENTLY ROCKING.")
    else:
        print("Behavior: Agent is NEARLY STILL.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_best_checkpoint(path)
