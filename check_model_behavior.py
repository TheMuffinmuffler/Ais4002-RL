import numpy as np
import torch
from stable_baselines3 import SAC
from qube_env import QubeEnv

def check():
    model_path = "models/qube_sac_final.zip"
    env = QubeEnv()
    try:
        model = SAC.load(model_path)
        print(f"Loaded {model_path}")
    except Exception as e:
        print(e)
        return

    obs, _ = env.reset()
    print(f"Initial obs: {obs}")
    
    total_reward = 0
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            theta = np.arctan2(obs[0], obs[1])
            alpha = np.arctan2(obs[2], obs[3])
            print(f"Step {step}: Theta={np.rad2deg(theta):.2f}, Alpha={np.rad2deg(alpha):.2f}, Action={action[0]:.2f}, Reward={reward:.2f}")
            
        if terminated or truncated:
            break
    
    print(f"Final Step: Theta={np.rad2deg(theta):.2f}, Alpha={np.rad2deg(alpha):.2f}")
    print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    check()
