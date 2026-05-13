import numpy as np
import torch
from stable_baselines3 import PPO
from qube_env import QubeEnv

def check():
    model_path = "models/qube_ppo_final.zip"
    env = QubeEnv()
    try:
        model = PPO.load(model_path)
        print(f"Loaded {model_path}")
    except Exception as e:
        print(e)
        return

    obs, _ = env.reset()
    total_reward = 0
    max_alpha = 180
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        alpha = np.rad2deg(np.arctan2(obs[2], obs[3]))
        if abs(alpha) < max_alpha:
            max_alpha = abs(alpha)
        
    print(f"PPO Model: Min |Alpha| = {max_alpha:.2f}, Total Reward = {total_reward:.2f}")

if __name__ == "__main__":
    check()
