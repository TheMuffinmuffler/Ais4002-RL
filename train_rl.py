import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from qube_env import QubeEnv
import os
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train():
    os.makedirs("models", exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    
    # Increase number of environments
    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=8)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        device=device
    )
    
    # We increase training to 300k steps to handle the new "Disturbance" complexity
    total_steps = 300000
    print(f"Starting training ({total_steps} steps) with random disturbances...")
    model.learn(total_timesteps=total_steps)
    
    model.save("models/qube_ppo_final")
    print("Training finished!")

if __name__ == "__main__":
    train()
