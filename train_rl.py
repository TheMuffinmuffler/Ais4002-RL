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

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger to save to CSV
    new_logger = configure("./logs/ppo_v4/", ["stdout", "csv", "tensorboard"])
    
    # Force CPU for PPO
    device = "cpu"
    print(f"Using device: {device}")
    
    # Increase number of environments
    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=8)

    # Evaluation environment
    eval_env = Monitor(QubeEnv(domain_randomization=False))
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="qube_ppo_checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/ppo_eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    # Patch the callback to use a custom filename
    eval_callback.best_model_save_path = os.path.join("models/", "best_ppo_model")
    
    # Linear schedule for learning rate
    def lr_schedule(progress_remaining: float):
        return 3e-4 * progress_remaining + 1e-5

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=512,
        n_epochs=15,
        ent_coef=0.01, # Encourage exploration
        device=device
    )
    model.set_logger(new_logger)
    
    total_steps = 1_000_000
    print(f"Starting training ({total_steps} steps) with random disturbances...")
    model.learn(total_timesteps=total_steps, callback=checkpoint_callback)
    
    model.save("models/qube_ppo_final")
    print("Training finished!")

if __name__ == "__main__":
    train()
