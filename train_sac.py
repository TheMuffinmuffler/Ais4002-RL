import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from qube_env import QubeEnv
import os
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger
    new_logger = configure("./logs/sac_v1/", ["stdout", "csv", "tensorboard"])
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # SAC is off-policy, so it usually benefits from fewer environments but higher quality steps
    # We'll use 1 environment with a Monitor wrapper for evaluation tracking
    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=1)

    eval_env = Monitor(QubeEnv(domain_randomization=False))
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="qube_sac_checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_sac_model",
        log_path="logs/sac_eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # SAC Hyperparameters optimized for Pendulum tasks
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto", # Automatically tune entropy (critical for SAC)
        target_update_interval=1,
        device=device
    )
    model.set_logger(new_logger)
    
    total_steps = 500000 # SAC usually converges faster than PPO
    print(f"Starting SAC training ({total_steps} steps) with Domain Randomization...")
    model.learn(total_timesteps=total_steps, callback=[checkpoint_callback, eval_callback])
    
    model.save("models/qube_sac_final")
    print("Training finished! Model saved to models/qube_sac_final.zip")

if __name__ == "__main__":
    train()
