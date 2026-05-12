import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from qube_env import QubeEnv
import os
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# --- MASTER TRAINING CONFIGURATION ---
PPO_TOTAL_STEPS = 2000000
TD3_TOTAL_STEPS = 500000
SAC_TOTAL_STEPS = 1000000
N_ENVS = 8
LEARNING_RATE = 3e-4
# -------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    new_logger = configure("./logs/ppo_v4/", ["stdout", "csv", "tensorboard"])
    
    # Force CPU for PPO
    device = "cpu"
    print(f"Using device: {device}")
    
    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=N_ENVS)
    eval_env = Monitor(QubeEnv(domain_randomization=False))
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="qube_ppo_checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("models/", "best_ppo_model"),
        log_path="logs/ppo_eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    def lr_schedule(progress_remaining: float):
        return LEARNING_RATE * progress_remaining + 1e-5

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=512,
        n_epochs=15,
        ent_coef=0.01,
        device=device
    )
    model.set_logger(new_logger)
    
    print(f"Starting PPO training ({PPO_TOTAL_STEPS} steps)...")
    model.learn(total_timesteps=PPO_TOTAL_STEPS, callback=checkpoint_callback)
    
    model.save("models/qube_ppo_final")
    print("Training finished!")

if __name__ == "__main__":
    train()
