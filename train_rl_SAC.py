import os
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from qube_env import QubeEnv


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Configure logger
    new_logger = configure("./logs/sac_v3/", ["stdout", "csv", "tensorboard"])

    device = get_device()
    print(f"Using device: {device}")

    # Use 8 environments for collection
    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=8)
    
    eval_env = QubeEnv(domain_randomization=False)
    eval_env = Monitor(eval_env)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=50,
        gradient_steps=50,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        device=device
    )
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="models/",
        name_prefix="qube_sac_checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_sac_model",
        log_path="logs/sac_eval/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    total_steps = 500_000
    print(f"Starting SAC training ({total_steps} steps)...")
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save("models/qube_sac_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
