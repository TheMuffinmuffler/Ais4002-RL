import os
import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
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

    # Configure logger to save to CSV and Tensorboard
    new_logger = configure("./logs/td3_v4/", ["stdout", "csv", "tensorboard"])

    device = get_device()
    print(f"Using device: {device}")

    # Use 8 environments for faster data collection
    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=8)
    
    eval_env = QubeEnv(domain_randomization=False)
    eval_env = Monitor(eval_env)

    n_actions = env.action_space.shape[-1]

    # Reduced noise for smoother actions
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4, # Lowered for stability
        buffer_size=300_000,
        learning_starts=10_000, # Increased for better initial buffer
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=50, # Update every 50 steps
        gradient_steps=50, # Do 50 updates at once
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.05, # Reduced from 0.2
        target_noise_clip=0.1,    # Reduced from 0.5
        verbose=1,
        device=device
    )
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="models/",
        name_prefix="qube_td3_checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/td3_eval/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )
    # Patch the callback to use a custom filename
    eval_callback.best_model_save_path = os.path.join("models/", "best_td3_model")

    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save("models/qube_td3_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()