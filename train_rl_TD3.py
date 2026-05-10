import os
import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from qube_env import QubeEnv


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_env():
    env = QubeEnv(domain_randomization=True)
    env = Monitor(env)
    return env


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    env = make_env()
    eval_env = QubeEnv(domain_randomization=False)
    eval_env = Monitor(eval_env)

    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=1.0 * np.ones(n_actions)
    )

    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        verbose=1,
        tensorboard_log="logs/td3_qube_tensorboard/",
        device=device
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="models/",
        name_prefix="qube_td3_checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save("models/qube_td3_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()