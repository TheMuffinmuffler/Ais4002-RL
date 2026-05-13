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
from train_rl import TD3_TOTAL_STEPS, N_ENVS, LEARNING_RATE


def get_device():
    return "cuda"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    new_logger = configure("./logs/td3_fresh_500k/", ["stdout", "csv", "tensorboard"])

    device = get_device()
    print(f"Using device: {device}")

    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=N_ENVS)
    eval_env = Monitor(QubeEnv(domain_randomization=False))

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model_path = "models/qube_td3_final.zip"

    custom_objects = {
        "observation_space": eval_env.observation_space,
        "action_space": eval_env.action_space
    }

    if False: # Forced fresh start due to obs space change
        print(f"Loading existing model {model_path} for retraining...")
        model = TD3.load(model_path, env=env, device=device, custom_objects=custom_objects)
    else:
        print("Starting FRESH TD3 training (Obs space changed)...")
        model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=LEARNING_RATE,
            buffer_size=300_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=50,
            gradient_steps=50,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.05,
            target_noise_clip=0.1,
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
        best_model_save_path=os.path.join("models/", "best_td3_model"),
        log_path="logs/td3_eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print(f"Starting TD3 training ({TD3_TOTAL_STEPS} steps)...")
    model.learn(
        total_timesteps=TD3_TOTAL_STEPS,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save("models/qube_td3_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
