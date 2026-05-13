"""Fast TD3 training script.

TD3 is deterministic and can work, but it is less forgiving than SAC here.
"""

import os
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from config import LEARNING_RATE, TD3_TOTAL_STEPS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs


def main():
    args = parse_args(TD3_TOTAL_STEPS, default_envs=4)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    logger = setup_dirs("td3")
    model_path = "models/qube_td3_final.zip"
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))

    if os.path.exists(model_path) and not args.fresh:
        print(f"Loading TD3 from {model_path}")
        model = TD3.load(model_path, env=env, device=device)
    else:
        print("Starting fresh TD3 training")
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            buffer_size=300_000,
            learning_starts=8_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.15,
            target_noise_clip=0.35,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            device=device,
        )

    model.set_logger(logger)
    print(f"Training TD3 for {args.steps} steps on {device} with {args.n_envs} envs")
    model.learn(total_timesteps=args.steps, callback=callbacks("td3", args.n_envs), reset_num_timesteps=args.fresh)
    model.save("models/qube_td3_final")
    env.close()
    print("Saved models/qube_td3_final.zip")


if __name__ == "__main__":
    main()
