"""Fast SAC training script.

This is the main controller. It is the best candidate for this project because it
usually learns swing-up + balance faster and more reliably than PPO/TD3.
"""

import os

from stable_baselines3 import SAC

from config import LEARNING_RATE, SAC_TOTAL_STEPS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs


def main():
    args = parse_args(SAC_TOTAL_STEPS, default_envs=8)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    logger = setup_dirs("sac")
    model_path = "models/qube_sac_final.zip"

    if os.path.exists(model_path) and not args.fresh:
        print(f"Loading SAC from {model_path}")
        model = SAC.load(model_path, env=env, device=device)
    else:
        print("Starting fresh SAC training")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            buffer_size=300_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            device=device,
        )

    model.set_logger(logger)
    print(f"Training SAC for {args.steps} steps on {device} with {args.n_envs} envs")
    model.learn(total_timesteps=args.steps, callback=callbacks("sac", args.n_envs), reset_num_timesteps=args.fresh)
    model.save("models/qube_sac_final")
    env.close()
    print("Saved models/qube_sac_final.zip")


if __name__ == "__main__":
    main()
