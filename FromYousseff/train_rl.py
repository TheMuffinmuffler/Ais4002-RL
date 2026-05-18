"""Fast PPO training script.

PPO is kept for comparison, but do not expect it to beat SAC on this plant.
"""

import os

from stable_baselines3 import PPO

from config import LEARNING_RATE, N_ENVS, PPO_TOTAL_STEPS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs, MODELS_DIR


def main():
    args = parse_args(PPO_TOTAL_STEPS, default_envs=N_ENVS)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    logger = setup_dirs("ppo")
    
    model_path = os.path.join(MODELS_DIR, "qube_ppo_final.zip")

    if os.path.exists(model_path) and not args.fresh:
        print(f"Loading PPO from {model_path}")
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("Starting fresh PPO training")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=1024,
            batch_size=512,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
        )

    model.set_logger(logger)
    print(f"Training PPO for {args.steps} steps on {device} with {args.n_envs} envs")
    model.learn(total_timesteps=args.steps, callback=callbacks("ppo", args.n_envs), reset_num_timesteps=args.fresh)
    
    final_save_path = os.path.join(MODELS_DIR, "qube_ppo_final")
    model.save(final_save_path)
    env.close()
    print(f"Saved {final_save_path}.zip")


if __name__ == "__main__":
    main()
