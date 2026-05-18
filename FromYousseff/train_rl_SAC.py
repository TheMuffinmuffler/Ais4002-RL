"""Standard SAC training script for QUBE-Servo 2.

Using user-provided parameters for stable training and refinement.
"""

import os
from stable_baselines3 import SAC
from config import N_ENVS, LEARNING_RATE, SAC_TOTAL_STEPS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs, MODELS_DIR

def main():
    args = parse_args(SAC_TOTAL_STEPS, default_envs=N_ENVS)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    logger = setup_dirs("sac")
    
    model_path = os.path.join(MODELS_DIR, "qube_sac_final.zip")

    if os.path.exists(model_path) and not args.fresh:
        print(f"Loading existing SAC model from {model_path}...")
        model = SAC.load(model_path, env=env, device=device)
    else:
        print("Starting fresh SAC training...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=512,
            tau=0.01,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=4,
            ent_coef="auto",
            target_update_interval=1,
            policy_kwargs=dict(net_arch=[512, 512, 512]),
            verbose=1,
            device=device,
        )

    model.set_logger(logger)
    print(f"Training SAC for {args.steps} steps on {device} with {args.n_envs} envs")
    
    # Use reset_num_timesteps=False to continue the existing timeline in logs
    model.learn(
        total_timesteps=args.steps, 
        callback=callbacks("sac", args.n_envs), 
        reset_num_timesteps=args.fresh
    )
    
    final_save_path = os.path.join(MODELS_DIR, "qube_sac_final")
    model.save(final_save_path)
    env.close()
    print(f"Saved {final_save_path}.zip")

if __name__ == "__main__":
    main()
