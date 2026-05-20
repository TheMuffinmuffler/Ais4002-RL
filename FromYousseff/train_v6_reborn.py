"""Targeted SAC retraining script for v6 environment overhaul.

Loads the 'AbsolutBest' model and retrains it with the new Elastic Safety 
and Survival Bonus environment.
"""

import os
import sys
from stable_baselines3 import SAC

# Ensure local directory is prioritized for imports
local_dir = os.path.dirname(os.path.abspath(__file__))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

from config import N_ENVS, SAC_TOTAL_STEPS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs, MODELS_DIR

def main():
    # Use 500k as the default for this specialized run
    args = parse_args(500_000, default_envs=N_ENVS)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    
    # Create the env with the new v6 logic (survival bonus, etc.)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    
    # Path to the specific target model (continuing from the first v6 run)
    target_model_path = os.path.join(MODELS_DIR, "qube_sac_v6_final.zip")
    
    if not os.path.exists(target_model_path):
        print(f"ERROR: Could not find target model at {target_model_path}")
        return

    print(f"Loading CONTINUATION model from {target_model_path}...")
    # This model already has ent_coef="auto", so we load it directly.
    model = SAC.load(target_model_path, env=env, device=device)
    
    # CRITICAL: Reset extreme entropy. 
    # The previous run hit ent_coef=50.1 due to chaos. 
    # For the new 2.5% poke environment, we reset it to 0.1 so it stops being so jittery.
    if hasattr(model, "log_ent_coef"):
        import torch
        import numpy as np
        print("Resetting extreme entropy (50.1 -> 0.1) for calmer environment...")
        with torch.no_grad():
            model.log_ent_coef.fill_(np.log(0.1))
    
    # Setup log directory
    logger = setup_dirs("sac_v6_refined")
    model.set_logger(logger)
    
    print(f"Starting V6 REFINE retraining for {args.steps} steps on {device}...")
    print("Environment: 2.5% Pokes, No Survival Bonus (0.0), Elastic Boundaries.")

    model.learn(
        total_timesteps=args.steps, 
        callback=callbacks("sac_v6_refined", args.n_envs), 
        reset_num_timesteps=False # Continue the learning curve
    )

    # Save as the new standard final model
    final_path = os.path.join(MODELS_DIR, "qube_sac_v6_refined.zip")
    model.save(final_path)
    print(f"Training complete. Saved to {final_path}")
    model.set_logger(logger)
    
    print(f"Starting V6 retraining for {args.steps} steps on {device}...")
    print("New Environment Features: No Survival Bonus (0.0), Elastic Boundaries, Full-Range Randomization.")

    model.learn(
        total_timesteps=args.steps, 
        callback=callbacks("sac_v6_reborn", args.n_envs), 
        reset_num_timesteps=False # Continue the learning curve
    )

    # Save as the new standard final model for FromYousseff
    final_path = os.path.join(MODELS_DIR, "qube_sac_v6_final.zip")
    model.save(final_path)
    print(f"Training complete. Saved to {final_path}")

if __name__ == "__main__":
    main()
