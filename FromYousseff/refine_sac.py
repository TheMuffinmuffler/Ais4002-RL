import os
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from config import N_ENVS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs, MODELS_DIR

class FixedEntropyCallback(BaseCallback):
    """Surgical lock for refinement: Force ent_coef to a specific low value."""
    def __init__(self, target_ent=0.1, verbose=0):
        super().__init__(verbose)
        self.target_ent = target_ent

    def _on_step(self) -> bool:
        # Forcefully overwrite the entropy coefficient every step
        self.model.ent_coef = self.target_ent
        self.model.ent_coef_tensor = torch.tensor(float(self.target_ent), device=self.model.device)
        return True

def main():
    # Refinement Parameters: Surgical precision and smoothing
    REFINE_STEPS = 100_000
    TARGET_LR = 5e-5
    TARGET_ENT = 0.1 # Lock to low entropy for smooth motor control
    
    args = parse_args(REFINE_STEPS, default_envs=N_ENVS)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    logger = setup_dirs("sac_refine_smooth")
    
    # Base is the newly trained 1M step model
    base_path = os.path.join(MODELS_DIR, "qube_sac_final.zip")

    if not os.path.exists(base_path):
        print(f"Error: No model found at {base_path}")
        return

    print(f"--- SAC SURGICAL SMOOTHING REFINEMENT ---")
    print(f"Loading from: {base_path}")
    print(f"Locking Entropy to: {TARGET_ENT} (Smooth Control)")
    print(f"Refining LR: {TARGET_LR}")

    # Load model
    model = SAC.load(base_path, env=env, device=device)
    
    # Overwrite LR scheduler
    def constant_lr_schedule(_progress_remaining):
        return TARGET_LR
    model.lr_schedule = constant_lr_schedule
    
    # Disable entropy optimizer if it exists
    if hasattr(model, "ent_coef_optimizer") and model.ent_coef_optimizer is not None:
        print("Disabling internal entropy optimizer...")
        model.ent_coef_optimizer = None

    model.set_logger(logger)
    
    # Callbacks: Standard + Entropy Lock
    cb_list = callbacks("sac_refine_smooth", args.n_envs)
    cb_list.append(FixedEntropyCallback(target_ent=TARGET_ENT))
    
    print(f"Refining for {args.steps} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=cb_list,
        reset_num_timesteps=False # Continue logging timeline
    )
    
    final_save_path = os.path.join(MODELS_DIR, "qube_sac_refined")
    model.save(final_save_path)
    print(f"Refinement complete. Saved to {final_save_path}.zip")
    env.close()

if __name__ == "__main__":
    main()
