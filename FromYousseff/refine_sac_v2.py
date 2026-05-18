import os
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from config import N_ENVS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs, MODELS_DIR

class SoftLockEntropyCallback(BaseCallback):
    """Callback to decay entropy coefficient from 0.2 to 0.05 over 200k steps."""
    def __init__(self, total_refine_steps=200_000, start_ent=0.2, end_ent=0.05, verbose=0):
        super().__init__(verbose)
        self.total_refine_steps = total_refine_steps
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.start_timesteps = None

    def _on_step(self) -> bool:
        if self.start_timesteps is None:
            self.start_timesteps = self.num_timesteps
        
        progress = (self.num_timesteps - self.start_timesteps) / self.total_refine_steps
        progress = min(max(progress, 0.0), 1.0)
        
        current_ent = self.start_ent + (self.end_ent - self.start_ent) * progress
        
        # Update the model's entropy coefficient
        self.model.ent_coef = current_ent
        self.model.ent_coef_tensor = torch.tensor(float(current_ent), device=self.model.device)
        
        return True

def main():
    # Refinement parameters
    REFINE_STEPS = 200_000
    TARGET_LR = 1e-4
    
    args = parse_args(REFINE_STEPS, default_envs=N_ENVS)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    logger = setup_dirs("sac_refine_v2")
    
    best_path = os.path.join(MODELS_DIR, "best_sac", "best_model.zip")
    final_path = os.path.join(MODELS_DIR, "qube_sac_final.zip")
    model_path = best_path if os.path.exists(best_path) else final_path

    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        return

    print(f"--- SAC SOFT-LOCK REFINEMENT (v2) ---")
    print(f"Loading from: {model_path}")
    print(f"Entropy: 0.2 -> 0.05 (Soft-Lock Decay)")
    print(f"Learning Rate: {TARGET_LR} (Forced)")
    print(f"Gradient Steps: 2 (High Intensity)")
    print(f"Persistence Gate: 22.5 Degrees (Wider)")

    # Load model normally first
    model = SAC.load(model_path, env=env, device=device)
    
    # Manually overwrite parameters for refinement
    model.learning_rate = TARGET_LR
    model.ent_coef = 0.2 # Starting point
    model.gradient_steps = 2
    
    # Update the internal tensor used for the loss calculation
    model.ent_coef_tensor = torch.tensor(float(model.ent_coef), device=model.device)
    
    # Properly override the learning rate scheduler to ensure 1e-4 is used
    def constant_lr_schedule(_progress_remaining):
        return TARGET_LR
    model.lr_schedule = constant_lr_schedule
    
    # Disable entropy optimizer if it exists
    if hasattr(model, "ent_coef_optimizer") and model.ent_coef_optimizer is not None:
        print("Disabling internal entropy optimizer...")
        model.ent_coef_optimizer = None

    # Manually update optimizers lr
    for param_group in model.actor.optimizer.param_groups:
        param_group['lr'] = TARGET_LR
    for param_group in model.critic.optimizer.param_groups:
        param_group['lr'] = TARGET_LR

    model.set_logger(logger)
    
    # Callbacks
    cb_list = callbacks("sac_refine_v2", args.n_envs)
    cb_list.append(SoftLockEntropyCallback(total_refine_steps=REFINE_STEPS))
    
    print(f"Refining for {args.steps} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=cb_list,
        reset_num_timesteps=False
    )
    
    final_save_path = os.path.join(MODELS_DIR, "qube_sac_refined_v2")
    model.save(final_save_path)
    print(f"Refinement complete. Saved to {final_save_path}.zip")
    env.close()

if __name__ == "__main__":
    main()
