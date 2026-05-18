import os
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from config import N_ENVS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs, MODELS_DIR

class SoftLockEntropyCallback(BaseCallback):
    """Callback to decay entropy coefficient from 0.2 to 0.05 over total steps."""
    def __init__(self, total_refine_steps=400_000, start_ent=0.2, end_ent=0.05, verbose=0):
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
    # Refinement v3: Longer duration, higher precision focus
    REFINE_STEPS = 400_000
    TARGET_LR = 1e-4
    
    args = parse_args(REFINE_STEPS, default_envs=N_ENVS)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    logger = setup_dirs("sac_refine_v3")
    
    # Base is now the refined_v2 model
    base_path = os.path.join(MODELS_DIR, "qube_sac_refined_v2.zip")
    if not os.path.exists(base_path):
        # Fallback to best_sac if v2 isn't found
        base_path = os.path.join(MODELS_DIR, "best_sac", "best_model.zip")

    if not os.path.exists(base_path):
        print(f"Error: No model found at {base_path}")
        return

    print(f"--- SAC STABILITY REFINEMENT (v3) ---")
    print(f"Loading from: {base_path}")
    print(f"Entropy: 0.2 -> 0.05 (Over {REFINE_STEPS} steps)")
    print(f"Learning Rate: {TARGET_LR} (Forced)")
    print(f"Gradient Steps: 2")
    print(f"New Features: Speed Trap, 300.0 Stable Bonus, Arm Damping")

    # Load model normally first
    model = SAC.load(base_path, env=env, device=device)
    
    # Manually overwrite parameters for refinement
    model.learning_rate = TARGET_LR
    model.ent_coef = 0.2
    model.gradient_steps = 2
    
    # Update the internal tensor
    model.ent_coef_tensor = torch.tensor(float(model.ent_coef), device=model.device)
    
    # Override LR scheduler
    def constant_lr_schedule(_progress_remaining):
        return TARGET_LR
    model.lr_schedule = constant_lr_schedule
    
    # Disable entropy optimizer
    if hasattr(model, "ent_coef_optimizer") and model.ent_coef_optimizer is not None:
        print("Disabling internal entropy optimizer...")
        model.ent_coef_optimizer = None

    # Update optimizers
    for param_group in model.actor.optimizer.param_groups:
        param_group['lr'] = TARGET_LR
    for param_group in model.critic.optimizer.param_groups:
        param_group['lr'] = TARGET_LR

    model.set_logger(logger)
    
    # Callbacks
    cb_list = callbacks("sac_refine_v3", args.n_envs)
    cb_list.append(SoftLockEntropyCallback(total_refine_steps=REFINE_STEPS))
    
    print(f"Refining for {args.steps} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=cb_list,
        reset_num_timesteps=False
    )
    
    final_save_path = os.path.join(MODELS_DIR, "qube_sac_refined_v3")
    model.save(final_save_path)
    print(f"Refinement complete. Saved to {final_save_path}.zip")
    env.close()

if __name__ == "__main__":
    main()
