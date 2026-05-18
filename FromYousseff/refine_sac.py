"""SAC Refinement Script: The Sniper (Bulletproof v2).

Loads the best v6.9.3 model and manually overrides the entropy 
to a fixed value to eliminate high-frequency vibration.
"""

import os
import torch as th
from stable_baselines3 import SAC
from train_common import callbacks, get_device, make_training_env, setup_dirs, MODELS_DIR

def main():
    device = get_device(prefer_cuda=True)
    n_envs = 12
    env = make_training_env(n_envs, domain_randomization=True)
    logger = setup_dirs("sac_refine")
    
    model_path = os.path.join(MODELS_DIR, "qube_sac_final.zip")

    if not os.path.exists(model_path):
        print(f"Error: Could not find base model at {model_path}")
        return

    print(f"--- Brain Freeze: Refining {model_path} ---")
    
    # 1. Load model normally
    model = SAC.load(model_path, env=env, device=device)

    # 2. Manually "Freeze" the entropy brain
    # Fixed entropy at 0.02 stops the random twitching
    print("Locking entropy to 0.02 and setting LR to 1e-4...")
    fixed_ent = 0.02
    model.ent_coef = fixed_ent
    
    # SB3 uses ent_coef_tensor internally during the train() call
    model.ent_coef_tensor = th.tensor([fixed_ent], device=device).reshape(1)
    model.log_ent_coef = th.log(model.ent_coef_tensor).requires_grad_(False)
    
    # Disable entropy optimization
    model.ent_coef_optimizer = None 
    
    # 3. Correctly update Learning Rate for SAC (Actor and Critic have separate optimizers)
    new_lr = 0.0001
    model.learning_rate = new_lr
    model.lr_schedule = lambda _: new_lr
    
    # Update Actor optimizer
    for param_group in model.actor.optimizer.param_groups:
        param_group['lr'] = new_lr
    
    # Update Critic optimizer
    for param_group in model.critic.optimizer.param_groups:
        param_group['lr'] = new_lr

    model.set_logger(logger)
    
    # 500k steps for the final "Lock-In"
    refine_steps = 500_000
    print(f"Starting refinement for {refine_steps} steps...")
    
    model.learn(
        total_timesteps=refine_steps, 
        callback=callbacks("sac_refine", n_envs), 
        reset_num_timesteps=True
    )
    
    final_save_path = os.path.join(MODELS_DIR, "qube_sac_final")
    model.save(final_save_path)
    env.close()
    print(f"Refinement complete. Saved improved model to {final_save_path}.zip")

if __name__ == "__main__":
    main()
