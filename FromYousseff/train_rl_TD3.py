"""Brave TD3 training script with Scheduled Exploration.

This version uses high noise for the first half of training to break local optima,
then decays back to normal noise for refinement.
"""

import os
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from config import LEARNING_RATE, N_ENVS, TD3_TOTAL_STEPS
from train_common import callbacks, get_device, make_training_env, parse_args, setup_dirs, MODELS_DIR

class NoiseScheduleCallback(BaseCallback):
    """Callback to decay action noise and target noise over time."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # After 1,000,000 steps, reduce noise for refinement
        if self.num_timesteps > 1_000_000:
            if hasattr(self.model, "action_noise") and self.model.action_noise is not None:
                # Target sigma 0.25 (Refinement)
                self.model.action_noise._sigma = 0.25 * np.ones_like(self.model.action_noise._sigma)
            # Reduce target policy noise
            self.model.target_policy_noise = 0.15
        return True

def main():
    args = parse_args(TD3_TOTAL_STEPS, default_envs=N_ENVS)
    device = args.device if args.device != "auto" else get_device(prefer_cuda=True)
    env = make_training_env(args.n_envs, domain_randomization=not args.no_randomization)
    logger = setup_dirs("td3")
    
    # We load from the latest 'best' or 'final' to keep current progress but un-jam it
    model_path = os.path.join(MODELS_DIR, "qube_td3_final.zip")
    
    n_actions = env.action_space.shape[-1]
    # START WITH BRAVE NOISE (sigma = 0.50)
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.50 * np.ones(n_actions))

    if os.path.exists(model_path) and not args.fresh:
        print(f"Loading TD3 from {model_path} for Brave-Exploration Refinement...")
        model = TD3.load(model_path, env=env, device=device, custom_objects={"action_noise": action_noise})
        # Explicitly set the brave parameters after loading
        model.action_noise = action_noise
        model.target_policy_noise = 0.25
    else:
        print("Starting fresh Brave TD3 training")
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            buffer_size=1_000_000,
            learning_starts=5_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"), # More intense training
            gradient_steps=1,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.25, # Braver target noise
            target_noise_clip=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            device=device,
        )

    model.set_logger(logger)
    print(f"Training Brave TD3 for {args.steps} steps on {device}")
    
    # Add the noise schedule callback to the list of callbacks
    cb_list = callbacks("td3", args.n_envs)
    cb_list.append(NoiseScheduleCallback())
    
    model.learn(total_timesteps=args.steps, callback=cb_list, reset_num_timesteps=False)
    
    final_save_path = os.path.join(MODELS_DIR, "qube_td3_final")
    model.save(final_save_path)
    env.close()
    print(f"Saved {final_save_path}.zip with Brave settings applied.")

if __name__ == "__main__":
    main()
