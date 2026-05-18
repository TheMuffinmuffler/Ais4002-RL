import os
import sys
import numpy as np
from stable_baselines3 import SAC
from qube_env import QubeEnv
from compat import apply_compat_shims

from train_common import MODELS_DIR

apply_compat_shims()

def run_eval(model_path, n_episodes=10, name="Model", force_no_bootcamp=False):
    print(f"\n--- Evaluating {name} from {model_path} ({n_episodes} eps) ---")
    if force_no_bootcamp:
        print("FORCE: 100% Random Starts (No Bootcamp)")
    
    env = QubeEnv(domain_randomization=False)
    
    # Custom loading
    try:
        model = SAC.load(model_path, env=env, custom_objects={"ent_coef": "auto"})
    except:
        model = SAC.load(model_path, env=env, custom_objects={"ent_coef": 0.01})

    all_rewards = []
    upright_pcts = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        if force_no_bootcamp:
            # Override to start hanging down or random
            env.unwrapped.state = np.array([
                env.unwrapped.np_random.uniform(-0.15, 0.15),
                env.unwrapped.np_random.uniform(-np.pi, np.pi),
                env.unwrapped.np_random.uniform(-0.5, 0.5),
                env.unwrapped.np_random.uniform(-1.0, 1.0)
            ])
            obs = env.unwrapped._get_obs()

        ep_reward = 0
        upright_steps = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            
            # Upright check (alpha error < 15 deg)
            alpha_wrapped = np.arctan2(obs[2], obs[3])
            alpha_error = np.abs((alpha_wrapped - np.pi + np.pi) % (2 * np.pi) - np.pi)
            if alpha_error < np.deg2rad(15.0):
                upright_steps += 1
            
            if terminated or truncated:
                break
        
        all_rewards.append(ep_reward)
        upright_pcts.append(upright_steps / 5.0) # /500 * 100

    print(f"Results: Mean Reward: {np.mean(all_rewards):.2f} | Mean Upright: {np.mean(upright_pcts):.1f}%")
    return np.mean(all_rewards)

if __name__ == "__main__":
    best_path = os.path.join(MODELS_DIR, "best_sac/best_model.zip")
    final_path = os.path.join(MODELS_DIR, "qube_sac_final.zip")
    
    # 1. Best vs Final
    run_eval(best_path, 10, "BEST SAC")
    run_eval(final_path, 10, "FINAL SAC")
    
    # 2. Robustness (50 eps)
    run_eval(best_path, 50, "BEST SAC ROBUSTNESS")
    
    # 3. Random Starts Stress Test
    run_eval(best_path, 20, "BEST SAC STRESS TEST", force_no_bootcamp=True)
