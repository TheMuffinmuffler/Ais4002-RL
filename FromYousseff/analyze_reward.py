import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from qube_env import QubeEnv
from compat import apply_compat_shims
from train_common import MODELS_DIR, PLOTS_DIR

apply_compat_shims()

def analyze_reward_components():
    model_path = os.path.join(MODELS_DIR, "qube_sac_final.zip")
    env = QubeEnv(domain_randomization=False)
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    model = SAC.load(model_path, env=env)
    
    obs, _ = env.reset()
    # Force a swing-up start (hanging down)
    env.unwrapped.state = np.array([0.0, 0.0, 0.0, 0.0])
    obs = env.unwrapped._get_obs()
    
    history = []
    for step in range(200): # 200 steps is enough to see swing-up and fall
        action, _ = model.predict(obs, deterministic=True)
        
        # Manual step to capture components
        theta, alpha, th_dot, al_dot = env.unwrapped.state
        alpha_error = abs(env.unwrapped._wrap_pi(alpha - np.pi))
        r_swing = 0.5 * (1.0 - np.cos(alpha))
        upright_indicator = np.exp(-3.0 * alpha_error**2)
        in_bounds = np.exp(-((theta / (0.5 * np.pi)) ** 4))
        r_lock = 80.0 * upright_indicator * np.exp(-al_dot**2) * in_bounds
        r_centre = 20.0 * upright_indicator * np.exp(-theta**2)
        kick = -2.0 if np.cos(alpha) < -0.8 else 0.0
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if upright_indicator > 0.5:
            print(f"Step {step:3d}: Alpha: {np.rad2deg(alpha):7.2f} | al_dot: {al_dot:7.2f} | upright_ind: {upright_indicator:5.2f} | r_lock: {r_lock:5.2f}")

        history.append({
            'step': step,
            'alpha': np.rad2deg(alpha),
            'r_swing': r_swing,
            'r_lock': r_lock,
            'r_centre': r_centre,
            'kick': kick,
            'total_reward': reward
        })
        
        if terminated or truncated:
            break
            
    # Plotting
    import pandas as pd
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(df['step'], df['alpha'], label='Alpha (deg)')
    plt.axhline(180, color='r', linestyle='--', label='Upright')
    plt.legend()
    plt.title('Pendulum Angle')
    
    plt.subplot(3, 1, 2)
    plt.plot(df['step'], df['r_swing'], label='r_swing')
    plt.plot(df['step'], df['r_lock'], label='r_lock')
    plt.plot(df['step'], df['r_centre'], label='r_centre')
    plt.plot(df['step'], df['kick'], label='kick (penalty)')
    plt.legend()
    plt.title('Reward Components')
    
    plt.subplot(3, 1, 3)
    plt.plot(df['step'], df['total_reward'], label='Total Reward')
    plt.legend()
    plt.title('Total Reward')
    
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, 'reward_analysis.png')
    plt.savefig(plot_path)
    print(f"Saved {plot_path}")
    
    # Print some stats
    peak_lock = df['r_lock'].max()
    print(f"Peak r_lock: {peak_lock:.2f}")
    print(f"Steps with r_lock > 10: {len(df[df['r_lock'] > 10])}")

if __name__ == "__main__":
    analyze_reward_components()
