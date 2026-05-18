import sys
import os

# --- INLINE COMPATIBILITY SHIM ---
def apply_compat_shims():
    try:
        import numpy.core.numeric as numeric
        sys.modules['numpy._core.numeric'] = numeric
        import numpy.core.multiarray as multiarray
        sys.modules['numpy._core.multiarray'] = multiarray
        import numpy.core.umath as umath
        sys.modules['numpy._core.umath'] = umath
        try:
            import numpy._core
        except ImportError:
            import numpy.core as core
            sys.modules['numpy._core'] = core
    except (ImportError, AttributeError): pass
    try:
        import stable_baselines3.common.utils as sb3_utils
        for name in ["FloatSchedule", "ConstantSchedule", "LinearSchedule"]:
            if not hasattr(sb3_utils, name):
                class DummySchedule:
                    def __init__(self, value=0.0, *args, **kwargs): self.value = value
                    def __call__(self, *args, **kwargs): return self.value
                setattr(sb3_utils, name, DummySchedule)
    except ImportError: pass
    try:
        import numpy.random._pickle as nprp
        for bg_name in ['MT19937', 'PCG64', 'PCG64DXSM', 'Philox', 'SFC64']:
            if hasattr(nprp, bg_name):
                bg_cls = getattr(nprp, bg_name)
                if hasattr(nprp, 'BitGenerators'):
                    nprp.BitGenerators[bg_cls] = bg_cls
    except Exception: pass
    try:
        from gymnasium.spaces.space import Space
        def patched_setstate(self, state):
            if isinstance(state, dict): self.__dict__.update(state)
        Space.__setstate__ = patched_setstate
    except (ImportError, AttributeError): pass

apply_compat_shims()
# --------------------------------

import numpy as np
import torch
import pickle
import cloudpickle
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, TD3, SAC
from qube_env import QubeEnv

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def evaluate_model(algo_name, model_class, model_path, n_episodes=5):
    device = get_device()
    if algo_name == "PPO":
        device = "cpu" # Force CPU as requested for PPO
        
    print(f"\nEvaluating {algo_name}...")
    env = QubeEnv()
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space
    }
    try:
        model = model_class.load(model_path, env=env, device=device, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading {algo_name}: {e}")
        env.close()
        return None

    all_rewards = []
    all_upright_pcts = []
    
    # Store trajectories for the first episode for plotting
    first_episode_data = None

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_history = []
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Reconstruction: user_alpha=0 is UP
            theta = np.arctan2(obs[0], obs[1])
            alpha = np.arctan2(obs[2], obs[3])
            
            ep_history.append([theta, alpha, obs[4], obs[5], float(action[0]), reward])
            ep_reward += reward
            
            if terminated or truncated:
                break
        
        ep_history = np.array(ep_history)
        all_rewards.append(ep_reward)
        
        # Calculate upright % (0 is UP in new coords)
        upright_mask = np.cos(ep_history[:, 1]) > 0.9
        upright_pct = (np.sum(upright_mask) / len(ep_history)) * 100
        all_upright_pcts.append(upright_pct)
        
        if ep == 0:
            first_episode_data = ep_history

    env.close()
    
    return {
        "name": algo_name,
        "avg_reward": np.mean(all_rewards),
        "avg_upright_pct": np.mean(all_upright_pcts),
        "trajectory": first_episode_data
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    models = [
        ("PPO", PPO, os.path.join(models_dir, "qube_ppo_final.zip")),
        ("TD3", TD3, os.path.join(models_dir, "qube_td3_final.zip")),
        ("SAC", SAC, os.path.join(models_dir, "qube_sac_final.zip"))
    ]
    
    results = []
    for name, m_class, path in models:
        if os.path.exists(path):
            res = evaluate_model(name, m_class, path)
            if res:
                results.append(res)
        else:
            print(f"Skipping {name}: model file not found at {path}")

    if not results:
        print("No models were successfully evaluated.")
        return

    # Print Summary Table
    print("\n" + "="*50)
    print(f"{'Algorithm':<12} | {'Avg Reward':<12} | {'Upright %':<12}")
    print("-" * 50)
    for res in results:
        print(f"{res['name']:<12} | {res['avg_reward']:<12.2f} | {res['avg_upright_pct']:<12.2f}%")
    print("="*50 + "\n")

    # Plot Comparison
    plt.figure(figsize=(15, 10))
    
    # Theta Plot
    plt.subplot(3, 1, 1)
    for res in results:
        plt.plot(np.rad2deg(res['trajectory'][:, 0]), label=res['name'])
    plt.ylabel("Theta (deg)")
    plt.title("Arm Angle Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Alpha Plot
    plt.subplot(3, 1, 2)
    for res in results:
        plt.plot(np.rad2deg(res['trajectory'][:, 1]), label=res['name'])
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.ylabel("Alpha (deg)")
    plt.title("Pendulum Angle Comparison (0 is Upright)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Action Plot
    plt.subplot(3, 1, 3)
    for res in results:
        plt.step(range(len(res['trajectory'])), res['trajectory'][:, 4], label=res['name'], alpha=0.7)
    plt.ylabel("Voltage (V)")
    plt.xlabel("Step")
    plt.title("Action (Voltage) Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_plot_path = os.path.join(plots_dir, "comparison_results.png")
    plt.savefig(comparison_plot_path, dpi=200)
    print(f"Comparison plot saved to {comparison_plot_path}")

if __name__ == "__main__":
    main()
