import os
import sys
import argparse
import datetime

# Ensure local directory is prioritized for imports (qube_env, config, etc.)
local_dir = os.path.dirname(os.path.abspath(__file__))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, TD3
from qube_env import QubeEnv
from compat import apply_compat_shims
from config import ACTION_LIMIT
from train_common import MODELS_DIR

apply_compat_shims()

def get_model_path(algo="sac", use_best=False):
    """
    Finds a model for a given algorithm.
    Priority:
    1. qube_{algo}_final.zip (Final model)
    2. best_{algo}/best_model.zip (Best model from training)
    
    If use_best is True, best_{algo}/best_model.zip is prioritized.
    """
    final_path = os.path.join(MODELS_DIR, f"qube_{algo}_final.zip")
    best_path = os.path.join(MODELS_DIR, f"best_{algo}/best_model.zip")
    
    if use_best:
        candidates = [best_path, final_path]
    else:
        candidates = [final_path, best_path]
        
    for path in candidates:
        if os.path.exists(path):
            return path
            
    # Fallback to any zip in MODELS_DIR
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".zip") and algo in f.lower():
                return os.path.join(MODELS_DIR, f)
                
    return None

def evaluate_detailed(algo="sac", n_episodes=10, save_plots=True, model_path_override=None, use_best=False):
    if model_path_override:
        model_path = model_path_override
    else:
        model_path = get_model_path(algo, use_best=use_best)
        
    if not model_path or not os.path.exists(model_path):
        print(f"No models found for {algo.upper()}" + (f" at {model_path}" if model_path else ""))
        return None
    
    print(f"\n--- Evaluating {algo.upper()} from {model_path} ---")
    
    # Create the environment with domain randomization enabled to test robustness (pushes)
    env = QubeEnv(domain_randomization=True)

    # Robust loading for all algorithms
    custom_objects = {}
    if algo == "sac":
        custom_objects["ent_coef"] = "auto" 

    try:
        if algo == "sac":
            model = SAC.load(model_path, env=env, custom_objects=custom_objects)
        elif algo == "ppo":
            model = PPO.load(model_path, env=env, device="cpu")
        elif algo == "td3":
            model = TD3.load(model_path, env=env)
    except Exception as e:
        print(f"Standard load failed for {algo.upper()}, attempting robust fallback... Error: {e}")
        try:
            if algo == "sac":
                custom_objects["ent_coef"] = 0.01
                model = SAC.load(model_path, env=env, custom_objects=custom_objects)
            elif algo == "ppo":
                model = PPO.load(model_path)
                model.set_env(env)
            elif algo == "td3":
                model = TD3.load(model_path)
                model.set_env(env)
        except Exception as e2:
            print(f"Critical: Failed to load {algo.upper()} model. Error: {e2}")
            return None

    # Create a timestamped directory for this evaluation run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path_plots = os.path.join(local_dir, "plots")
    run_dir = os.path.join(base_path_plots, f"eval_{timestamp}")
    if save_plots:
        os.makedirs(run_dir, exist_ok=True)
        print(f"Saving all episode plots to: {run_dir}")

    cases = ["upright", "downright"]
    case_results = {}

    for case in cases:
        print(f"\nEvaluating Case: {case.upper()}")
        all_rewards = []
        all_upright_pcts = []
        all_max_alphas = []

        for ep in range(n_episodes):
            env.reset()
            # Force specific start state
            if case == "upright":
                theta = 0.0
                alpha = np.pi + np.random.uniform(-0.05, 0.05)
            else:
                theta = 0.0
                alpha = 0.0 + np.random.uniform(-0.05, 0.05)
            
            theta_dot = 0.0
            alpha_dot = 0.0
            
            env.unwrapped.state = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float64)
            obs = env.unwrapped._get_obs()
            
            start_alpha = env.unwrapped.state[1]
            ep_reward = 0
            upright_count = 0
            swingup_step = None
            max_alpha = 0.0
            prev_voltage = 0.0
            ep_history = []

            for step in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Extract components directly from environment info
                comps = info.get("reward_components", {})
                
                theta_arm = np.arctan2(obs[0], obs[1])
                alpha_raw = env.unwrapped.state[1]
                max_alpha = max(max_alpha, abs(alpha_raw))
                
                alpha_error = np.abs((np.arctan2(obs[2], obs[3]) - np.pi + np.pi) % (2 * np.pi) - np.pi)
                is_upright = alpha_error < np.deg2rad(15.0)
                
                if is_upright:
                    upright_count += 1
                    if swingup_step is None:
                        swingup_step = step
                
                norm_action = float(np.asarray(action).reshape(-1)[0])
                is_hit = 1.0 if info.get("hit", False) else 0.0
                
                ep_history.append([
                    theta_arm, alpha_raw, obs[4], obs[5], 
                    norm_action, 
                    reward, 
                    comps.get("r_swing", 0.0), 
                    comps.get("r_balance", 0.0), 
                    comps.get("r_persistence", 0.0), 
                    comps.get("departure_tax", 0.0), 
                    comps.get("p_center", 0.0), 
                    comps.get("p_effort", 0.0),
                    comps.get("p_smooth", 0.0), 
                    comps.get("p_boundary", 0.0), 
                    is_hit
                ])
                
                ep_reward += reward
                if terminated or truncated:
                    break
            
            upright_pct = 100.0 * upright_count / (step + 1)
            all_rewards.append(ep_reward)
            all_upright_pcts.append(upright_pct)
            all_max_alphas.append(max_alpha)
            
            term_str = "TERM" if terminated else "MAX "
            print(f"Ep {ep:2d}: Start Alpha: {np.rad2deg(start_alpha):7.2f} | Max Alpha: {np.rad2deg(max_alpha):7.1f} | Steps: {step+1:3d} ({term_str}) | Rew: {ep_reward:8.2f} | Upright: {upright_pct:5.1f}% | Swingup: {swingup_step}")

            if save_plots and ep_history:
                history = np.array(ep_history)
                x = np.arange(len(history))
                plt.figure(figsize=(12, 18))
                
                plt.subplot(6, 1, 1)
                plt.plot(x, np.rad2deg(history[:, 0]), label="Theta (arm)")
                plt.plot(x, np.rad2deg(history[:, 1]), label="Alpha Raw (pendulum)")
                if len(np.where(history[:, 14] > 0)[0]) > 0:
                    plt.scatter(x[np.where(history[:, 14] > 0)], np.rad2deg(history[np.where(history[:, 14] > 0), 1]), 
                                color='red', marker='x', s=100, label="Random Poke")
                plt.axhline(180, color='r', linestyle='--', alpha=0.3)
                plt.axhline(-180, color='r', linestyle='--', alpha=0.3)
                plt.axhline(400, color='k', linestyle=':', label="Limit (400)")
                plt.axhline(-400, color='k', linestyle=':')
                plt.ylabel("Angle (deg)")
                plt.title(f"{algo.upper()} - {case.upper()} - Ep {ep}")
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.subplot(6, 1, 2)
                plt.plot(x, np.rad2deg(history[:, 2]), label="Theta dot")
                plt.plot(x, np.rad2deg(history[:, 3]), label="Alpha dot")
                plt.ylabel("Velocity (deg/s)")
                plt.legend(); plt.grid(True, alpha=0.3)

                plt.subplot(6, 1, 3)
                plt.step(x, history[:, 4] * ACTION_LIMIT, label="Voltage")
                plt.ylabel("Voltage (V)"); plt.legend(); plt.grid(True, alpha=0.3)

                plt.subplot(6, 1, 4)
                plt.plot(x, history[:, 6], label="r_swing")
                plt.plot(x, history[:, 7], label="r_balance")
                plt.plot(x, history[:, 8], label="r_persistence", color='gold', linewidth=2)
                plt.ylabel("Positive Reward"); plt.legend(); plt.grid(True, alpha=0.3)

                plt.subplot(6, 1, 5)
                plt.plot(x, history[:, 9], label="Departure Tax", color='red', linewidth=2)
                plt.plot(x, history[:, 10], label="p_center", color='purple')
                plt.plot(x, history[:, 11], label="p_effort", color='brown')
                plt.plot(x, history[:, 12], label="p_smooth", color='cyan')
                plt.plot(x, history[:, 13], label="p_boundary", color='orange')
                plt.ylabel("Penalty"); plt.legend(ncol=2, fontsize='small'); plt.grid(True, alpha=0.3)

                plt.subplot(6, 1, 6)
                total_rew = np.cumsum(history[:, 5])
                plt.plot(x, total_rew, label="Return", color='blue')
                plt.ylabel("Reward"); plt.xlabel("Step"); plt.legend(); plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, f"{algo}_{case}_ep{ep:02d}.png"), dpi=150)
                plt.close()

        case_results[case] = {"mean_rew": np.mean(all_rewards), "mean_upright": np.mean(all_upright_pcts), "mean_max_alpha": np.mean(all_max_alphas)}

    return {"algo": algo, "upright": case_results["upright"], "downright": case_results["downright"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac")
    parser.add_argument("--eps", type=int, default=10)
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    
    res = evaluate_detailed(algo=args.algo.lower(), n_episodes=args.eps, use_best=args.best, model_path_override=args.path)
    if res:
        print(f"\nEvaluation of {args.algo.upper()} complete.")
