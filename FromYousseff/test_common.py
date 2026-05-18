import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from compat import apply_compat_shims
from config import ACTION_LIMIT
from qube_env import QubeEnv

apply_compat_shims()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(algo):
    algo = algo.lower()
    if algo == "ppo":
        from stable_baselines3 import PPO as Algo
    elif algo == "td3":
        from stable_baselines3 import TD3 as Algo
    elif algo == "sac":
        from stable_baselines3 import SAC as Algo
    else:
        raise ValueError(algo)
    model_path = os.path.join(MODELS_DIR, f"qube_{algo}_final.zip")
    return Algo.load(model_path, device=get_device()), model_path


def run_test(algo="sac", steps=500):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    try:
        model, model_path = _load(algo)
        print(f"Loaded {algo.upper()} model from {model_path}")
    except Exception as exc:
        print(f"Could not load {algo.upper()} model: {exc}")
        return

    env = QubeEnv(domain_randomization=False)
    obs, _ = env.reset()
    history = []
    total_reward = 0.0

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        theta = np.arctan2(obs[0], obs[1])
        alpha = np.arctan2(obs[2], obs[3])
        history.append([theta, alpha, obs[4], obs[5], float(np.asarray(action).reshape(-1)[0]), reward])
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps")
            break

    history = np.asarray(history, dtype=float)
    if history.size == 0:
        print("No data collected")
        return

    # Save history to CSV
    header = "theta,alpha,theta_dot,alpha_dot,action,reward"
    csv_path = os.path.join(PLOTS_DIR, f"{algo.lower()}_history.csv")
    np.savetxt(csv_path, history, delimiter=",", header=header, comments="")
    print(f"Saved {csv_path}")

    alpha_error = np.abs((history[:, 1] - np.pi + np.pi) % (2 * np.pi) - np.pi)
    upright = alpha_error < np.deg2rad(15.0)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Upright within ±15 deg: {100 * np.mean(upright):.1f}%")
    if np.any(upright):
        print(f"Mean |theta| while upright: {np.rad2deg(np.mean(np.abs(history[upright, 0]))):.2f} deg")
        print(f"Mean alpha error while upright: {np.rad2deg(np.mean(alpha_error[upright])):.2f} deg")

    cumulative_reward = np.cumsum(history[:, 5])
    x = np.arange(len(history))

    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(x, np.rad2deg(history[:, 0]), label="Theta (arm)")
    plt.plot(x, np.rad2deg(history[:, 1]), label="Alpha (pendulum)")
    plt.axhline(180, linestyle="--", alpha=0.4)
    plt.axhline(-180, linestyle="--", alpha=0.4)
    plt.axhline(0, linestyle="-", alpha=0.2)
    plt.ylabel("Angle (deg)")
    plt.title(f"QUBE-Servo 2 {algo.upper()} Simulation Test")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(x, np.rad2deg(history[:, 2]), label="Theta dot")
    plt.plot(x, np.rad2deg(history[:, 3]), label="Alpha dot")
    plt.ylabel("Velocity (deg/s)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    # Super-Env: history[:, 4] is normalized [-1, 1], scale it to ACTION_LIMIT for the plot
    voltage_history = history[:, 4] * ACTION_LIMIT
    plt.step(x, np.clip(voltage_history, -ACTION_LIMIT, ACTION_LIMIT), label="Action (voltage)")
    plt.ylabel("Voltage (V)")
    plt.ylim(-ACTION_LIMIT - 1, ACTION_LIMIT + 1)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(x, history[:, 5], label="Step reward", alpha=0.7)
    plt.plot(x, cumulative_reward / (x + 1), label="Average reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{algo.lower()}_dynamics.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved {plot_path}")
    plt.close()
    env.close()
