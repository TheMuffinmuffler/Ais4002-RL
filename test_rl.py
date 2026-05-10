import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from qube_env import QubeEnv
import matplotlib.pyplot as plt

def test():
    # Load the final model
    try:
        model_path = "models/qube_ppo_final.zip"
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return
    
    env = QubeEnv()
    obs, _ = env.reset()
    history = []
    total_reward = 0.0
    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        # Recover theta and alpha from obs [sin_th, cos_th, sin_al, cos_al, ...]
        theta = np.arctan2(obs[0], obs[1])
        alpha = np.arctan2(obs[2], obs[3])

        history.append([
            theta, 
            alpha, 
            obs[4], 
            obs[5], 
            float(action[0]),
            reward
        ])
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps.")
            break

    print(f"Total reward: {total_reward:.2f}")

    if len(history) == 0:
        print("No data collected.")
        return

    history = np.array(history)
    cumulative_reward = np.cumsum(history[:, 5])

    # --- Figure 1: Time Series Dynamics ---
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(np.rad2deg(history[:, 0]), label="Theta (Arm)", color='blue')
    plt.plot(np.rad2deg(history[:, 1]), label="Alpha (Pendulum)", color='red')
    plt.axhline(180, linestyle="--", color='black', alpha=0.3)
    plt.axhline(-180, linestyle="--", color='black', alpha=0.3)
    plt.axhline(0, linestyle="-", color='black', alpha=0.1)
    plt.ylabel("Angle (deg)")
    plt.title("QUBE-Servo 2 PPO Performance")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(np.rad2deg(history[:, 2]), label="Theta dot", color='cyan')
    plt.plot(np.rad2deg(history[:, 3]), label="Alpha dot", color='magenta')
    plt.ylabel("Velocity (deg/s)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.step(range(len(history)), history[:, 4], label="Action (Voltage)", color='green')
    plt.ylabel("Voltage (V)")
    plt.ylim(-11, 11)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(history[:, 5], label="Step Reward", color='orange', alpha=0.6)
    plt.plot(cumulative_reward / (np.arange(len(history)) + 1), label="Avg Reward", color='red')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ppo_dynamics.png", dpi=200)
    print("Dynamics plot saved to ppo_dynamics.png")

    # --- Figure 2: Analysis & Phase Space ---
    plt.figure(figsize=(12, 5))

    # Phase Plot: Pendulum
    plt.subplot(1, 2, 1)
    plt.plot(np.rad2deg(history[:, 1]), np.rad2deg(history[:, 3]), color='purple')
    plt.axvline(180, color='red', linestyle='--', alpha=0.5)
    plt.axvline(-180, color='red', linestyle='--', alpha=0.5)
    plt.xlabel("Alpha (deg)")
    plt.ylabel("Alpha Dot (deg/s)")
    plt.title("Pendulum Phase Space")
    plt.grid(True, alpha=0.3)

    # Cumulative Reward
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward, color='brown', linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Learning Progress (Cumulative)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ppo_analysis.png", dpi=200)
    print("Analysis plot saved to ppo_analysis.png")

    env.close()
if __name__ == "__main__":
    test()
