import numpy as np
import torch
from stable_baselines3 import TD3
from qube_env import QubeEnv
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def test():
    # Load the trained TD3 model
    model_path = "models/qube_td3_final.zip"
    device = get_device()

    try:
        model = TD3.load(model_path, device=device)
        print(f"Loaded TD3 model from {model_path} on {device}")
    except Exception as e:
        print(f"Could not load TD3 model: {e}")
        return

    env = QubeEnv()
    obs, _ = env.reset()

    history = []
    total_reward = 0.0

    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        # Observation format:
        # [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]
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

    plt.figure(figsize=(10, 9))

    plt.subplot(4, 1, 1)
    plt.plot(np.rad2deg(history[:, 0]), label="Theta / Arm angle")
    plt.plot(np.rad2deg(history[:, 1]), label="Alpha / Pendulum angle")
    plt.axhline(180, linestyle="--", label="Upright +180 deg")
    plt.axhline(-180, linestyle="--", label="Upright -180 deg")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(np.rad2deg(history[:, 2]), label="Theta dot")
    plt.plot(np.rad2deg(history[:, 3]), label="Alpha dot")
    plt.ylabel("Velocity (deg/s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(history[:, 4], label="TD3 action / voltage")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(history[:, 5], label="Reward per step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("td3_test_plot.png", dpi=200)
    print("Test plot saved to td3_test_plot.png")

    env.close()


if __name__ == "__main__":
    test()