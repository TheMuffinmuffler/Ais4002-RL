import numpy as np
import torch
from stable_baselines3 import PPO
from qube_env import QubeEnv
import matplotlib.pyplot as plt
import os

def test_stability():
    model_path = "models/qube_ppo_final.zip"
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    model = PPO.load(model_path)
    env = QubeEnv()
    obs, _ = env.reset()

    velocities = []
    angles = []
    
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # obs[4] is theta_dot, obs[5] is alpha_dot
        velocities.append([obs[4], obs[5]])
        # obs[2], obs[3] are sin(al), cos(al)
        alpha = np.arctan2(obs[2], obs[3])
        angles.append(alpha)
        
        if terminated or truncated:
            break

    velocities = np.array(velocities)
    angles = np.array(angles)
    
    # Calculate Mean Absolute Velocity in the second half of the episode (the "balancing" phase)
    steady_state_v = np.mean(np.abs(velocities[250:, :]), axis=0)
    
    print(f"--- Stability Analysis ---")
    print(f"Mean Alpha (Pendulum) Speed: {steady_state_v[1]:.4f} rad/s")
    print(f"Mean Theta (Arm) Speed:      {steady_state_v[0]:.4f} rad/s")
    
    if steady_state_v[1] < 0.5:
        print("RESULT: The pendulum is STATIONARY (Balancing).")
    else:
        print("RESULT: The pendulum is MOVING (Swinging).")

    plt.figure(figsize=(10, 4))
    plt.plot(np.rad2deg(velocities[:, 1]), label="Alpha Dot (Pendulum Velocity)")
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel("Deg/s")
    plt.title("Pendulum Velocity during Episode")
    plt.legend()
    plt.savefig("plots/velocity_check.png")
    print("Plot saved to plots/velocity_check.png")

if __name__ == "__main__":
    test_stability()
