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
    rewards = 0
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Recover theta and alpha from obs [sin_th, cos_th, sin_al, cos_al, ...]
        theta = np.arctan2(obs[0], obs[1])
        alpha = np.arctan2(obs[2], obs[3])
        history.append([theta, alpha, obs[4], obs[5], action[0]])
        rewards += reward
        
        if terminated or truncated:
            break
            
    print(f"Total reward: {rewards}")
    history = np.array(history)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.rad2deg(history[:, 0]), label='Theta (Arm)')
    plt.plot(np.rad2deg(history[:, 1]), label='Alpha (Pendulum)')
    plt.axhline(180, color='r', linestyle='--', label='Upright')
    plt.axhline(-180, color='r', linestyle='--')
    plt.legend()
    plt.ylabel('Angle (deg)')
    
    plt.subplot(3, 1, 2)
    plt.plot(np.rad2deg(history[:, 2]), label='Theta Dot')
    plt.plot(np.rad2deg(history[:, 3]), label='Alpha Dot')
    plt.legend()
    plt.ylabel('Velocity (deg/s)')
    
    plt.subplot(3, 1, 3)
    plt.plot(history[:, 4], label='Action (Voltage)')
    plt.legend()
    plt.ylabel('Voltage (V)')
    
    plt.savefig('rl_test_plot.png')
    print("Test plot saved to rl_test_plot.png")

if __name__ == "__main__":
    test()
