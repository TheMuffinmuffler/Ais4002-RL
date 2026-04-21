import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from qube_env import QubeEnv

def check():
    model = PPO.load("models/qube_ppo_final.zip")
    env = QubeEnv()
    obs, _ = env.reset()
    
    # Force start from hanging down for the check
    env.state = np.array([0, 0, 0, 0], dtype=np.float32)
    obs = env._get_obs()
    
    upright_steps = 0
    total_steps = 500
    
    for i in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # alpha is at index 2 (sin) and 3 (cos)
        alpha = np.arctan2(obs[2], obs[3])
        # Upright is near pi or -pi
        if np.abs(alpha) > 2.8: # roughly > 160 degrees
            upright_steps += 1
            
    percentage = (upright_steps / total_steps) * 100
    print(f"Pendulum was upright for {upright_steps}/{total_steps} steps ({percentage:.1f}%)")
    
    if percentage > 50:
        print("SUCCESS: The pendulum is staying upright!")
    else:
        print("FAILURE: The pendulum is not staying upright consistently.")

if __name__ == "__main__":
    check()
