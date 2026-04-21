import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
import quanser_robots

def deploy_on_hardware():
    print("--- Quanser Qube Servo 2: RL Deployment ---")
    
    # 1. Load the trained model
    model = PPO.load("models/qube_ppo_final.zip")
    print("Model loaded.")

    # 2. Initialize Hardware
    # 'Qube-v0' is the standard ID for the real hardware in quanser_robots
    env = gym.make('Qube-v0')
    
    print("Safety Check: Pendulum should be STILL and HANGING DOWN.")
    print("Starting in 3 seconds...")
    time.sleep(3)

    obs, _ = env.reset()
    
    try:
        while True:
            # The hardware environment usually returns [theta, alpha, th_dot, al_dot]
            # but we need to check if the quanser_robots gym env already returns sin/cos.
            # If gym.make('Qube-v0') returns raw [th, al, th_d, al_d]:
            if len(obs) == 4:
                theta, alpha, th_dot, al_dot = obs
                processed_obs = np.array([
                    np.sin(theta), np.cos(theta),
                    np.sin(alpha), np.cos(alpha),
                    th_dot, al_dot
                ], dtype=np.float32)
            else:
                processed_obs = obs

            # AI Inference
            action, _ = model.predict(processed_obs, deterministic=True)
            
            # Apply Action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # The real Qube driver usually handles the timing (50Hz), 
            # but we can add a tiny sleep if it runs too fast.
            # time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\nManual Stop Triggered.")
    finally:
        print("Shutting down motor...")
        env.step(np.array([0.0]))
        env.close()

if __name__ == "__main__":
    deploy_on_hardware()
