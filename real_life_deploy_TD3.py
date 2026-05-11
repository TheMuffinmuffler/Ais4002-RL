import gymnasium as gym
import numpy as np
import time
import torch
from stable_baselines3 import TD3
import quanser_robots

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def deploy_on_hardware():
    print("--- Quanser Qube Servo 2: TD3 V3 Deployment ---")
    
    device = get_device()
    # 1. Load the trained TD3 V3 model
    try:
        model = TD3.load("models/qube_td3_final.zip", device=device)
        print(f"TD3 model loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize Hardware
    # 'Qube-v0' is the standard ID for the real hardware in quanser_robots
    try:
        env = gym.make('Qube-v0')
    except Exception as e:
        print(f"Error initializing hardware environment: {e}")
        return
    
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
        try:
            env.step(np.array([0.0]))
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    deploy_on_hardware()
