import sys
import os
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from QUBE import QUBE
from control import COM_PORT
from config import VELOCITY_FILTER, ACTION_FILTER, POWER_GAIN, MOTOR_INVERT, THETA_INVERT, PENDULUM_INVERT, SAFETY_LIMIT_RAD, SAFETY_KILL_RAD, STICTION_VOLTAGE, ENCODER_RES
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from logger import AsyncLogger

def deploy():
    # 1. Load SAC model
    from qube_env import QubeEnv
    model_path = "models/qube_sac_final.zip"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dummy_env = QubeEnv()
    custom_objects = {
        "observation_space": dummy_env.observation_space,
        "action_space": dummy_env.action_space
    }
    
    try:
        model = SAC.load(model_path, env=dummy_env, custom_objects=custom_objects, device=device)
        print(f"SAC Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize hardware
    try:
        qube = QUBE(COM_PORT, 115200)
        print(f"Connected to Qube on {COM_PORT}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 3. Auto-Homing Routine
    from homing import auto_home
    user_home = input("\nRun auto-homing routine for the arm? (y/n): ").lower()
    if user_home == 'y':
        auto_home(qube)
    else:
        print("Skipping auto-homing. Please ensure arm is manually centered.")
        qube.resetMotorEncoder()
        time.sleep(1)

    print("Calibrating Pendulum... (Ensure it is HANGING DOWN)")
    qube.setRGB(999, 999, 0) # Yellow
    qube.resetPendulumEncoder() # Resetting at BOTTOM
    qube.update()
    
    print("\n--- READY ---")
    print("1. Pendulum should be HANGING DOWN.")
    print("2. Deployment starts in 3s...")
    time.sleep(3)
    qube.setRGB(0, 0, 999) # Blue for SAC

    log_filename = f"logs/deploy_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger = AsyncLogger(log_filename)

    qube.update()
    t_start = time.time()
    t_last = t_start
    
    # --- VERIFIED HARDWARE INITIALIZATION ---
    # Apply SENSOR inversions to match AI's World View
    theta_deg_init = qube.getMotorAngle() * THETA_INVERT
    alpha_raw_deg_init = qube.getPendulumAngle() * PENDULUM_INVERT
    
    prev_theta = np.deg2rad(theta_deg_init)
    # Hardware reset at BOTTOM=0. Sim BOTTOM is now 0 as well.
    prev_alpha = np.deg2rad(alpha_raw_deg_init)
    
    th_dot_filt = 0
    al_dot_filt = 0
    prev_voltage = 0.0

    hits_left = 0
    hits_right = 0
    was_in_kill_left = False
    was_in_kill_right = False

    try:
        while True:
            t_now = time.time()
            qube.update()
            
            # Apply SENSOR inversions to match AI's World View
            theta_deg = qube.getMotorAngle() * THETA_INVERT
            alpha_raw_deg = qube.getPendulumAngle() * PENDULUM_INVERT
            
            # --- VERIFIED MAPPING ---
            # HW BOTTOM=0 -> Sim BOTTOM=180. HW TOP=-180 -> Sim TOP=0.
            alpha_deg = alpha_raw_deg + 180.0
            if alpha_deg > 180.0:
                alpha_deg -= 360.0
            elif alpha_deg < -180.0:
                alpha_deg += 360.0
            # Result: 0 -> 180, -180 -> 0. Correct!
            
            theta = np.deg2rad(theta_deg)
            alpha = np.deg2rad(alpha_deg)
            
            dt = t_now - t_last
            t_last = t_now
            if dt < 0.01: dt = 0.02
            
            def shortest_angle_diff(a, b):
                diff = a - b
                return (diff + np.pi) % (2 * np.pi) - np.pi

            th_dot_raw = (theta - prev_theta) / dt
            al_dot_raw = shortest_angle_diff(alpha, prev_alpha) / dt
            th_dot_filt = VELOCITY_FILTER * th_dot_filt + (1 - VELOCITY_FILTER) * th_dot_raw
            al_dot_filt = VELOCITY_FILTER * al_dot_filt + (1 - VELOCITY_FILTER) * al_dot_raw
            
            obs = np.array([
                np.sin(theta), np.cos(theta),
                np.sin(alpha), np.cos(alpha),
                th_dot_filt, al_dot_filt,
                prev_voltage / 10.0
            ], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            requested_voltage = float(action[0]) * POWER_GAIN * MOTOR_INVERT
            
            voltage = (1 - ACTION_FILTER) * prev_voltage + ACTION_FILTER * requested_voltage
            
            # Corrected Hit Counter Logic (Left is Positive, Right is Negative)
            if theta > SAFETY_KILL_RAD:
                if not was_in_kill_left:
                    hits_left += 1
                    was_in_kill_left = True
                    print(f"\nHit Left: {hits_left}/20")
            else:
                was_in_kill_left = False

            if theta < -SAFETY_KILL_RAD:
                if not was_in_kill_right:
                    hits_right += 1
                    was_in_kill_right = True
                    print(f"\nHit Right: {hits_right}/20")
            else:
                was_in_kill_right = False

            if hits_left >= 20 or hits_right >= 20:
                print("\nSafety Limit Reached (20 Hits). Stopping.")
                break

            qube.setMotorVoltage(voltage)
            
            prev_theta = theta
            prev_alpha = alpha
            prev_voltage = voltage

            if int(t_now * 50) % 5 == 0:
                print(f"Th:{theta_deg:5.1f} | Al:{alpha_deg:5.1f} | V:{voltage:5.2f}", end='\r')

            logger.log([t_now - t_start, dt, theta_deg, alpha_deg, th_dot_filt, al_dot_filt, voltage, "SAC"])
            
            # Control loop frequency ~50Hz
            elapsed = time.time() - t_now
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

    except KeyboardInterrupt:
        print("\nDeployment stopped.")
    finally:
        qube.setMotorVoltage(0.0)
        qube.setRGB(999, 0, 0)
        logger.stop()
        if hasattr(qube, 'master'):
            qube.master.close()

if __name__ == "__main__":
    deploy()
