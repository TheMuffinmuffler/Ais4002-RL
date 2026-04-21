import numpy as np
import time
from stable_baselines3 import PPO
from QUBE import QUBE
from control import COM_PORT

def deploy():
    # 1. Load your trained RL model
    try:
        model = PPO.load("models/qube_ppo_final.zip")
        print("RL Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize connection to Teensy
    port = COM_PORT
    baudrate = 115200
    try:
        qube = QUBE(port, baudrate)
        print(f"Connected to Qube on {port}")
    except Exception as e:
        print(f"Could not connect to {port}: {e}")
        return

    # Calibration
    print("Initializing Encoders...")
    qube.setRGB(999, 999, 999) # White
    qube.resetMotorEncoder()
    time.sleep(1)
    qube.resetPendulumEncoder()
    qube.update()
    
    print("--- READY ---")
    print("1. Hold the pendulum HANGING DOWN and STILL.")
    print("2. Starting in 5 seconds...")
    time.sleep(5)
    qube.setRGB(0, 999, 0) # Green

    t_last = time.time()
    prev_theta = 0
    prev_alpha = 0

    try:
        while True:
            # 3. Read sensors from Teensy
            qube.update()
            
            # The library returns degrees. 
            # In your setup, 0 is hanging down because we just reset it.
            theta_deg = qube.getMotorAngle()
            alpha_deg = qube.getPendulumAngle()
            
            # Convert to Radians
            theta = np.deg2rad(theta_deg)
            alpha = np.deg2rad(alpha_deg)
            
            # Calculate dt
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            
            if dt <= 0: continue
            
            # 4. Calculate Velocities
            th_dot = (theta - prev_theta) / dt
            al_dot = (alpha - prev_alpha) / dt
            
            prev_theta = theta
            prev_alpha = alpha

            # 5. Pre-process for the Neural Network
            # Our model expects: [sin_th, cos_th, sin_al, cos_al, th_dot, al_dot]
            obs = np.array([
                np.sin(theta), np.cos(theta),
                np.sin(alpha), np.cos(alpha),
                th_dot, al_dot
            ], dtype=np.float32)

            # 6. Get AI Action
            action, _ = model.predict(obs, deterministic=True)
            voltage = action[0]

            # 7. SAFETY WALL: Hard stop at 90 degrees (+/- 1.57 rad)
            if np.abs(theta) > 1.57:
                print(f"Safety Wall Hit! Theta: {theta_deg:.1f}")
                voltage = 0.0 # Stop motor

            # 8. Send voltage back to Teensy
            qube.setMotorVoltage(voltage)

            # 8. Maintain 50Hz (Training dt = 0.02)
            time.sleep(0.015) # Adjusted for processing time

    except KeyboardInterrupt:
        print("\nStopping... Setting voltage to 0.")
    finally:
        qube.setMotorVoltage(0.0)
        qube.setRGB(999, 0, 0) # Red
        if hasattr(qube, 'master'):
            qube.master.close()

if __name__ == "__main__":
    deploy()
