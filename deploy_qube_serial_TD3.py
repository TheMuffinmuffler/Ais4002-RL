import numpy as np
import time
import sys
import os
import torch
import csv
import threading
import queue
from datetime import datetime
from stable_baselines3 import TD3
from QUBE import QUBE
from control import COM_PORT

# --- PERFORMANCE TUNING ---
POWER_GAIN = 0.5       
MOTOR_INVERT = -1.0     
VELOCITY_FILTER = 0.4  
ACTION_FILTER = 0.5    
SAFETY_LIMIT = 2.1     
SAFETY_KILL = 2.3      
DEADBAND = 0.45        
# --------------------------

class AsyncLogger:
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.queue = queue.Queue()
        self.filename = filename
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self):
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "dt", "theta", "alpha", "th_dot", "al_dot", "voltage", "mode"])
            while not (self.stop_event.is_set() and self.queue.empty()):
                try:
                    data = self.queue.get(timeout=0.1)
                    writer.writerow(data)
                except queue.Empty:
                    continue

    def log(self, data):
        self.queue.put(data)

    def stop(self):
        self.stop_event.set()
        self.thread.join()

def deploy():
    # 1. Load trained TD3 model
    model_path = "models/qube_td3_final.zip"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = TD3.load(model_path, device=device)
        print(f"TD3 Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize hardware
    try:
        qube = QUBE(COM_PORT, 115200)
        print(f"Connected to Qube on {COM_PORT}")
        time.sleep(2)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("Calibrating...")
    qube.setRGB(999, 999, 999) # White
    qube.resetMotorEncoder()
    time.sleep(1)
    qube.resetPendulumEncoder() # Resetting at BOTTOM
    qube.update()
    
    print("\n--- READY ---")
    print("1. Ensure pendulum is HANGING DOWN.")
    print("2. Deployment starts in 5s...")
    time.sleep(5)
    qube.setRGB(0, 999, 999) # Cyan for TD3

    log_filename = f"logs/deploy_td3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger = AsyncLogger(log_filename)

    t_start = time.time()
    t_last = t_start
    prev_theta = 0
    prev_alpha = 0
    th_dot_filt = 0
    al_dot_filt = 0
    prev_voltage = 0.0

    try:
        while True:
            t_now = time.time()
            qube.update()
            
            theta_deg = qube.getMotorAngle()
            alpha_raw = qube.getPendulumAngle()
            
            # Coordinate Transform: Model expects 0 at TOP
            alpha_deg = ((alpha_raw) % 360) - 180
            
            theta = np.deg2rad(theta_deg)
            alpha = np.deg2rad(alpha_deg)
            
            dt = t_now - t_last
            t_last = t_now
            if dt <= 0: dt = 0.02
            
            th_dot_raw = (theta - prev_theta) / dt
            al_dot_raw = (alpha - prev_alpha) / dt
            th_dot_filt = VELOCITY_FILTER * th_dot_filt + (1 - VELOCITY_FILTER) * th_dot_raw
            al_dot_filt = VELOCITY_FILTER * al_dot_filt + (1 - VELOCITY_FILTER) * al_dot_raw
            
            pred_theta = theta + th_dot_filt * 0.02
            pred_alpha = alpha + al_dot_filt * 0.02

            obs = np.array([
                np.sin(pred_theta), np.cos(pred_theta),
                np.sin(pred_alpha), np.cos(pred_alpha),
                th_dot_filt, al_dot_filt
            ], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            requested_voltage = float(action[0]) * POWER_GAIN * MOTOR_INVERT
            
            voltage = (1 - ACTION_FILTER) * prev_voltage + ACTION_FILTER * requested_voltage
            
            if abs(voltage) > 0.1:
                voltage += np.sign(voltage) * DEADBAND

            abs_theta = abs(theta)
            if abs_theta > SAFETY_LIMIT:
                voltage = np.sign(theta) * (abs_theta - SAFETY_LIMIT) * 15.0
                if abs_theta > SAFETY_KILL:
                    print(f"Safety Kill! Theta: {theta_deg:.1f}")
                    break

            voltage = np.clip(voltage, -8.0, 8.0)
            qube.setMotorVoltage(voltage)
            prev_voltage = voltage
            prev_theta = theta
            prev_alpha = alpha

            if int(t_now * 50) % 5 == 0:
                print(f"Th:{theta_deg:5.1f} | Al:{alpha_deg:5.1f} | V:{voltage:5.2f}", end='\r')

            logger.log([t_now - t_start, dt, theta_deg, alpha_deg, th_dot_filt, al_dot_filt, voltage, "TD3"])
            
            elapsed = time.time() - t_now
            time.sleep(max(0, 0.02 - elapsed))

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
