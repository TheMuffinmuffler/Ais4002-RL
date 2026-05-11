import numpy as np
import time
import sys
import os
import torch
import csv
import threading
import queue
import pickle
import cloudpickle
from datetime import datetime

# --- STABLE BASELINES 3 COMPATIBILITY HACK ---
import stable_baselines3.common.utils as sb3_utils
try:
    import numpy.random._pickle as nprp
    for bg_name in ['MT19937', 'PCG64', 'PCG64DXSM', 'Philox', 'SFC64']:
        if hasattr(nprp, bg_name):
            bg_cls = getattr(nprp, bg_name)
            nprp.BitGenerators[bg_cls] = bg_cls
except:
    pass

class DummySchedule:
    def __init__(self, value=0.0, *args, **kwargs): self.value = value
    def __call__(self, *args, **kwargs): return self.value
    @classmethod
    def load(cls, *args, **kwargs): return cls()

for name in ["ConstantSchedule", "FloatSchedule", "LinearSchedule"]:
    if not hasattr(sb3_utils, name):
        setattr(sb3_utils, name, DummySchedule)

original_cloudpickle_loads = cloudpickle.loads

def patched_cloudpickle_loads(data, *args, **kwargs):
    try:
        return original_cloudpickle_loads(data, *args, **kwargs)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            import numpy.core.numeric as _num
            import numpy.core.multiarray as _mul
            import numpy.core.umath as _uma
            sys.modules["numpy._core"] = np.core
            sys.modules["numpy._core.numeric"] = _num
            sys.modules["numpy._core.multiarray"] = _mul
            sys.modules["numpy._core.umath"] = _uma
            try:
                return original_cloudpickle_loads(data, *args, **kwargs)
            finally:
                for mod in ["numpy._core", "numpy._core.numeric", "numpy._core.multiarray", "numpy._core.umath"]:
                    if mod in sys.modules: del sys.modules[mod]
        raise

cloudpickle.loads = patched_cloudpickle_loads
# --- END HACK ---

from stable_baselines3 import SAC
from QUBE import QUBE
from control import COM_PORT

# --- PERFORMANCE TUNING ---
POWER_GAIN = 1.0       
MOTOR_INVERT = 1.0     
VELOCITY_FILTER = 0.8  # Increased from 0.2 for more smoothing (80% old / 20% new)
ACTION_FILTER = 0.5    # Increased from 0.2 for more responsiveness (50% old / 50% new)
SAFETY_LIMIT = 1.4     # ~80 deg
SAFETY_KILL = 1.65     # ~95 deg
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
                    f.flush() # CRITICAL: Allows live analysis while script is running
                except queue.Empty:
                    continue

    def log(self, data):
        self.queue.put(data)

    def stop(self):
        self.stop_event.set()
        self.thread.join()

def deploy():
    # 1. Load trained SAC model
    model_path = "models/qube_sac_final.zip"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from qube_env import QubeEnv
    env = QubeEnv()
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space
    }

    try:
        model = SAC.load(model_path, device=device, custom_objects=custom_objects)
        print(f"SAC Model loaded from {model_path}")
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
    qube.setRGB(0, 0, 999) # Blue for SAC

    log_filename = f"logs/deploy_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger = AsyncLogger(log_filename)

    # Pre-initialize readings to prevent velocity spikes
    qube.update()
    prev_theta = np.deg2rad(qube.getMotorAngle())
    prev_alpha = np.deg2rad(((qube.getPendulumAngle()) % 360) - 180)

    t_start = time.time()
    t_last = t_start
    th_dot_filt = 0
    al_dot_filt = 0
    prev_voltage = 0.0
    
    left_hits = 0
    right_hits = 0
    MAX_HITS = 20

    try:
        while True:
            t_now = time.time()
            qube.update()
            
            # Raw readings
            theta_deg = qube.getMotorAngle()
            alpha_raw = qube.getPendulumAngle() # 0 is BOTTOM
            
            # Coordinate Transform: Model expects 0 at TOP
            # alpha_raw = 0 (Bottom) -> alpha_deg = -180
            # alpha_raw = 180 (Top)  -> alpha_deg = 0
            alpha_deg = ((alpha_raw) % 360) - 180
            
            theta = np.deg2rad(theta_deg)
            alpha = np.deg2rad(alpha_deg)
            
            dt = t_now - t_last
            t_last = t_now
            if dt < 0.005: dt = 0.02
            
            # Velocities with filtering and capping to prevent overflows
            th_dot_raw = np.clip((theta - prev_theta) / dt, -50, 50)
            al_dot_raw = np.clip((alpha - prev_alpha) / dt, -50, 50)
            th_dot_filt = VELOCITY_FILTER * th_dot_filt + (1 - VELOCITY_FILTER) * th_dot_raw
            al_dot_filt = VELOCITY_FILTER * al_dot_filt + (1 - VELOCITY_FILTER) * al_dot_raw
            
            # Latency compensation (Prediction)
            pred_theta = theta + th_dot_filt * 0.02
            pred_alpha = alpha + al_dot_filt * 0.02

            obs = np.array([
                np.sin(pred_theta), np.cos(pred_theta),
                np.sin(pred_alpha), np.cos(pred_alpha),
                th_dot_filt, al_dot_filt
            ], dtype=np.float32)

            # Inference
            action, _ = model.predict(obs, deterministic=True)
            requested_voltage = float(action[0]) * POWER_GAIN * MOTOR_INVERT
            
            # Smoothing
            voltage = (1 - ACTION_FILTER) * prev_voltage + ACTION_FILTER * requested_voltage
            
            # Deadband / Stiction compensation
            if abs(voltage) > 0.1:
                voltage += np.sign(voltage) * DEADBAND

            # Safety
            abs_theta = abs(theta)
            theta_deg_abs = abs(theta_deg)
            
            # 1. Hard Kill for physically impossible angles (beyond 180 deg)
            if theta_deg_abs > 180.0:
                print(f"\nCRITICAL SAFETY KILL! Angle out of bounds: {theta_deg:.1f}")
                break

            # 2. Virtual bumper and Hit counting
            if abs_theta > SAFETY_LIMIT:
                voltage = -np.sign(theta) * (abs_theta - SAFETY_LIMIT) * 15.0 # Virtual spring
                
                if abs_theta > SAFETY_KILL:
                    if theta > 0:
                        right_hits += 1
                        print(f"\nHit RIGHT Stopper! ({right_hits}/{MAX_HITS})")
                    else:
                        left_hits += 1
                        print(f"\nHit LEFT Stopper! ({left_hits}/{MAX_HITS})")
                    
                    if left_hits >= MAX_HITS or right_hits >= MAX_HITS:
                        print(f"Safety Kill! Max hits reached. Theta: {theta_deg:.1f}")
                        break
                    
                    # Bounce back and wait a tiny bit
                    qube.setMotorVoltage(-np.sign(theta) * 4.0)
                    time.sleep(0.2)
                    qube.update() # Update to new position after bounce
                    prev_theta = np.deg2rad(qube.getMotorAngle())
                    th_dot_filt = 0 # Reset velocity estimation
                    t_last = time.time() # Reset timing
                    continue

            voltage = np.clip(voltage, -8.0, 8.0)
            qube.setMotorVoltage(voltage)
            prev_voltage = voltage
            prev_theta = theta
            prev_alpha = alpha

            if int(t_now * 50) % 5 == 0:
                print(f"Th:{theta_deg:5.1f} | Al:{alpha_deg:5.1f} | V:{voltage:5.2f}", end='\r')

            logger.log([t_now - t_start, dt, theta_deg, alpha_deg, th_dot_filt, al_dot_filt, voltage, "SAC"])
            
            # 50Hz Loop
            elapsed = time.time() - t_now
            time.sleep(max(0, 0.02 - elapsed))

    except KeyboardInterrupt:
        print("\nDeployment stopped by user.")
    finally:
        qube.setMotorVoltage(0.0)
        qube.setRGB(999, 0, 0)
        logger.stop()
        print(f"Log saved to {log_filename}")
        if hasattr(qube, 'master'):
            qube.master.close()

if __name__ == "__main__":
    deploy()
