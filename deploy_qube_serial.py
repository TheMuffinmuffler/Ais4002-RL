import sys
import os

# --- INLINE COMPATIBILITY SHIM ---
def apply_compat_shims():
    # 1. NumPy 2.0 compatibility (for loading models saved with NumPy 2.0 in NumPy 1.x)
    try:
        import numpy.core.numeric as numeric
        sys.modules['numpy._core.numeric'] = numeric
        import numpy.core.multiarray as multiarray
        sys.modules['numpy._core.multiarray'] = multiarray
        import numpy.core.umath as umath
        sys.modules['numpy._core.umath'] = umath
        try:
            import numpy._core
        except ImportError:
            import numpy.core as core
            sys.modules['numpy._core'] = core
    except (ImportError, AttributeError): pass

    # 2. Stable Baselines 3 legacy model support
    try:
        import stable_baselines3.common.utils as sb3_utils
        for name in ["FloatSchedule", "ConstantSchedule", "LinearSchedule"]:
            if not hasattr(sb3_utils, name):
                class DummySchedule:
                    def __init__(self, value=0.0, *args, **kwargs): self.value = value
                    def __call__(self, *args, **kwargs): return self.value
                setattr(sb3_utils, name, DummySchedule)
    except ImportError: pass

    # 3. NumPy RNG BitGenerator compatibility
    try:
        import numpy.random._pickle as nprp
        for bg_name in ['MT19937', 'PCG64', 'PCG64DXSM', 'Philox', 'SFC64']:
            if hasattr(nprp, bg_name):
                bg_cls = getattr(nprp, bg_name)
                if hasattr(nprp, 'BitGenerators'):
                    nprp.BitGenerators[bg_cls] = bg_cls
    except Exception: pass

    # 4. Gymnasium Space compatibility (for legacy picklings)
    try:
        from gymnasium.spaces.space import Space
        def patched_setstate(self, state):
            if isinstance(state, dict): self.__dict__.update(state)
        Space.__setstate__ = patched_setstate
    except (ImportError, AttributeError): pass

apply_compat_shims()
# --------------------------------

import numpy as np
import time
import torch
import csv
import threading
import queue
from datetime import datetime
from stable_baselines3 import PPO
from QUBE import QUBE
from control import COM_PORT
from config import VELOCITY_FILTER, ACTION_FILTER, POWER_GAIN, MOTOR_INVERT, SAFETY_LIMIT_RAD, SAFETY_KILL_RAD, STICTION_VOLTAGE

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
    # 1. Load PPO model
    model_path = "models/qube_ppo_final.zip"
    try:
        model = PPO.load(model_path, device="cpu")
        print(f"PPO Model loaded from {model_path}")
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
    qube.setRGB(0, 999, 0) # Green for PPO

    log_filename = f"logs/deploy_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger = AsyncLogger(log_filename)

    qube.update()
    t_start = time.time()
    t_last = t_start
    prev_theta = np.deg2rad(qube.getMotorAngle())
    prev_alpha = np.deg2rad(qube.getPendulumAngle())
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
            
            theta_deg = qube.getMotorAngle()
            alpha_raw = qube.getPendulumAngle()
            
            # Coordinate Transform: Model expects 0 at TOP
            alpha_deg = ((alpha_raw) % 360) - 180
            
            theta = np.deg2rad(theta_deg)
            alpha = np.deg2rad(alpha_deg)
            
            dt = t_now - t_last
            t_last = t_now
            if dt <= 0: dt = 0.02
            
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
                th_dot_filt, al_dot_filt
            ], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            requested_voltage = float(action[0]) * POWER_GAIN * MOTOR_INVERT
            
            voltage = (1 - ACTION_FILTER) * prev_voltage + ACTION_FILTER * requested_voltage
            
            # Hit Counter Logic
            if theta < -SAFETY_KILL_RAD:
                if not was_in_kill_left:
                    hits_left += 1
                    was_in_kill_left = True
                    print(f"\nHit Left: {hits_left}/20")
            else:
                was_in_kill_left = False

            if theta > SAFETY_KILL_RAD:
                if not was_in_kill_right:
                    hits_right += 1
                    was_in_kill_right = True
                    print(f"\nHit Right: {hits_right}/20")
            else:
                was_in_kill_right = False

            if hits_left >= 20 or hits_right >= 20:
                print(f"Safety Kill! 20 hits reached on one side.")
                break

            voltage = np.clip(voltage, -10.0, 10.0)
            qube.setMotorVoltage(voltage)
            prev_voltage = voltage
            prev_theta = theta
            prev_alpha = alpha

            if int(t_now * 50) % 5 == 0:
                print(f"Th:{theta_deg:5.1f} | Al:{alpha_deg:5.1f} | V:{voltage:5.2f}", end='\r')

            logger.log([t_now - t_start, dt, theta_deg, alpha_deg, th_dot_filt, al_dot_filt, voltage, "PPO"])
            
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
