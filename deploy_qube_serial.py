import numpy as np
import time
import sys
import io
import pickle
import csv
import threading
import queue
from datetime import datetime

# --- STABLE BASELINES 3 COMPATIBILITY HACK ---
import stable_baselines3.common.utils as sb3_utils

class DummySchedule:
    def __init__(self, value=0.0, *args, **kwargs): self.value = value
    def __call__(self, *args, **kwargs): return self.value
    @classmethod
    def load(cls, *args, **kwargs): return cls()

for name in ["ConstantSchedule", "FloatSchedule", "LinearSchedule"]:
    if not hasattr(sb3_utils, name):
        setattr(sb3_utils, name, DummySchedule)

class AmbiguousUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

import cloudpickle
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

from stable_baselines3 import PPO
from QUBE import QUBE
from control import COM_PORT

# --- PERFORMANCE TUNING ---
POWER_GAIN = 1.0       # TUNED
MOTOR_INVERT = 1.0     
VELOCITY_FILTER = 0.4  # REACTIVE
ACTION_FILTER = 0.5    # RESPONSIVE
SAFETY_LIMIT = 2.0     # ~115 deg
SAFETY_KILL = 2.27     # ~130 deg
DEADBAND = 0.45        # Matches stiction in env
# --------------------------

# --- BACKGROUND LOGGER ---
class AsyncLogger:
    def __init__(self, filename):
        self.queue = queue.Queue()
        self.filename = filename
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self):
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "dt", "theta", "alpha", "th_dot", "al_dot", "raw_th_dot", "raw_al_dot", "voltage", "raw_action", "safety_v", "current", "mode"])
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
    # 1. Load your trained RL model
    from qube_env import QubeEnv
    env = QubeEnv()
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space
    }

    try:
        model = PPO.load("models/qube_ppo_final.zip", custom_objects=custom_objects)
        print("PPO Model loaded.")
    except Exception as e:
        import traceback
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return

    # 2. Initialize connection to Teensy
    port = COM_PORT
    baudrate = 115200
    try:
        qube = QUBE(port, baudrate)
        print(f"Connected to Qube on {port}")
        print("Waiting for hardware synchronization...")
        time.sleep(2)
        qube.master.reset_input_buffer()
        qube.master.reset_output_buffer()
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
    
    # Pre-initialize readings to prevent velocity spikes
    prev_theta = np.deg2rad(qube.getMotorAngle())
    prev_alpha = np.deg2rad(((qube.getPendulumAngle()) % 360) - 180)
    
    print("--- READY ---")
    print("1. Hold the pendulum HANGING DOWN and STILL.")
    print("2. Starting in 5 seconds...")
    time.sleep(5)
    qube.setRGB(0, 999, 0) # Green

    # --- START-UP KICK ---
    print("Kicking motor to overcome stiction...")
    qube.setMotorVoltage(1.5)
    qube.update()
    time.sleep(0.1)
    qube.setMotorVoltage(0.0)
    qube.update()
    # ---------------------

    log_filename = f"Data/deploy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logger = AsyncLogger(log_filename)

    t_last = time.time()
    t_start = t_last
    
    th_dot_filt = 0
    al_dot_filt = 0
    voltage = 0.0
    prev_voltage = 0.0
    step_count = 0
    
    left_hits = 0
    right_hits = 0
    MAX_HITS = 5
    
    stall_counter = 0
    STALL_LIMIT = 50 # 1.0 second at 50Hz
    
    # HYSTERESIS AND STATE
    in_balance_mode = False
    BAL_THRESHOLD_IN = 150 # Deg
    BAL_THRESHOLD_OUT = 110 # Deg
    
    try:
        while True:
            t_now = time.time()
            qube.update()
            
            # Raw readings
            theta_deg = qube.getMotorAngle()
            alpha_raw = qube.getPendulumAngle() 
            
            # Coordinate Transform: Model expects 0 at TOP
            alpha_deg = ((alpha_raw) % 360) - 180
            
            # Convert to Radians
            theta = np.deg2rad(theta_deg)
            alpha = np.deg2rad(alpha_deg)
            
            dt = t_now - t_last
            t_last = t_now
            if dt <= 0: dt = 0.02
            
            # 4. Calculate Velocities with clipping
            th_dot_raw = np.clip((theta - prev_theta) / dt, -50, 50)
            al_dot_raw = np.clip((alpha - prev_alpha) / dt, -50, 50)
            
            # Exponential Moving Average Filter
            th_dot_filt = VELOCITY_FILTER * th_dot_filt + (1 - VELOCITY_FILTER) * th_dot_raw
            al_dot_filt = VELOCITY_FILTER * al_dot_filt + (1 - VELOCITY_FILTER) * al_dot_raw
            
            # --- HYSTERESIS LOGIC (ADJUSTED FOR 0=UP) ---
            abs_alpha = abs(alpha_deg)
            
            if not in_balance_mode and abs_alpha < 30: # Within 30 deg of upright
                in_balance_mode = True
                th_dot_filt = 0
                al_dot_filt = 0
            elif in_balance_mode and abs_alpha > 50: # Fell outside 50 deg
                in_balance_mode = False

            if in_balance_mode:
                # BALANCE MODE: Focus on high precision
                pred_theta = theta + th_dot_filt * 0.05
                pred_alpha = alpha + al_dot_filt * 0.02
                current_power = 1.2 # Slightly more power to hold balance
                current_deadband = 0.2
            else:
                # SWING-UP MODE: Focus on momentum
                pred_theta = theta + th_dot_filt * 0.02
                pred_alpha = alpha + al_dot_filt * 0.02
                current_power = 1.8 # INCREASED for swing-up
                current_deadband = 1.0 # INCREASED for stiction

            obs = np.array([
                np.sin(pred_theta), np.cos(pred_theta),
                np.sin(pred_alpha), np.cos(pred_alpha),
                th_dot_filt, al_dot_filt
            ], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            requested_voltage = float(action[0]) * current_power * POWER_GAIN * MOTOR_INVERT
            
            # Action Filter (EMA)
            voltage = (1 - ACTION_FILTER) * prev_voltage + ACTION_FILTER * requested_voltage
            
            # Deadband compensation
            if abs(voltage) > 0.05:
                voltage += np.sign(voltage) * current_deadband

            # 7. SAFETY WALL
            abs_theta = np.abs(theta)
            theta_deg_abs = abs(theta_deg)
            
            # Hard Kill for physically impossible angles
            if theta_deg_abs > 180.0:
                print(f"\nCRITICAL SAFETY KILL! Angle out of bounds: {theta_deg:.1f}")
                break

            if abs_theta > SAFETY_LIMIT:
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
                    
                    # Bounce back
                    qube.setMotorVoltage(-np.sign(theta) * 6.0)
                    time.sleep(0.2)
                    qube.update()
                    prev_theta = np.deg2rad(qube.getMotorAngle())
                    th_dot_filt = 0
                    continue
                else:
                    overshoot = abs_theta - SAFETY_LIMIT
                    spring_k = 20.0 
                    voltage = -np.sign(theta) * (overshoot * spring_k + 2.0)

            # HARD LIMIT TO PROTECT MAGNETS
            voltage = np.clip(voltage, -6.0, 6.0)
            # 7b. STALL DETECTION
            if abs(voltage) > 2.0 and abs(th_dot_filt) < 0.2:
                stall_counter += 1
            else:
                stall_counter = 0
            
            if stall_counter > STALL_LIMIT:
                print(f"\nSTALL DETECTED! Theta: {theta_deg:5.1f} | Power Off")
                voltage = 0.0
                qube.setMotorVoltage(0.0)
                break # STOP THE LOOP PERMANENTLY
            
            prev_voltage = voltage

            # --- TELEMETRY ---
            mode_str = "BAL" if in_balance_mode else "SWG"
            if step_count % 10 == 0:
                fps = 1.0 / dt if dt > 0 else 0
                print(f"[{mode_str}] Th:{theta_deg:5.1f} | Al:{alpha_deg:5.1f} | V:{voltage:5.2f} | FPS:{fps:4.1f}", end='\r')
            step_count += 1

            # --- LOGGING ---
            # Columns: time, dt, theta, alpha, th_dot, al_dot, raw_th_dot, raw_al_dot, voltage, raw_action, safety_v, current, mode
            logger.log([
                time.time() - t_start, dt, theta_deg, alpha_deg, 
                th_dot_filt, al_dot_filt, th_dot_raw, al_dot_raw,
                voltage, requested_voltage, safety_v, qube.getMotorCurrent(), mode_str
            ])

            # Update previous values
            prev_theta = theta
            prev_alpha = alpha

            # 8. Send voltage
            qube.setMotorVoltage(voltage)
            
            if step_count % 200 == 0:
                qube.master.reset_input_buffer()

            # Maintain 50Hz
            elapsed = time.time() - t_now
            sleep_time = max(0, 0.02 - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping... Setting voltage to 0.")
    finally:
        logger.stop()
        print(f"\nData saved to {log_filename}")
        qube.setMotorVoltage(0.0)
        qube.setRGB(999, 0, 0) # Red
        if hasattr(qube, 'master'):
            qube.master.close()

if __name__ == "__main__":
    deploy()
