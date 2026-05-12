import sys
import os

# --- INLINE COMPATIBILITY SHIM ---
def apply_compat_shims():
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
    try:
        import stable_baselines3.common.utils as sb3_utils
        for name in ["FloatSchedule", "ConstantSchedule", "LinearSchedule"]:
            if not hasattr(sb3_utils, name):
                class DummySchedule:
                    def __init__(self, value=0.0, *args, **kwargs): self.value = value
                    def __call__(self, *args, **kwargs): return self.value
                setattr(sb3_utils, name, DummySchedule)
    except ImportError: pass
    try:
        import numpy.random._pickle as nprp
        for bg_name in ['MT19937', 'PCG64', 'PCG64DXSM', 'Philox', 'SFC64']:
            if hasattr(nprp, bg_name):
                bg_cls = getattr(nprp, bg_name)
                if hasattr(nprp, 'BitGenerators'):
                    nprp.BitGenerators[bg_cls] = bg_cls
    except Exception: pass
    try:
        from gymnasium.spaces.space import Space
        def patched_setstate(self, state):
            if isinstance(state, dict): self.__dict__.update(state)
        Space.__setstate__ = patched_setstate
    except (ImportError, AttributeError): pass

apply_compat_shims()
# --------------------------------

import gymnasium as gym
import numpy as np
import time
import torch
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
