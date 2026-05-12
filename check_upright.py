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
