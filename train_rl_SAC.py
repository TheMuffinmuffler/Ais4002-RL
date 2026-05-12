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

    # SB3 Legacy Schedule support
    try:
        import stable_baselines3.common.utils as sb3_utils
        for name in ["FloatSchedule", "ConstantSchedule", "LinearSchedule"]:
            if not hasattr(sb3_utils, name):
                class DummySchedule:
                    def __init__(self, value=0.0, *args, **kwargs):
                        self.value = float(value)
                    def __call__(self, progress_remaining=1.0):
                        return float(getattr(self, "value", 0.0))
                    def __setstate__(self, state):
                        if isinstance(state, dict):
                            self.__dict__.update(state)
                        else:
                            self.value = float(state)
                setattr(sb3_utils, name, DummySchedule)
    except ImportError: pass

    try:
        from gymnasium.spaces.space import Space
        def patched_setstate(self, state):
            if isinstance(state, dict): self.__dict__.update(state)
        Space.__setstate__ = patched_setstate
    except (ImportError, AttributeError): pass

apply_compat_shims()
# --------------------------------

import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from qube_env import QubeEnv
from train_rl import SAC_TOTAL_STEPS, N_ENVS, LEARNING_RATE

# Training steps for fresh start on MacBook
RETRAIN_STEPS = 500000 
N_ENVS_LAPTOP = 4

def get_device():
    # Force CPU for Intel MacBook compatibility
    return "cpu"

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    new_logger = configure("./logs/sac_fresh/", ["stdout", "csv", "tensorboard"])

    device = get_device()
    print(f"Using device: {device}")

    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=N_ENVS_LAPTOP)
    eval_env = Monitor(QubeEnv(domain_randomization=False))

    # Fresh start: removed loading logic to ensure a clean policy
    print("Starting FRESH SAC training...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=300_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        device=device
    )
    
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // N_ENVS_LAPTOP,
        save_path="models/",
        name_prefix="qube_sac_fresh"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("models/", "best_sac_fresh"),
        log_path="logs/sac_eval_fresh/",
        eval_freq=10000 // N_ENVS_LAPTOP,
        deterministic=True,
        render=False
    )

    print(f"Training fresh SAC for {RETRAIN_STEPS} steps...")
    model.learn(
        total_timesteps=RETRAIN_STEPS,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=True
    )

    model.save("models/qube_sac_final")
    env.close()
    eval_env.close()
    print("Retraining finished!")

if __name__ == "__main__":
    main()
