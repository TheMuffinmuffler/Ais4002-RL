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

from stable_baselines3.common.vec_env import SubprocVecEnv
from qube_env import QubeEnv

# Training steps for integration
RETRAIN_STEPS = 1000000 
N_ENVS_LAPTOP = 16

def get_device():
    return "cuda"

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    new_logger = configure("./logs/sac_fresh/", ["stdout", "csv", "tensorboard"])

    device = get_device()
    print(f"Using device: {device}")

    # Use SubprocVecEnv to force true multi-core parallelization
    env = make_vec_env(
        lambda: QubeEnv(domain_randomization=True), 
        n_envs=N_ENVS_LAPTOP,
        vec_env_cls=SubprocVecEnv
    )
    eval_env = Monitor(QubeEnv(domain_randomization=False))

    model_path = "models/qube_sac_final.zip"
    
    # LOAD the previous model for Curriculum Learning
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} for Curriculum Learning...")
        model = SAC.load(model_path, env=env, device=device)
    else:
        print("Starting FRESH SAC training...")
        policy_kwargs = dict(net_arch=[400, 300])
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=20000,
            batch_size=1024,
            tau=0.005,
            gamma=0.99,
            train_freq=16,
            gradient_steps=16,
            ent_coef='auto',
            target_update_interval=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device
        )
    
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // N_ENVS_LAPTOP,
        save_path="models/",
        name_prefix="qube_sac_curriculum"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("models/", "best_sac_curriculum"),
        log_path="logs/sac_eval_curriculum/",
        eval_freq=10000 // N_ENVS_LAPTOP,
        deterministic=True,
        render=False
    )

    print(f"Training SAC for {RETRAIN_STEPS} steps...")
    model.learn(
        total_timesteps=RETRAIN_STEPS,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=True
    )

    model.save("models/qube_sac_final")
    env.close()
    eval_env.close()
    print("Training finished!")

if __name__ == "__main__":
    main()
