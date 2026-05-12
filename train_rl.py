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
                        # SB3 expects this to return a float
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

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from qube_env import QubeEnv
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# --- MASTER TRAINING CONFIGURATION ---
PPO_TOTAL_STEPS = 200000  # Reduced for quick retraining on MacBook
TD3_TOTAL_STEPS = 100000
SAC_TOTAL_STEPS = 100000
N_ENVS = 4 # Reduced for laptop thermal management
LEARNING_RATE = 2e-4
# -------------------------------------

def get_device():
    # Force CPU for Intel MacBook compatibility and thermal stability
    return "cpu"

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    new_logger = configure("./logs/ppo_retrain/", ["stdout", "csv", "tensorboard"])
    
    device = get_device()
    print(f"Using device: {device}")
    
    env = make_vec_env(lambda: QubeEnv(domain_randomization=True), n_envs=N_ENVS)
    eval_env = Monitor(QubeEnv(domain_randomization=False))
    
    # Check if we should load the existing model to continue
    model_path = "models/qube_ppo_final.zip"
    
    custom_objects = {
        "observation_space": eval_env.observation_space,
        "action_space": eval_env.action_space
    }

    if os.path.exists(model_path):
        print(f"Loading existing model {model_path} for retraining...")
        model = PPO.load(model_path, env=env, device=device, custom_objects=custom_objects)
        model.learning_rate = LEARNING_RATE # Reset LR for fine-tuning
    else:
        print("Starting fresh PPO training...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            ent_coef=0.01,
            device=device
        )
    
    model.set_logger(new_logger)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // N_ENVS,
        save_path="./models/",
        name_prefix="qube_ppo_retrain"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("models/", "best_ppo_retrain"),
        log_path="logs/ppo_eval_retrain/",
        eval_freq=10000 // N_ENVS,
        deterministic=True,
        render=False
    )

    print(f"Retraining PPO for {PPO_TOTAL_STEPS} steps...")
    model.learn(total_timesteps=PPO_TOTAL_STEPS, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)
    
    model.save("models/qube_ppo_final")
    print("Retraining finished!")

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()
