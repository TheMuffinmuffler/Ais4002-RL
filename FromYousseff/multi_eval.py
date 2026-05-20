import os
import sys
import numpy as np
from stable_baselines3 import SAC

# Add both root and FromYousseff to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "FromYousseff"))

from qube_env import QubeEnv
from compat import apply_compat_shims

apply_compat_shims()

def quick_eval(model_path):
    if not os.path.exists(model_path):
        return "Not found"
    env = QubeEnv(domain_randomization=False)
    model = SAC.load(model_path, env=env)
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated: break
    return total_reward

models = [
    "FromYousseff/models/qube_sac_final.zip",
    "FromYousseff/models/qube_sac_refine_smooth_checkpoint_1050000_steps.zip",
    "FromYousseff/models/qube_sac_refine_smooth_checkpoint_1149984_steps.zip",
    "FromYousseff/models/qube_sac_refine_smooth_checkpoint_1199976_steps.zip",
]

for m in models:
    res = quick_eval(m)
    print(f"{m}: {res}")
