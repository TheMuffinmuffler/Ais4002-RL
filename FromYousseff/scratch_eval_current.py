import os
import numpy as np
from stable_baselines3 import SAC
from qube_env import QubeEnv
from compat import apply_compat_shims
from train_common import MODELS_DIR

apply_compat_shims()

env = QubeEnv(domain_randomization=False)

# Let's find candidate models in the directory
models = [
    os.path.join(MODELS_DIR, "best_sac/best_model.zip"),
    os.path.join(MODELS_DIR, "qube_sac_final.zip")
]

for mpath in models:
    if not os.path.exists(mpath):
        print(f"Candidate {mpath} does not exist.")
        continue
    
    print(f"\nEvaluating: {mpath}")
    model = SAC.load(mpath)
    
    # 3 episodes evaluation
    for ep in range(3):
        obs, _ = env.reset()
        # Force start upright
        env.unwrapped.state = np.array([0.0, np.pi, 0.0, 0.0])
        obs = env.unwrapped._get_obs()
        
        ep_reward = 0.0
        upright_steps = 0
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            # Check if upright
            alpha_error = abs(env.unwrapped._wrap_pi(env.unwrapped.state[1] - np.pi))
            if alpha_error < np.deg2rad(20.0):
                upright_steps += 1
                
            if terminated or truncated:
                break
        
        print(f"  Ep {ep}: Total Reward: {ep_reward:10.2f} | Upright steps: {upright_steps}/500")
