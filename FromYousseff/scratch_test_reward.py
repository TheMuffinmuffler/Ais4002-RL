import numpy as np
from qube_env import QubeEnv

env = QubeEnv(domain_randomization=False)
env.reset()
# perfectly upright (alpha = pi)
env.unwrapped.state = np.array([0.0, np.pi, 0.0, 0.0])
obs = env.unwrapped._get_obs()

print("Perfect balance state rewards:")
for i in range(25):
    obs, reward, terminated, truncated, info = env.step([0.0])
    comps = info.get("reward_components", {})
    print(f"Step {i:2d}: Total: {reward:7.2f} | r_bal: {comps.get('r_balance'):6.2f} | r_pers: {comps.get('r_persistence'):6.2f} | steps: {env.consecutive_upright_steps:3d}")
