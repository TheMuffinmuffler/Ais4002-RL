import os
import sys
import numpy as np
from stable_baselines3 import SAC

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qube_env import QubeEnv
from compat import apply_compat_shims

apply_compat_shims()

def verify_final_model():
    model_path = os.path.join("FromYousseff", "models", "qube_sac_refined_FINAL.zip")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Create Evaluation Environment (Deterministic, No Pokes)
    env = QubeEnv(domain_randomization=False)
    
    print(f"--- FINAL VERIFICATION: {model_path} ---")
    model = SAC.load(model_path, env=env)
    
    total_rewards = []
    for ep in range(5):
        obs, _ = env.reset()
        ep_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            
        print(f"  Episode {ep+1}: Reward: {ep_reward:10.2f} | Steps: {steps}")
        total_rewards.append(ep_reward)
        
    avg_reward = np.mean(total_rewards)
    print(f"\nAVERAGE EVALUATION REWARD: {avg_reward:10.2f}")
    
    if avg_reward > 140000:
        print("RESULT: SUCCESS - Model is a Master Balancer.")
    elif avg_reward > 25000:
        print("RESULT: PASS - Model balances, but not perfectly.")
    else:
        print("RESULT: FAIL - Model has degraded.")

if __name__ == "__main__":
    verify_final_model()
