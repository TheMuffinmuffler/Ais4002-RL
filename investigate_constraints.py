import gymnasium as gym
import numpy as np
from qube_env import QubeEnv
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

def investigate():
    model_path = "models/qube_sac_final.zip"
    env = QubeEnv(domain_randomization=False)
    model = SAC.load(model_path)

    # Run simulation
    obs, _ = env.reset()
    history = []
    energies = []
    
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Calculate Energy (from qube_env.py)
        theta, alpha, th_dot, al_dot = env.state
        total_energy = 0.5 * env.J_p * al_dot**2 + env.m_p * env.g * env.l_p * (1 + np.cos(alpha))
        energies.append(total_energy)
        
        # Access internal state for plotting
        history.append(np.concatenate([env.state, action]))
        if terminated or truncated:
            break
            
    history = np.array(history)
    energies = np.array(energies)

    # --- Analysis ---
    thetas = history[:, 0]
    max_theta = np.rad2deg(np.max(np.abs(thetas)))
    max_safety_penalty = 0.1 * np.exp(4.0 * np.max(np.abs(thetas)))
    E_up = 2 * env.m_p * env.g * env.l_p
    max_energy_ratio = np.max(energies) / E_up
    
    print(f"Investigation Results:")
    print(f"Max Arm Deviation: {max_theta:.2f} degrees")
    print(f"Max Safety Penalty Encountered: {max_safety_penalty:.2f}")
    print(f"Max Energy Reached: {max_energy_ratio*100:.2f}% of target")
    print(f"Balance Success: {np.abs(np.rad2deg(history[-1, 1])) < 5.0}")

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.rad2deg(history[:, 0]), label="Theta (Arm)")
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel("Arm Angle (deg)")
    plt.title("Is the AI too afraid to move?")
    
    plt.subplot(3, 1, 2)
    plt.plot(np.rad2deg(history[:, 1]), label="Alpha (Pendulum)")
    plt.axhline(0, color='red', linestyle='--', label="Target")
    plt.ylabel("Pendulum Angle (deg)")
    
    plt.subplot(3, 1, 3)
    plt.plot(energies / E_up, label="Energy Ratio", color='green')
    plt.axhline(1.0, color='black', linestyle='--', label="Balance Energy")
    plt.ylabel("E / E_up")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("plots/centering_investigation.png")
    print("Investigation plot saved to plots/centering_investigation.png")

if __name__ == "__main__":
    investigate()
