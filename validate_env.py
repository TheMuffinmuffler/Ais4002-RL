import numpy as np
from qube_env import QubeEnv
import matplotlib.pyplot as plt

def swing_up_pd_controller(obs, env):
    # obs: [sin(th), cos(th), sin(al), cos(al), th_dot, al_dot, prev_v]
    sin_th, cos_th, sin_al, cos_al, th_dot, al_dot, prev_v = obs
    theta = np.arctan2(sin_th, cos_th)
    alpha = np.arctan2(sin_al, cos_al)
    
    # alpha=0 is upright, alpha=pi (or -pi) is hanging
    
    # Parameters for swing-up
    mu = 15.0 # Energy gain
    J_p_total = env.J_p + env.m_p * env.l_p**2
    
    # Energy: E = 0.5 * J * alpha_dot^2 + mgh(1 + cos(alpha))
    # At alpha=0 (upright), E = 2*mgl
    # At alpha=pi (hanging), E = 0
    E = 0.5 * J_p_total * al_dot**2 + env.m_p * env.g * env.l_p * (1 + np.cos(alpha))
    E_up = 2 * env.m_p * env.g * env.l_p
    
    if np.abs(alpha) < 0.5: # Close to upright
        # PD Control to balance at 0
        kp = 15.0
        kd = 1.0
        kp_th = 2.0
        kd_th = 1.0
        
        u = -kp * alpha - kd * al_dot - kp_th * theta - kd_th * th_dot
    else:
        # Energy swing-up
        # We want to increase energy towards E_up
        # accel = mu * (E - E_up) * sign(alpha_dot * cos(alpha))
        # Note: cos(alpha) is positive near upright, negative near hanging.
        # But the standard energy swingup is u = mu * (E - E_up) * alpha_dot * cos(alpha)
        # or similar.
        u = mu * (E - E_up) * al_dot * np.cos(alpha)
        
        # Add some damping to theta to keep it centered
        u -= 1.0 * theta + 0.5 * th_dot

    return np.array([u], dtype=np.float32)

def main():
    env = QubeEnv()
    obs, _ = env.reset()
    
    history = []
    rewards = []
    for _ in range(500):
        action = swing_up_pd_controller(obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        history.append(obs)
        rewards.append(reward)
        if terminated or truncated:
            break
            
    history = np.array(history)
    
    # Reconstruct angles for plotting
    thetas = np.arctan2(history[:, 0], history[:, 1])
    alphas = np.arctan2(history[:, 2], history[:, 3])
    theta_dots = history[:, 4]
    alpha_dots = history[:, 5]
    
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(np.rad2deg(thetas), label='Theta (Arm)')
    plt.plot(np.rad2deg(alphas), label='Alpha (Pendulum)')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.ylabel('Angle (deg)')
    plt.title('Validation PD Controller')
    
    plt.subplot(3, 1, 2)
    plt.plot(np.rad2deg(theta_dots), label='Theta Dot')
    plt.plot(np.rad2deg(alpha_dots), label='Alpha Dot')
    plt.legend()
    plt.ylabel('Velocity (deg/s)')
    
    plt.subplot(3, 1, 3)
    plt.plot(rewards, label='Reward')
    plt.legend()
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('validation_plot.png')
    print(f"Validation plot saved to validation_plot.png. Total Reward: {sum(rewards):.2f}")
    print(f"Min |Alpha|: {np.rad2deg(np.min(np.abs(alphas))):.2f} degrees")
    print(f"Max |Theta|: {np.rad2deg(np.max(np.abs(thetas))):.2f} degrees")

if __name__ == "__main__":
    main()
