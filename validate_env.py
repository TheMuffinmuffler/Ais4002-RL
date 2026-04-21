import numpy as np
from qube_env import QubeEnv
import matplotlib.pyplot as plt

def swing_up_pd_controller(obs, env):
    theta, alpha, th_dot, al_dot = obs
    # alpha=0 is down, pi is upright
    
    # Parameters for swing-up
    mu = 50.0 
    # Total inertia of pendulum around pivot
    J_p_total = env.J_p + env.m_p * env.l_p**2
    # Energy
    E = 0.5 * J_p_total * al_dot**2 + env.m_p * env.g * env.l_p * (1 - np.cos(alpha))
    E_up = 2 * env.m_p * env.g * env.l_p
    
    # Angle wrapped to [-pi, pi], 0 is down
    alpha_wrapped = ((alpha + np.pi) % (2 * np.pi)) - np.pi
    
    if np.abs(alpha_wrapped) > 2.5: # Close to upright (pi or -pi)
        # PD Control
        # Target alpha is pi or -pi
        target_alpha = np.pi if alpha_wrapped > 0 else -np.pi
        error_alpha = target_alpha - alpha_wrapped
        
        # PD Gains (tuned for Qube)
        kp = 12.0
        kd = 1.5
        kp_th = 2.0
        kd_th = 1.5
        
        u = kp * error_alpha - kd * al_dot - kp_th * theta - kd_th * th_dot
    else:
        # Energy swing-up
        # Acceleration command
        accel = mu * (E - E_up) * np.sign(al_dot * np.cos(alpha))
        # Convert accel to voltage (rough approximation)
        u = accel * 0.1 
        # Add some damping to theta to keep it centered
        u -= 0.5 * theta + 0.1 * th_dot

    return np.array([u], dtype=np.float32)

def main():
    env = QubeEnv()
    obs, _ = env.reset()
    
    history = []
    for _ in range(500):
        action = swing_up_pd_controller(obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        history.append(obs)
        if terminated or truncated:
            break
            
    history = np.array(history)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.rad2deg(history[:, 0]), label='Theta (Arm)')
    plt.plot(np.rad2deg(history[:, 1]), label='Alpha (Pendulum)')
    plt.legend()
    plt.ylabel('Angle (deg)')
    
    plt.subplot(2, 1, 2)
    plt.plot(np.rad2deg(history[:, 2]), label='Theta Dot')
    plt.plot(np.rad2deg(history[:, 3]), label='Alpha Dot')
    plt.legend()
    plt.ylabel('Velocity (deg/s)')
    
    plt.savefig('validation_plot.png')
    print("Validation plot saved to validation_plot.png")

if __name__ == "__main__":
    main()
