import gymnasium as gym
from gymnasium import spaces
import numpy as np

class QubeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, domain_randomization=False):
        super(QubeEnv, self).__init__()
        self.domain_randomization = domain_randomization

        # Nominal Quanser Qube Servo 2 Parameters
        self.m_r_nom = 0.095  
        self.L_r_nom = 0.085  
        self.J_r_nom = 0.000057  
        self.D_r_nom = 0.0015  
        
        self.m_p_nom = 0.024  
        self.L_p_nom = 0.129  
        self.l_p_nom = 0.0645 
        self.J_p_nom = 0.000033  
        self.D_p_nom = 0.0005  
        
        self.g = 9.81
        self.Rm = 8.4    
        self.kt = 0.042  
        self.km = 0.042  

        self._apply_params()

        # Action: Voltage applied to the motor [-10, 10] V
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        # State: [theta, alpha, theta_dot, alpha_dot]
        # Observation: [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]
        high = np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.dt = 0.02  # 50 Hz
        self.render_mode = render_mode

    def _apply_params(self, randomization_scale=0.1):
        if self.domain_randomization:
            # Randomize by +/- scale
            self.m_r = self.m_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.L_r = self.L_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.m_p = self.m_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.l_p = self.l_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.D_r = self.D_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.D_p = self.D_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            
            # Recalculate inertias approximately
            self.J_r = (1/3) * self.m_r * self.L_r**2
            self.J_p = (1/3) * self.m_p * (self.l_p*2)**2 # simplified
        else:
            self.m_r = self.m_r_nom
            self.L_r = self.L_r_nom
            self.m_p = self.m_p_nom
            self.l_p = self.l_p_nom
            self.D_r = self.D_r_nom
            self.D_p = self.D_p_nom
            self.J_r = self.J_r_nom
            self.J_p = self.J_p_nom

    def _get_obs(self):
        theta, alpha, theta_dot, alpha_dot = self.state
        return np.array([
            np.sin(theta), np.cos(theta),
            np.sin(alpha), np.cos(alpha),
            theta_dot, alpha_dot
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_params()
        # Start in random positions to help exploration
        # 50% chance to start near upright, 50% near hanging down
        if self.np_random.random() < 0.5:
            # Near upright (alpha = pi)
            self.state = np.array([0, np.pi, 0, 0], dtype=np.float32) + self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        else:
            # Near hanging down (alpha = 0)
            self.state = np.array([0, 0, 0, 0], dtype=np.float32) + self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self._get_obs(), {}

    def step(self, action):
        voltage = np.clip(action[0], -10.0, 10.0)
        
        # Add random disturbance impulses (pushes)
        # 1% chance per step of a sudden torque spike
        disturbance_torque = 0.0
        if self.np_random.random() < 0.01:
            disturbance_torque = self.np_random.uniform(-0.5, 0.5)
        
        # Physics integration using RK4
        def dynamics(y, u, dist):
            theta, alpha, th_dot, al_dot = y
            
            # Torque from voltage + disturbance
            tau = (self.kt / self.Rm) * (u - self.km * th_dot) + dist
            
            # Equations of motion
            m11 = self.J_r + self.m_p * self.L_r**2 + self.m_p * self.l_p**2 * np.sin(alpha)**2
            m12 = self.m_p * self.L_r * self.l_p * np.cos(alpha)
            m21 = m12
            m22 = self.J_p + self.m_p * self.l_p**2
            M = np.array([[m11, m12], [m21, m22]])
            
            c11 = self.D_r + self.m_p * self.l_p**2 * np.sin(2*alpha) * al_dot
            c12 = -self.m_p * self.L_r * self.l_p * np.sin(alpha) * al_dot
            c21 = -0.5 * self.m_p * self.l_p**2 * np.sin(2*alpha) * th_dot
            c22 = self.D_p
            C = np.array([[c11, c12], [c21, c22]])
            
            G = np.array([0, -self.m_p * self.g * self.l_p * np.sin(alpha)])
            
            rhs = np.array([tau, 0]) - C @ np.array([th_dot, al_dot]) - G
            q_ddot = np.linalg.solve(M, rhs)
            
            return np.array([th_dot, al_dot, q_ddot[0], q_ddot[1]])

        # RK4
        y0 = self.state
        k1 = dynamics(y0, voltage, disturbance_torque)
        k2 = dynamics(y0 + 0.5 * self.dt * k1, voltage, disturbance_torque)
        k3 = dynamics(y0 + 0.5 * self.dt * k2, voltage, disturbance_torque)
        k4 = dynamics(y0 + self.dt * k3, voltage, disturbance_torque)
        
        self.state = y0 + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Reward function
        theta, alpha, th_dot, al_dot = self.state
        
        # 1. Pendulum upright goal (alpha near pi)
        # alpha_err is 0 when alpha = pi, and 4 when alpha = 0
        alpha_err = (np.cos(alpha) + 1)**2 + (np.sin(alpha))**2 
        
        # 2. Centering the arm (theta near 0)
        # We increase the weight of theta_penalty to encourage centering.
        # We also make it more aggressive when the pendulum is upright.
        is_upright = np.cos(alpha) < -0.9 # Pendulum is within ~25 degrees of upright
        
        theta_weight = 2.0 if is_upright else 0.5
        theta_penalty = theta_weight * (theta**2)
        
        # Hard penalty for hitting the safety limits
        if np.abs(theta) > 1.4: # Slightly before 90 degrees (1.57)
            theta_penalty += 100.0 

        # Total reward
        # weights: 20 for pendulum upright, higher theta penalty for centering
        reward = -(20.0 * alpha_err + theta_penalty + 0.1 * al_dot**2 + 0.1 * th_dot**2 + 0.001 * voltage**2)

        terminated = False
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            theta, alpha, th_dot, al_dot = self.state
            print(f"Theta: {np.rad2deg(theta):.2f}, Alpha: {np.rad2deg(alpha):.2f}")
