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
        self.L_r_nom = 0.086  
        self.J_r_nom = 0.0001172  
        self.D_r_nom = 0.0004  
        
        self.m_p_nom = 0.053  
        self.L_p_nom = 0.128  
        self.l_p_nom = 0.064 
        self.J_p_nom = 0.0000235  
        self.D_p_nom = 0.000003  
        
        self.g = 9.81
        self.Rm = 8.94    
        self.kt = 0.0431  
        self.km = 0.0431  

        # NEW: Hardware imperfections
        self.stiction_nom = 0.45 # Volts
        self.encoder_res = 2048   # Counts per 360 deg
        
        self._apply_params()

        # Action: Voltage applied to the motor [-10, 10] V
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        # State: [theta, alpha, theta_dot, alpha_dot]
        high = np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.dt = 0.02  # 50 Hz
        self.render_mode = render_mode
        self._step_count = 0
        self._max_episode_steps = 500 
        self.hard_stop_angle = 1.60 

    def _apply_params(self, randomization_scale=0.1):
        if self.domain_randomization:
            self.m_r = self.m_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.L_r = self.L_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.m_p = self.m_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.l_p = self.l_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.D_r = self.D_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.D_p = self.D_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.stiction = self.stiction_nom * self.np_random.uniform(0.8, 1.2)
            
            self.J_r = (1/3) * self.m_r * self.L_r**2
            self.J_p = (1/3) * self.m_p * (self.l_p*2)**2 
        else:
            self.m_r = self.m_r_nom
            self.L_r = self.L_r_nom
            self.m_p = self.m_p_nom
            self.l_p = self.l_p_nom
            self.D_r = self.D_r_nom
            self.D_p = self.D_p_nom
            self.J_r = self.J_r_nom
            self.J_p = self.J_p_nom
            self.stiction = self.stiction_nom

    def _get_obs(self):
        theta, alpha, theta_dot, alpha_dot = self.state
        
        # Quantize observations to match real encoders
        counts_per_rad = self.encoder_res / (2 * np.pi)
        theta_q = np.round(theta * counts_per_rad) / counts_per_rad
        alpha_q = np.round(alpha * counts_per_rad) / counts_per_rad
        
        # Add sensor noise to velocities
        theta_dot_n = theta_dot + self.np_random.normal(0, 0.02)
        alpha_dot_n = alpha_dot + self.np_random.normal(0, 0.02)

        return np.array([
            np.sin(theta_q), np.cos(theta_q),
            np.sin(alpha_q), np.cos(alpha_q),
            theta_dot_n, alpha_dot_n
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_params()
        self._step_count = 0
        self.prev_voltage = 0.0
        if self.np_random.random() < 0.5:
            self.state = np.array([0, np.pi, 0, 0], dtype=np.float32) + self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        else:
            self.state = np.array([0, 0, 0, 0], dtype=np.float32) + self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        requested_voltage = np.clip(action[0], -10.0, 10.0)
        voltage = 0.7 * self.prev_voltage + 0.3 * requested_voltage

        disturbance_torque = 0.0
        if self.np_random.random() < 0.01:
            disturbance_torque = self.np_random.uniform(-0.5, 0.5)
        
        def dynamics(y, u, dist):
            theta, alpha, th_dot, al_dot = y
            eff_u = u
            if abs(u) < self.stiction and abs(th_dot) < 0.1:
                eff_u = 0.0
            elif abs(u) >= self.stiction:
                eff_u = u - np.sign(u) * self.stiction

            tau = (self.kt / self.Rm) * (eff_u - self.km * th_dot) + dist
            
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

        n_substeps = 4
        sub_dt = self.dt / n_substeps
        for _ in range(n_substeps):
            y0 = self.state
            k1 = dynamics(y0, voltage, disturbance_torque)
            k2 = dynamics(y0 + 0.5 * sub_dt * k1, voltage, disturbance_torque)
            k3 = dynamics(y0 + 0.5 * sub_dt * k2, voltage, disturbance_torque)
            k4 = dynamics(y0 + sub_dt * k3, voltage, disturbance_torque)
            self.state = y0 + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            disturbance_torque = 0.0
            if self.state[0] > self.hard_stop_angle:
                self.state[0] = self.hard_stop_angle
                self.state[2] = 0.0
            elif self.state[0] < -self.hard_stop_angle:
                self.state[0] = -self.hard_stop_angle
                self.state[2] = 0.0

        theta, alpha, th_dot, al_dot = self.state
        alpha_err = (np.cos(alpha) + 1)**2 + (np.sin(alpha))**2 
        height_bonus = (1.0 - np.cos(alpha)) * 10.0
        theta_penalty = 5.0 * (theta**2)
        abs_theta = abs(theta)
        if abs_theta > 1.1:
            ramp = np.exp(min(4.0 * (abs_theta - 1.1), 10.0)) - 1.0
            theta_penalty += 30.0 * ramp
        if abs_theta >= self.hard_stop_angle - 0.05:
            theta_penalty += 400.0
        
        reward = height_bonus - (25.0 * alpha_err + theta_penalty + 0.1 * al_dot**2 + 0.1 * th_dot**2 + 0.05 * voltage**2)
        if alpha_err < 0.05 and abs_theta < 0.1:
            reward += 10.0

        self.prev_voltage = voltage
        terminated = bool(abs_theta > (self.hard_stop_angle + 0.1) or abs(th_dot) > 100.0 or abs(al_dot) > 100.0)
        if terminated:
            reward -= 1000.0
        truncated = self._step_count >= self._max_episode_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass
