import gymnasium as gym
from gymnasium import spaces
import numpy as np

class QubeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, domain_randomization=False):
        super(QubeEnv, self).__init__()
        self.domain_randomization = domain_randomization

        # Official Quanser QUBE-Servo 2 Parameters
        self.m_r_nom = 0.095   # kg
        self.L_r_nom = 0.085   # m
        self.J_r_nom = 5.72e-5 # kg.m^2 (CoM)
        self.D_r_nom = 0.0015  # N.m.s/rad
        
        self.m_p_nom = 0.024   # kg
        self.L_p_nom = 0.129   # m
        self.l_p_nom = 0.0645  # m (CoM distance)
        self.J_p_nom = 3.33e-5 # kg.m^2 (CoM)
        self.D_p_nom = 0.0005  # N.m.s/rad
        
        self.g = 9.81
        self.Rm = 8.4      # Ohms
        self.kt = 0.042    # N.m/A
        self.km = 0.042    # V/(rad/s)

        # Hardware imperfections
        self.stiction_nom = 0.45 # Volts
        self.encoder_res = 2048   # Counts per 360 deg
        
        self._apply_params()

        # Action: Voltage applied to the motor [-10, 10] V
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        # State: [sin_th, cos_th, sin_al, cos_al, th_dot, al_dot]
        # Note: alpha=0 is UPRIGHT
        high = np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None # [theta, alpha, theta_dot, alpha_dot] where alpha=0 is DOWN internally
        self.dt = 0.02  # 50 Hz
        self.render_mode = render_mode
        self._step_count = 0
        self._max_episode_steps = 500 
        self.hard_stop_angle = 2.37 # 136 degrees (measured)

        # Hardware Emulation Parameters
        self.vel_filter = 0.8  # Matches deploy scripts
        self.act_filter = 0.5  # Matches deploy scripts
        self.th_dot_filt = 0.0
        self.al_dot_filt = 0.0

    def _apply_params(self, randomization_scale=0.1):
        if self.domain_randomization:
            self.m_r = self.m_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.L_r = self.L_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.m_p = self.m_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.l_p = self.l_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.D_r = self.D_r_nom * self.np_random.uniform(0.5, 2.0)
            self.D_p = self.D_p_nom * self.np_random.uniform(0.5, 2.0)
            self.stiction = self.stiction_nom * self.np_random.uniform(0.8, 1.2)
            
            self.J_r = self.J_r_nom * (self.m_r / self.m_r_nom) * (self.L_r / self.L_r_nom)**2
            self.J_p = self.J_p_nom * (self.m_p / self.m_p_nom) * ((self.l_p*2) / self.L_p_nom)**2
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
        user_alpha = alpha - np.pi
        
        counts_per_rad = self.encoder_res / (2 * np.pi)
        theta_q = np.round(theta * counts_per_rad) / counts_per_rad
        alpha_q = np.round(user_alpha * counts_per_rad) / counts_per_rad
        
        # Latency/Filter emulation (EMA)
        self.th_dot_filt = self.vel_filter * self.th_dot_filt + (1 - self.vel_filter) * theta_dot
        self.al_dot_filt = self.vel_filter * self.al_dot_filt + (1 - self.vel_filter) * alpha_dot

        return np.array([
            np.sin(theta_q), np.cos(theta_q),
            np.sin(alpha_q), np.cos(alpha_q),
            self.th_dot_filt, self.al_dot_filt
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_params()
        self._step_count = 0
        self.prev_voltage = 0.0
        self.th_dot_filt = 0.0
        self.al_dot_filt = 0.0
        self.state = np.array([0, 0, 0, 0], dtype=np.float32) + self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        requested_voltage = np.clip(action[0], -10.0, 10.0)
        
        # Filter emulation for motor ramp-up (EMA)
        voltage = (1 - self.act_filter) * self.prev_voltage + self.act_filter * requested_voltage

        def dynamics(y, u):
            theta, alpha, th_dot, al_dot = y
            eff_u = u
            if abs(u) < self.stiction and abs(th_dot) < 0.1:
                eff_u = 0.0
            elif abs(u) >= self.stiction:
                eff_u = u - np.sign(u) * self.stiction

            tau = (self.kt / self.Rm) * (eff_u - self.km * th_dot)
            
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
            k1 = dynamics(y0, voltage)
            k2 = dynamics(y0 + 0.5 * sub_dt * k1, voltage)
            k3 = dynamics(y0 + 0.5 * sub_dt * k2, voltage)
            k4 = dynamics(y0 + sub_dt * k3, voltage)
            self.state = y0 + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            if self.state[0] > self.hard_stop_angle:
                self.state[0] = self.hard_stop_angle
                self.state[2] = 0.0
            elif self.state[0] < -self.hard_stop_angle:
                self.state[0] = -self.hard_stop_angle
                self.state[2] = 0.0

        theta, alpha, th_dot, al_dot = self.state
        
        # 1. Upright Bonus (Height + Precision)
        # alpha=0 is DOWN internally, so cos(alpha) = -1 at bottom, 1 at top.
        dist_upright = (np.cos(alpha) + 1)**2 + (np.sin(alpha))**2 # ~0 when upright
        height_reward = (1.0 - np.cos(alpha)) * 10.0 # ~20 at top, 0 at bottom
        
        # 2. Energy-Guided Swing-up
        # We reward building energy to overcome gravity
        total_energy = 0.5 * self.J_p * al_dot**2 + self.m_p * self.g * self.l_p * (1 - np.cos(alpha))
        E_target = 2 * self.m_p * self.g * self.l_p # Energy needed to reach the top
        energy_error = abs(total_energy - E_target)
        
        # 3. Smooth Exponential Safety Spring
        # Penalty grows exponentially as it approaches hard_stop_angle (2.37 rad / 136 deg)
        # Begins to be felt significantly at ~1.5 rad (85 deg)
        safety_penalty = 0.05 * np.exp(3.5 * abs(theta)) 
        
        # 4. Precision & Effort
        theta_penalty = 2.0 * theta**2
        effort_penalty = 0.01 * voltage**2
        velocity_penalty = 0.05 * (th_dot**2 + al_dot**2)

        # Composite Reward
        reward = height_reward - (15.0 * dist_upright + 5.0 * energy_error + safety_penalty + theta_penalty + velocity_penalty + effort_penalty)
        
        if dist_upright < 0.1 and abs(theta) < 0.1:
            reward += 15.0 # Stay-Up Bonus

        self.prev_voltage = voltage
        terminated = bool(abs(th_dot) > 100.0 or abs(al_dot) > 100.0)
        if terminated:
            reward -= 1000.0
        truncated = self._step_count >= self._max_episode_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass
