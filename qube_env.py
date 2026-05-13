import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import VELOCITY_FILTER, ACTION_FILTER, STICTION_VOLTAGE, HARD_STOP_RAD, ENCODER_RES

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
        self.kt_nom = 0.042    # N.m/A
        self.km = 0.042    # V/(rad/s)

        # Hardware imperfections
        self.stiction_nom = STICTION_VOLTAGE
        self.encoder_res = ENCODER_RES
        
        self._apply_params()

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        # Observation space: sin(th), cos(th), sin(al), cos(al), th_dot, al_dot, prev_voltage
        high = np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.dt = 0.02 
        self.render_mode = render_mode
        self._step_count = 0
        self._max_episode_steps = 500 
        self.hard_stop_angle = HARD_STOP_RAD

        # Hardware Emulation Parameters (from config)
        self.vel_filter = VELOCITY_FILTER
        self.act_filter = ACTION_FILTER
        self.th_dot_filt = 0.0
        self.al_dot_filt = 0.0
        self.prev_voltage = 0.0
        self.delayed_action = 0.0

    def _apply_params(self, randomization_scale=0.1):
        if self.domain_randomization:
            self.m_r = self.m_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.L_r = self.L_r_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.m_p = self.m_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            self.l_p = self.l_p_nom * self.np_random.uniform(1-randomization_scale, 1+randomization_scale)
            # Increased randomization for damping to cover various hardware conditions
            self.D_r = self.D_r_nom * self.np_random.uniform(0.5, 3.0)
            self.D_p = self.D_p_nom * self.np_random.uniform(0.5, 3.0)
            self.stiction = self.stiction_nom * self.np_random.uniform(0.7, 1.3)
            # Motor constant randomization (thermal/strength variation)
            self.kt = self.kt_nom * self.np_random.uniform(0.9, 1.1)
            
            # Randomized Tilt: 0 to 5 degrees (0 to 0.087 rad)
            # This simulates a non-level table surface
            self.tilt = self.np_random.uniform(-0.087, 0.087)
            
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
            self.kt = self.kt_nom
            self.tilt = 0.0

    def _get_obs(self):
        theta, alpha, theta_dot, alpha_dot = self.state
        
        counts_per_rad = self.encoder_res / (2 * np.pi)
        theta_q = np.round(theta * counts_per_rad) / counts_per_rad
        alpha_q = np.round(alpha * counts_per_rad) / counts_per_rad
        
        # --- Real-world Sensor Noise ---
        theta_q += self.np_random.normal(0, 0.001)
        alpha_q += self.np_random.normal(0, 0.001)

        self.th_dot_filt = self.vel_filter * self.th_dot_filt + (1 - self.vel_filter) * theta_dot
        self.al_dot_filt = self.vel_filter * self.al_dot_filt + (1 - self.vel_filter) * alpha_dot

        # alpha=0 is TOP (upright), alpha=pi is BOTTOM (hanging)
        return np.array([
            np.sin(theta_q), np.cos(theta_q),
            np.sin(alpha_q), np.cos(alpha_q),
            self.th_dot_filt, self.al_dot_filt,
            self.prev_voltage / 10.0 # Normalized prev_voltage for state context
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_params()
        self._step_count = 0
        self.prev_voltage = 0.0
        self.delayed_action = 0.0
        self.th_dot_filt = 0.0
        self.al_dot_filt = 0.0
        
        # --- FINAL STAGE: Standard Bottom Reset ---
        # The agent has now learned to swing, catch, and balance in stages.
        alpha_init = np.pi + self.np_random.uniform(-0.5, 0.5)

        self.state = np.array([0, alpha_init, 0, 0], dtype=np.float32) + \
                     self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        
        # --- Control Latency Emulation (1-step delay) ---
        current_requested_voltage = self.delayed_action
        self.delayed_action = np.clip(action[0], -10.0, 10.0)
        
        # Jerk Penalty: Penalize rapid changes in voltage
        jerk_penalty = 0.01 * (self.delayed_action - self.prev_voltage)**2

        voltage = (1 - self.act_filter) * self.prev_voltage + self.act_filter * current_requested_voltage

        def dynamics(y, u):
            theta, alpha, th_dot, al_dot = y
            eff_u = u
            if abs(u) < self.stiction and abs(th_dot) < 0.1:
                eff_u = 0.0
            elif abs(u) >= self.stiction:
                eff_u = u - np.sign(u) * self.stiction

            tau = (self.kt / self.Rm) * (eff_u - self.km * th_dot)
            
            m11 = self.J_r + self.m_p * self.L_r**2 + self.m_p * self.l_p**2 * np.sin(alpha)**2
            m12 = -self.m_p * self.L_r * self.l_p * np.cos(alpha) # Fixed for TOP=0
            m21 = m12
            m22 = self.J_p + self.m_p * self.l_p**2
            M = np.array([[m11, m12], [m21, m22]])
            
            c11 = self.D_r + self.m_p * self.l_p**2 * np.sin(2*alpha) * al_dot
            c12 = self.m_p * self.L_r * self.l_p * np.sin(alpha) * al_dot # Fixed for TOP=0
            c21 = -0.5 * self.m_p * self.l_p**2 * np.sin(2*alpha) * th_dot
            c22 = self.D_p
            C = np.array([[c11, c12], [c21, c22]])
            
            # Restorative force towards BOTTOM (alpha=pi)
            # sin(alpha + pi) = -sin(alpha). 
            G = np.array([0, -self.m_p * self.g * self.l_p * np.sin(alpha + self.tilt)])
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
                self.state[2] = -0.3 * self.state[2] 
            elif self.state[0] < -self.hard_stop_angle:
                self.state[0] = -self.hard_stop_angle
                self.state[2] = -0.3 * self.state[2]

        # --- Random Perturbation (Hitting the pendulum) ---
        if self.domain_randomization and np.cos(self.state[1]) > 0.8:
            if self.np_random.random() < 0.01: 
                hit_velocity = self.np_random.uniform(-5.0, 5.0) 
                self.state[3] += hit_velocity

        theta, alpha, th_dot, al_dot = self.state
        
        # Upright Target: alpha = 0 (cos = 1)
        dist_upright = (np.cos(alpha) - 1)**2 + (np.sin(alpha))**2 
        
        # INCREASED GRADIENT: More reward for just getting higher
        height_reward = 40.0 * (np.cos(alpha) + 1.0)**2 + 60.0 * np.exp(-1.0 * dist_upright)
        
        total_energy = 0.5 * self.J_p * al_dot**2 + self.m_p * self.g * self.l_p * (1 + np.cos(alpha))
        E_target = 2 * self.m_p * self.g * self.l_p 
        energy_gain = 100.0 * min(total_energy, E_target)

        # Hanging Penalty (Stick): Discourage staying at the bottom (cos(alpha) near -1)
        hanging_penalty = 0.0
        if np.cos(alpha) < -0.707: # Within 45 degrees of bottom
            hanging_penalty = 20.0 * ((-np.cos(alpha) - 0.707) / (1.0 - 0.707))

        # Spinning Penalty (Anti-Helicopter): Discourage continuous rotation
        spinning_penalty = 0.0
        if abs(alpha) > 2.0 * np.pi: # More than one full rotation in either direction
            spinning_penalty = 100.0 * (abs(alpha) / (2.0 * np.pi))

        # Removed discontinuous swingup_bonus to prevent "Reward Cliff Blindness"

        safety_penalty = 0.05 * np.exp(2.0 * abs(theta)) 
        theta_penalty = 1.0 * theta**2
        effort_penalty = 0.005 * voltage**2
        velocity_penalty = 0.005 * th_dot**2 + 0.005 * al_dot**2
        
        if np.cos(alpha) > 0.8:
            velocity_penalty += 0.1 * al_dot**2
            effort_penalty += 0.05 * voltage**2 # Strongly penalize jittery voltage near the top

        centering_reward = 2.0 * (1.0 - (theta / self.hard_stop_angle)**2)

        reward = height_reward + centering_reward + energy_gain - (1.0 * dist_upright + safety_penalty + theta_penalty + velocity_penalty + effort_penalty + jerk_penalty + hanging_penalty + spinning_penalty)
        
        if np.cos(alpha) > 0.5: # Top half
            proximity = (np.cos(alpha) - 0.5) / 0.5
            stability = np.exp(-2.0 * abs(theta)) * np.exp(-0.1 * (abs(th_dot) + abs(al_dot)))
            reward += 100.0 * proximity * stability

        if np.cos(alpha) > 0.95: # Balance zone
            reward += 20.0

        self.prev_voltage = voltage
        terminated = bool(abs(th_dot) > 100.0 or abs(al_dot) > 100.0)
        if terminated:
            reward -= 1000.0
        truncated = self._step_count >= self._max_episode_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass
