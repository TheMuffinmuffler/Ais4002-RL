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
        self._step_count = 0
        self._max_episode_steps = 500 # 10 seconds at 50Hz

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
        self._step_count = 0
        self.prev_voltage = 0.0
        # Curriculum Learning (Option 3): 
        # We start near upright 80% of the time to master balancing first.
        # Only 20% of the time do we start from the bottom to practice swing-up.
        if self.np_random.random() < 0.8:
            # Near upright (alpha = pi)
            self.state = np.array([0, np.pi, 0, 0], dtype=np.float32) + self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        else:
            # Near hanging down (alpha = 0)
            self.state = np.array([0, 0, 0, 0], dtype=np.float32) + self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        requested_voltage = np.clip(action[0], -10.0, 10.0)
        
        # Action Filtering (Option 3: Sim-to-Real Gap)
        # We blend the previous voltage with the new request to simulate motor lag/inertia.
        # This makes the training much more robust for real hardware.
        voltage = 0.7 * self.prev_voltage + 0.3 * requested_voltage

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

        # RK4 with Substepping (Option: Numerical Stability)
        # We divide the 0.02s step into 4 smaller 0.005s steps for better physics accuracy
        n_substeps = 4
        sub_dt = self.dt / n_substeps
        
        for _ in range(n_substeps):
            y0 = self.state
            k1 = dynamics(y0, voltage, disturbance_torque)
            k2 = dynamics(y0 + 0.5 * sub_dt * k1, voltage, disturbance_torque)
            k3 = dynamics(y0 + 0.5 * sub_dt * k2, voltage, disturbance_torque)
            k4 = dynamics(y0 + sub_dt * k3, voltage, disturbance_torque)
            self.state = y0 + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Reset disturbance torque after first substep so it's an impulse
            disturbance_torque = 0.0

        # Reward function
        theta, alpha, th_dot, al_dot = self.state
        
        # 1. Pendulum upright goal (alpha near pi)
        # alpha_err is 0 when alpha = pi, and 4 when alpha = 0
        alpha_err = (np.cos(alpha) + 1)**2 + (np.sin(alpha))**2 
        
        # 2. Centering the arm (theta near 0) with V3 Barrier Penalty
        # We ramp up the penalty significantly as it approaches the safety limit (1.4 rad)
        theta_penalty = 5.0 * (theta**2)
        abs_theta = abs(theta)
        if abs_theta > 1.0:
            # Exponential ramp starts at 1.0 rad, capped to prevent overflow
            ramp = np.exp(min(3.0 * (abs_theta - 1.0), 10.0)) - 1.0
            theta_penalty += 20.0 * ramp
        
        if abs_theta > 1.4:
            # The "Brick Wall" penalty for hitting safety limits
            theta_penalty += 100.0
        
        # 3. Action smoothness and energy penalties
        voltage_diff = voltage - self.prev_voltage
        smoothness_penalty = 0.5 * (voltage_diff**2)
        energy_penalty = 0.05 * (voltage**2)
        
        # Total reward
        reward = -(20.0 * alpha_err + theta_penalty + 0.1 * al_dot**2 + 0.1 * th_dot**2 + energy_penalty + smoothness_penalty)

        # Bonus for balancing perfectly
        is_balanced = alpha_err < 0.05 and abs_theta < 0.1
        if is_balanced:
            reward += 10.0

        self.prev_voltage = voltage

        # Early Termination (Safety/Numerical Stability)
        # If the system "explodes" or moves too fast, stop the episode.
        # Max theta: 2.0 rad (~115 deg), Max velocities: 100 rad/s
        terminated = bool(
            abs_theta > 2.0 or 
            abs(th_dot) > 100.0 or 
            abs(al_dot) > 100.0
        )
        
        if terminated:
            reward -= 1000.0 # Heavy penalty for failing safety limits completely

        truncated = self._step_count >= self._max_episode_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            theta, alpha, th_dot, al_dot = self.state
            print(f"Theta: {np.rad2deg(theta):.2f}, Alpha: {np.rad2deg(alpha):.2f}")
