import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    ACTION_FILTER,
    ACTION_LIMIT,
    DAMPING_RANDOMIZATION,
    DOMAIN_RANDOMIZATION_SCALE,
    DT,
    D_P,
    D_R,
    ENCODER_RES,
    EPISODE_STEPS,
    G,
    HARD_STOP_RAD,
    J_P,
    J_R,
    KM,
    KT,
    K_CABLE,
    L_P,
    L_P_COG,
    L_R,
    M_P,
    M_R,
    MOTOR_RANDOMIZATION,
    RM,
    STICTION_RANDOMIZATION,
    STICTION_VOLTAGE,
    TILT_RANDOMIZATION_RAD,
    VELOCITY_FILTER,
)


class QubeEnv(gym.Env):
    """
    SUPER-ENV v6.11: "Centered Sniper" configuration.
    
    Features:
        - Centering Gate (exp(-theta^2)) applied to major rewards.
        - Coupled Balance & Stillness (Gaussian).
        - Mastery Curriculum (70% Classic starts).
        - Beefed up r_kick (-5.0).
    """

    metadata = {"render_modes": ["human"], "render_fps": int(1.0 / DT)}

    def __init__(self, render_mode=None, domain_randomization=False):
        super().__init__()

        self.render_mode = render_mode
        self.domain_randomization = domain_randomization

        # Nominal physical parameters
        self.m_r_nom = M_R
        self.L_r_nom = L_R
        self.J_r_nom = J_R
        self.D_r_nom = D_R

        self.m_p_nom = M_P
        self.L_p_nom = L_P
        self.l_p_nom = L_P_COG
        self.J_p_nom = J_P
        self.D_p_nom = D_P

        self.g = G
        self.Rm = RM
        self.kt_nom = KT
        self.km_nom = KM
        self.k_cable_nom = K_CABLE
        self.stiction_nom = STICTION_VOLTAGE
        self.encoder_res = ENCODER_RES

        self.dt = DT
        self._max_episode_steps = EPISODE_STEPS
        self.hard_stop_angle = HARD_STOP_RAD
        
        # Super-Env: Persistent Disturbance params
        self.tau_dist_max = 0.005 if domain_randomization else 0.0
        self.tau_dist_prob = 0.8
        self._tau_dist = 0.0

        # Super-Env: Action Space [-1.0, 1.0] (to match existing model)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Super-Env: 9-dim Observation Space (includes normalized alpha rotations)
        obs_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 60.0, 80.0, 1.0, 1.0, 100.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        self.state = np.zeros(4, dtype=np.float64)

        self._step_count = 0
        self.vel_filter = VELOCITY_FILTER
        self.act_filter = ACTION_FILTER

        self.th_dot_filt = 0.0
        self.al_dot_filt = 0.0
        self.prev_voltage = 0.0
        self.delayed_action = 0.0
        self.consecutive_upright_steps = 0

        self._apply_params()

    @staticmethod
    def _wrap_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _apply_params(self):
        if self.domain_randomization:
            s = DOMAIN_RANDOMIZATION_SCALE
            ur = self.np_random.uniform

            self.m_r = self.m_r_nom * ur(1.0 - s, 1.0 + s)
            self.L_r = self.L_r_nom * ur(1.0 - s, 1.0 + s)
            self.J_r = self.J_r_nom * ur(1.0 - s, 1.0 + s)

            self.m_p = self.m_p_nom * ur(1.0 - s, 1.0 + s)
            self.l_p = self.l_p_nom * ur(1.0 - s, 1.0 + s)
            self.J_p = self.J_p_nom * ur(1.0 - s, 1.0 + s)

            self.D_r = self.D_r_nom * ur(*DAMPING_RANDOMIZATION)
            self.D_p = self.D_p_nom * ur(*DAMPING_RANDOMIZATION)

            self.kt = self.kt_nom * ur(*MOTOR_RANDOMIZATION)
            self.km = self.km_nom * ur(*MOTOR_RANDOMIZATION)

            self.k_cable = self.k_cable_nom * ur(0.5, 1.5)
            self.stiction = self.stiction_nom * ur(*STICTION_RANDOMIZATION)
            self.tilt = ur(-TILT_RANDOMIZATION_RAD, TILT_RANDOMIZATION_RAD)

        else:
            self.m_r = self.m_r_nom
            self.L_r = self.L_r_nom
            self.J_r = self.J_r_nom
            self.D_r = self.D_r_nom

            self.m_p = self.m_p_nom
            self.l_p = self.l_p_nom
            self.J_p = self.J_p_nom
            self.D_p = self.D_p_nom

            self.kt = self.kt_nom
            self.km = self.km_nom

            self.k_cable = self.k_cable_nom
            self.stiction = self.stiction_nom
            self.tilt = 0.0

    def _get_obs(self):
        theta, alpha, theta_dot, alpha_dot = self.state

        counts_per_rad = self.encoder_res / (2.0 * np.pi)
        theta_q = np.round(theta * counts_per_rad) / counts_per_rad
        alpha_q = np.round(alpha * counts_per_rad) / counts_per_rad

        if self.domain_randomization:
            theta_q += self.np_random.normal(0.0, 0.0015)
            alpha_q += self.np_random.normal(0.0, 0.0015)

        self.th_dot_filt = (
            self.vel_filter * self.th_dot_filt
            + (1.0 - self.vel_filter) * theta_dot
        )
        self.al_dot_filt = (
            self.vel_filter * self.al_dot_filt
            + (1.0 - self.vel_filter) * alpha_dot
        )

        return np.array(
            [
                np.sin(theta_q),
                np.cos(theta_q),
                np.sin(alpha_q),
                np.cos(alpha_q),
                np.clip(self.th_dot_filt, -60.0, 60.0),
                np.clip(self.al_dot_filt, -80.0, 80.0),
                self.prev_voltage / ACTION_LIMIT,
                self._step_count / self._max_episode_steps,
                alpha_q / (2.0 * np.pi), # normalized rotation count
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._apply_params()

        self._step_count = 0
        self.prev_voltage = 0.0
        self.delayed_action = 0.0
        self.th_dot_filt = 0.0
        self.al_dot_filt = 0.0
        self.consecutive_upright_steps = 0
        
        if self.tau_dist_max > 0.0 and self.np_random.random() < self.tau_dist_prob:
            self._tau_dist = float(self.np_random.uniform(-self.tau_dist_max, self.tau_dist_max))
        else:
            self._tau_dist = 0.0

        # v6.4 Mastery Curriculum (70% Classic starts)
        r = self.np_random.random()
        if r < 0.20:
            alpha = np.pi + self.np_random.uniform(-0.15, 0.15)
            alpha_dot = self.np_random.uniform(-0.5, 0.5)
        elif r < 0.90:
            alpha = self.np_random.uniform(-0.25, 0.25)
            alpha_dot = self.np_random.uniform(-0.1, 0.1)
        else:
            alpha = self.np_random.uniform(-np.pi, np.pi)
            alpha_dot = self.np_random.uniform(-1.0, 1.0)

        theta = self.np_random.uniform(-0.15, 0.15)
        theta_dot = self.np_random.uniform(-0.5, 0.5)

        self.state = np.array(
            [theta, self._wrap_pi(alpha), theta_dot, alpha_dot],
            dtype=np.float64,
        )

        return self._get_obs(), {}

    def _dynamics(self, y, u):
        theta, alpha, th_dot, al_dot = y
        u = float(np.clip(u, -ACTION_LIMIT, ACTION_LIMIT))

        if abs(u) < self.stiction and abs(th_dot) < 0.15:
            eff_u = 0.0
        else:
            eff_u = u - np.sign(u) * min(abs(u), self.stiction)

        tau = (self.kt / self.Rm) * (eff_u - self.km * th_dot)
        cable_torque = -self.k_cable * theta

        s = np.sin(alpha)
        c = np.cos(alpha)

        m11 = self.J_r + self.m_p * self.L_r**2 + self.m_p * self.l_p**2 * s**2
        m12 = self.m_p * self.L_r * self.l_p * c
        m22 = self.J_p + self.m_p * self.l_p**2
        M = np.array([[m11, m12], [m12, m22]], dtype=np.float64)

        c11 = self.D_r + self.m_p * self.l_p**2 * np.sin(2.0 * alpha) * al_dot
        c12 = -self.m_p * self.L_r * self.l_p * s * al_dot
        c21 = -0.5 * self.m_p * self.l_p**2 * np.sin(2.0 * alpha) * th_dot
        c22 = self.D_p
        C = np.array([[c11, c12], [c21, c22]], dtype=np.float64)

        G_vec = np.array([0.0, self.m_p * self.g * self.l_p * np.sin(alpha + self.tilt)], dtype=np.float64)
        q_dot = np.array([th_dot, al_dot], dtype=np.float64)
        rhs = np.array([tau + cable_torque + self._tau_dist, 0.0], dtype=np.float64) - C @ q_dot - G_vec

        q_ddot = np.linalg.solve(M, rhs)
        return np.array([th_dot, al_dot, q_ddot[0], q_ddot[1]], dtype=np.float64)

    def compute_reward(self, theta, alpha, th_dot, al_dot, requested_norm, fell_out=False):
        alpha_error = abs(self._wrap_pi(alpha - np.pi))
        
        # 1. v6.9.3 High-Fidelity Stillness Carrot (Soft Catch)
        # Reduced stillness weight (0.8 -> 0.05) to reward being vertical even with motion
        r_balance = 200.0 * np.exp(-(5.0 * alpha_error**2 + 0.05 * al_dot**2))

        # 2. Continuous Parabolic Centering
        # Reduced centering force (10.0 -> 2.0) to focus on catch stability
        p_center = 2.0 * theta**2 + 0.1 * th_dot**2

        # 3. Streamlined Swing Reward
        r_swing = 10.0 * (1.0 - np.cos(alpha))

        # 4. Effort Penalty
        p_effort = 0.05 * requested_norm**2

        # 5. Smoothness Penalty (Dynamic Gear Saver)
        smooth_mult = 5.0 if alpha_error < np.deg2rad(90.0) else 0.1
        p_smooth = smooth_mult * (requested_norm - (self.prev_voltage / ACTION_LIMIT))**2

        # 6. Boundary Wall (Concrete Safety)
        out_excess = max(0.0, abs(theta) - np.deg2rad(120.0))
        p_boundary = 2000.0 * out_excess**2

        # 7. The "Stillness-Filtered" Jackpot (The Catch)
        r_persistence = 0.0
        departure_tax = 0.0
        if alpha_error < np.deg2rad(20.0):
            # Gentler Stillness Filter: 10 rad/s now only reduces reward by ~50%
            stillness_scaling = 1.0 / (1.0 + 0.1 * abs(al_dot))
            r_persistence = min(500.0, 5.0 * self.consecutive_upright_steps) * stillness_scaling
        
        # 8. Departure Tax (Kept at -100 to maintain engagement)
        if fell_out:
            departure_tax = -100.0
        
        reward = r_balance + r_swing + r_persistence + departure_tax - p_center - p_effort - p_smooth - p_boundary
        
        components = {
            "r_balance": r_balance,
            "r_swing": r_swing,
            "r_persistence": r_persistence,
            "departure_tax": departure_tax,
            "p_center": -p_center,
            "p_effort": -p_effort,
            "p_smooth": -p_smooth,
            "p_boundary": -p_boundary,
            "total": reward
        }
        return reward, components

    def step(self, action):
        self._step_count += 1
        raw_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        requested_norm = float(np.clip(raw_action, -1.0, 1.0))
        requested_v = requested_norm * ACTION_LIMIT

        applied_request = self.delayed_action
        self.delayed_action = requested_v

        voltage = (
            (1.0 - self.act_filter) * self.prev_voltage
            + self.act_filter * applied_request
        )

        n_substeps = 4
        h = self.dt / n_substeps

        for _ in range(n_substeps):
            y0 = self.state
            k1 = self._dynamics(y0, voltage)
            k2 = self._dynamics(y0 + 0.5 * h * k1, voltage)
            k3 = self._dynamics(y0 + 0.5 * h * k2, voltage)
            k4 = self._dynamics(y0 + h * k3, voltage)
            self.state = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if self.state[0] > self.hard_stop_angle:
                self.state[0] = self.hard_stop_angle
                self.state[2] = -0.25 * self.state[2]
            elif self.state[0] < -self.hard_stop_angle:
                self.state[0] = -self.hard_stop_angle
                self.state[2] = -0.25 * self.state[2]

        theta, alpha, th_dot, al_dot = self.state
        alpha_error = abs(self._wrap_pi(alpha - np.pi))
        
        # Detect if we were upright but now we're not (Departure Tax detection)
        is_upright = alpha_error < np.deg2rad(20.0)
        fell_out = (self.consecutive_upright_steps > 10) and (not is_upright)

        if is_upright:
            self.consecutive_upright_steps += 1
        else:
            self.consecutive_upright_steps = 0

        reward, components = self.compute_reward(theta, alpha, th_dot, al_dot, requested_norm, fell_out)

        # Termination logic: Goldilocks Runway (400 degrees)
        terminated = bool(abs(th_dot) > 70.0 or abs(al_dot) > 100.0 or abs(alpha) > np.deg2rad(400.0))
        if terminated:
            reward -= 500.0

        self.prev_voltage = voltage
        truncated = self._step_count >= self._max_episode_steps

        return self._get_obs(), float(reward), terminated, truncated, {"reward_components": components}

    def render(self): return None
    def close(self): pass
