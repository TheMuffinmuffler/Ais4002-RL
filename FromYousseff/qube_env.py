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
    QUBE-Servo 2 rotary inverted pendulum environment.

    Angle convention:
        theta = motor/arm angle, 0 rad at centre.
        alpha = pendulum angle, 0 rad hanging down, +/-pi rad upright.

    Observation:
        [
            sin(theta),
            cos(theta),
            sin(alpha),
            cos(alpha),
            theta_dot,
            alpha_dot,
            previous_voltage / ACTION_LIMIT
        ]
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

        self.action_space = spaces.Box(
            low=-ACTION_LIMIT,
            high=ACTION_LIMIT,
            shape=(1,),
            dtype=np.float32,
        )

        obs_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 60.0, 80.0, 1.0],
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

        self._apply_params()

    @staticmethod
    def _wrap_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _apply_params(self):
        """
        Apply nominal parameters or randomized parameters.

        Domain randomization is kept moderate. Too much randomization makes
        short training worse, not better.
        """
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

        # Encoder quantization
        counts_per_rad = self.encoder_res / (2.0 * np.pi)
        theta_q = np.round(theta * counts_per_rad) / counts_per_rad
        alpha_q = np.round(alpha * counts_per_rad) / counts_per_rad

        # Small sensor noise only during randomized training
        if self.domain_randomization:
            theta_q += self.np_random.normal(0.0, 0.0015)
            alpha_q += self.np_random.normal(0.0, 0.0015)

        # Filter velocities to make simulation closer to hardware observation
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

        # Curriculum-style reset:
        # - some episodes start near upright to teach stabilization
        # - most start near bottom to teach swing-up
        # - some are random to improve robustness
        r = self.np_random.random()

        if r < 0.25:
            alpha = np.pi + self.np_random.uniform(-0.25, 0.25)
            alpha_dot = self.np_random.uniform(-1.0, 1.0)

        elif r < 0.80:
            alpha = self.np_random.uniform(-0.35, 0.35)
            alpha_dot = self.np_random.uniform(-0.5, 0.5)

        else:
            alpha = self.np_random.uniform(-np.pi, np.pi)
            alpha_dot = self.np_random.uniform(-2.0, 2.0)

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

        # Simple stiction/dead-zone model
        if abs(u) < self.stiction and abs(th_dot) < 0.15:
            eff_u = 0.0
        else:
            eff_u = u - np.sign(u) * min(abs(u), self.stiction)

        # Motor torque model
        tau = (self.kt / self.Rm) * (eff_u - self.km * th_dot)

        # Encoder cable restoring torque
        cable_torque = -self.k_cable * theta

        s = np.sin(alpha)
        c = np.cos(alpha)

        m11 = self.J_r + self.m_p * self.L_r**2 + self.m_p * self.l_p**2 * s**2
        m12 = self.m_p * self.L_r * self.l_p * c
        m22 = self.J_p + self.m_p * self.l_p**2

        M = np.array(
            [
                [m11, m12],
                [m12, m22],
            ],
            dtype=np.float64,
        )

        # Practical Coriolis/centrifugal + damping structure
        c11 = self.D_r + self.m_p * self.l_p**2 * np.sin(2.0 * alpha) * al_dot
        c12 = -self.m_p * self.L_r * self.l_p * s * al_dot
        c21 = -0.5 * self.m_p * self.l_p**2 * np.sin(2.0 * alpha) * th_dot
        c22 = self.D_p

        C = np.array(
            [
                [c11, c12],
                [c21, c22],
            ],
            dtype=np.float64,
        )

        G_vec = np.array(
            [
                0.0,
                self.m_p * self.g * self.l_p * np.sin(alpha + self.tilt),
            ],
            dtype=np.float64,
        )

        q_dot = np.array([th_dot, al_dot], dtype=np.float64)
        rhs = np.array([tau + cable_torque, 0.0], dtype=np.float64) - C @ q_dot - G_vec

        q_ddot = np.linalg.solve(M, rhs)

        return np.array(
            [
                th_dot,
                al_dot,
                q_ddot[0],
                q_ddot[1],
            ],
            dtype=np.float64,
        )

    def step(self, action):
        self._step_count += 1

        requested = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        requested = float(np.clip(requested, -ACTION_LIMIT, ACTION_LIMIT))

        # One-step delay + voltage filtering.
        # This makes simulation less fake and closer to deployment.
        applied_request = self.delayed_action
        self.delayed_action = requested

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

            # Hard-stop bounce model
            if self.state[0] > self.hard_stop_angle:
                self.state[0] = self.hard_stop_angle
                self.state[2] = -0.25 * self.state[2]

            elif self.state[0] < -self.hard_stop_angle:
                self.state[0] = -self.hard_stop_angle
                self.state[2] = -0.25 * self.state[2]

        self.state[1] = self._wrap_pi(self.state[1])

        theta, alpha, th_dot, al_dot = self.state

        terminated = bool(
            abs(th_dot) > 80.0
            or abs(al_dot) > 100.0
            or not np.all(np.isfinite(self.state))
        )

        # ------------------------------------------------------------------
        # Reward: energy-assisted swing-up + upright stabilization
        # ------------------------------------------------------------------

        # Target:
        # theta = 0
        # alpha = +/-pi
        alpha_error = abs(self._wrap_pi(alpha - np.pi))
        theta_norm = theta / self.hard_stop_angle

        # Pendulum energy.
        # At bottom: E approx 0
        # At upright: E_target approx 2*m*g*l
        E = (
            0.5 * self.J_p * al_dot**2
            + self.m_p * self.g * self.l_p * (1.0 - np.cos(alpha))
        )
        E_target = 2.0 * self.m_p * self.g * self.l_p
        energy_error = abs(E - E_target) / max(E_target, 1e-6)

        # Use tanh so high velocity does not create insane negative rewards.
        energy_penalty = np.tanh(energy_error)

        upright_shape = (1.0 - np.cos(alpha)) / 2.0
        upright = np.exp(-4.0 * alpha_error**2)
        centered = np.exp(-2.0 * theta**2)
        slow = np.exp(-0.02 * (th_dot**2 + al_dot**2))

        reward = 0.0

        # 1. Swing-up guidance
        reward += 4.0 * upright_shape
        reward -= 1.5 * energy_penalty

        # 2. Stabilization when close to upright
        if alpha_error < 0.45:
            reward += 18.0 * upright
            reward += 10.0 * upright * centered * slow

        # 3. Keep arm near centre and away from hard stops
        reward -= 1.5 * theta_norm**2
        reward -= 20.0 * max(0.0, abs(theta) - 0.85 * self.hard_stop_angle) ** 2

        # 4. Penalize violent motion and voltage abuse
        reward -= 0.015 * th_dot**2
        reward -= 0.008 * al_dot**2
        reward -= 0.004 * voltage**2
        reward -= 0.02 * (voltage - self.prev_voltage) ** 2

        # 5. Failure penalty
        if terminated:
            reward -= 250.0

        self.prev_voltage = voltage

        truncated = self._step_count >= self._max_episode_steps

        return self._get_obs(), float(reward), terminated, truncated, {}

    def render(self):
        return None

    def close(self):
        pass
