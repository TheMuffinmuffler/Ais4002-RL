"""Central configuration for the QUBE-Servo 2 RL project.

The values here are deliberately conservative.  They are meant to give usable
models in a reasonable time, not to chase a 10-hour training run.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical QUBE-Servo 2 model parameters
# Source: 2025 AIS4002 QUBE pendulum report table.
# Convention used everywhere in this repo:
#   theta = arm angle, 0 rad at the centre
#   alpha = pendulum angle, 0 rad hanging down, +/- pi rad upright
# ---------------------------------------------------------------------------
M_R = 0.095                  # kg, rotary arm mass; kept for documentation
L_R = 0.086                  # m, rotary arm length
J_R = 0.0001172              # kg m^2, rotary arm inertia
D_R = 0.0004                 # N m s/rad, arm damping

M_P = 0.053                  # kg, pendulum mass
L_P = 0.128                  # m, pendulum total length
L_P_COG = 0.128 / 2.0        # m, pendulum centre of gravity
J_P = 0.0000235              # kg m^2, pendulum inertia
D_P = 0.000003               # N m s/rad, pendulum damping

G = 9.81                     # m/s^2
RM = 8.94                    # Ohm
KT = 0.0431                  # N m/A
KM = 0.0431                  # V/(rad/s)
K_CABLE = 0.002              # N m/rad, encoder-cable restoring stiffness

# ---------------------------------------------------------------------------
# Simulation and hardware emulation
# ---------------------------------------------------------------------------
DT = 0.02                    # 50 Hz control loop
EPISODE_STEPS = 500
ACTION_LIMIT = 24.0          # V, safe training/deployment voltage limit
HARD_STOP_RAD = 2.37         # roughly +/-136 deg
ENCODER_RES = 2048

VELOCITY_FILTER = 0.65       # used in simulation observation and deployment
ACTION_FILTER = 0.25         # voltage low-pass filter for deployment/sim
STICTION_VOLTAGE = 0.35      # simple motor dead-zone estimate

# Domain randomization: not too wide. Brutal truth: huge randomization makes
# short training worse, not more robust.
DOMAIN_RANDOMIZATION_SCALE = 0.12
DAMPING_RANDOMIZATION = (0.7, 1.8)
MOTOR_RANDOMIZATION = (0.9, 1.1)
STICTION_RANDOMIZATION = (0.7, 1.3)
TILT_RANDOMIZATION_RAD = np.deg2rad(3.0)

# ---------------------------------------------------------------------------
# Training presets. These are intentionally modest for laptop runtime.
# Use --steps on the train scripts to override.
# ---------------------------------------------------------------------------
N_ENVS = 12
LEARNING_RATE = 3e-4
PPO_TOTAL_STEPS = 600_000
TD3_TOTAL_STEPS = 350_000
SAC_TOTAL_STEPS = 1_000_000
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 20_000

# ---------------------------------------------------------------------------
# Deployment mapping. These must be identical for PPO, TD3, and SAC.
# If the motor goes the wrong way on your hardware, change MOTOR_INVERT only.
# ---------------------------------------------------------------------------
POWER_GAIN = 1.0
MOTOR_INVERT = -1.0
THETA_INVERT = 1.0
PENDULUM_INVERT = 1.0
SAFETY_LIMIT_RAD = 2.1
SAFETY_KILL_RAD = 2.3
MAX_SAME_SIDE_HITS = 20

# ---------------------------------------------------------------------------
# Plotting defaults used by the original GUI/liveplot files
# ---------------------------------------------------------------------------
UPDATE_FREQUENCY = 10
MAX_DATA_POINTS = 500
PLOT1_TITLE = "Motor Angle"
PLOT2_TITLE = "Pendulum Angle"
PLOT3_TITLE = "RPM"
PLOT4_TITLE = "Voltage"
PLOT1_AXISTITLE = "Degrees"
PLOT2_AXISTITLE = "Degrees"
PLOT3_AXISTITLE = "RPM"
PLOT4_AXISTITLE = "Volts"
PLOT1_LEGENDS = ("Actual", "Target")
PLOT2_LEGENDS = ("Actual", "Target")
PLOT3_LEGENDS = ("Actual", "Target")
PLOT4_LEGENDS = ("Actual", "Target")
PLOT1_VALUE_1 = "MOTOR_ANGLE"
PLOT1_VALUE_2 = "MOTOR_TARGET_ANGLE"
PLOT2_VALUE_1 = "PENDULUM_ANGLE"
PLOT2_VALUE_2 = "PENDULUM_TARGET_ANGLE"
PLOT3_VALUE_1 = "RPM"
PLOT3_VALUE_2 = "RPM_TARGET"
PLOT4_VALUE_1 = "VOLTAGE"
PLOT4_VALUE_2 = "NONE"
