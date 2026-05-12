# Plotting config settings
import numpy as np

# --- HARDWARE & PHYSICS ---
VELOCITY_FILTER = 0.65  # Increased for smoother state inputs
STICTION_VOLTAGE = 0.45 
HARD_STOP_RAD = 2.37   # ±136 degrees
ENCODER_RES = 2048

# --- RL TRAINING HYPERPARAMETERS ---
PPO_TOTAL_STEPS = 2000000
TD3_TOTAL_STEPS = 1000000
SAC_TOTAL_STEPS = 1000000
N_ENVS = 8
LEARNING_RATE = 3e-4

# --- DEPLOYMENT PERFORMANCE TUNING ---
POWER_GAIN = 1.0
MOTOR_INVERT = -1.0     # Invert voltage so +u moves arm LEFT
THETA_INVERT = -1.0     # Invert arm sensor so LEFT is positive
PENDULUM_INVERT = -1.0  # Invert pendulum sensor so CCW is positive
ACTION_FILTER = 0.25   # Increased to smooth out over-corrections
SAFETY_LIMIT_RAD = 2.1  # Warning zone
SAFETY_KILL_RAD = 2.3   # Shutdown zone

# --- PLOTTING ---
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
