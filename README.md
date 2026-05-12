# Quanser Qube Servo 2 Control & RL Project

## Introduction
This project provides a comprehensive framework for controlling and simulating the Quanser Qube Servo 2 (Inverted Pendulum). It features hardware interfacing via Serial, a custom physics simulation, Reinforcement Learning (RL) training with multiple state-of-the-art algorithms, and classical control methods.

The primary goal is to achieve a robust "swing-up and balance" behavior that transfers seamlessly from simulation to physical hardware.

## Key Features
- **Multiple RL Algorithms**: Support for **PPO**, **SAC**, and **TD3** via Stable Baselines 3.
- **High-Fidelity Simulation**: Custom Gymnasium environment (`QubeEnv`) with RK4 integration.
- **Sim-to-Real Robustness**:
    - **Domain Randomization**: Randomizes mass, length, damping, and stiction.
    - **Hardware Emulation**: Models latency (20ms), sensor noise, and encoder quantization.
    - **Action Filtering**: Smooths control signals to protect hardware.
    - **Robustness Training**: Models are trained with random "force perturbations" (hits) while balancing to improve real-world recovery.
- **Verified Hardware Mapping**: Deployment scripts automatically map hardware encoders (Reset at BOTTOM=0) to Simulation coordinates (TOP=0, BOTTOM=-180).

## Project Structure

### 1. Reinforcement Learning
- **Environments**:
    - `qube_env.py`: The core Gymnasium environment. Alpha=0 is TOP.
- **Training**:
    - `train_rl.py`: Main script for PPO training/retraining.
    - `train_rl_SAC.py`: Specialized script for fresh SAC training.
    - `train_rl_TD3.py`: Specialized script for TD3 training.
- **Deployment**:
    - `deploy_qube_serial.py`: Deploys trained PPO models to hardware.
    - `deploy_qube_serial_SAC.py`: Deploys SAC models with optimized coordinate mapping.
    - `deploy_qube_serial_TD3.py`: Deploys TD3 models.
- **Evaluation**:
    - `test_rl.py` / `test_rl_SAC.py` / `test_rl_TD3.py`: Evaluate models in simulation.
    - `check_upright.py`: Quick diagnostic for model balance stability.

### 2. Hardware Interface & Diagnostics
- **Core**:
    - `QUBE.py`: Python serial driver for the Qube Servo 2.
    - `com.py`: Serial protocol definitions.
    - `QUBE/`: C++ firmware for the microcontroller.
- **Diagnostics**:
    - `check_sensors.py`: Verify encoder and RPM readings.
    - `check_polarity.py`: Ensure motor and pendulum directions are correct.
    - `check_limits.py`: Test the software safety limits.
    - `check_upright.py`: Verify the "top" position calibration.

### 3. Classical Control
- `main.py`: Entry point for the live PID tuner GUI.
- `control.py`: Central file for implementing custom control laws.
- `PID.py`: Robust PID implementation with anti-windup.
- `inverted_pendulum.py`: Energy-based swing-up + PD balance controller.

## Getting Started

### Installation
1. Install dependencies:
   ```bash
   python install.py
   ```
2. (Optional) Manual install:
   ```bash
   pip install gymnasium stable-baselines3 torch numpy pyserial PyQt5 pyqtgraph
   ```

### Hardware Setup
1. Upload the code in `QUBE/examples/Python_Serial/Python_Serial.ino` to your Arduino/Teensy.
2. Find your Serial port and update `COM_PORT` in `control.py`.
3. Verify connection:
   ```bash
   python check_sensors.py
   ```

### Training an Agent
To train a new SAC agent (recommended for best performance):
```bash
python train_rl_SAC.py
```
Models are saved in the `models/` directory.

### Deployment
To run your trained model on the physical Qube:
```bash
python deploy_qube_serial_SAC.py
```

## Technical Details

### Observation Space (7-D)
The model receives the following state vector:
1. `sin(theta)` (Motor Arm)
2. `cos(theta)`
3. `sin(alpha)` (Pendulum)
4. `cos(alpha)`
5. `theta_dot` (Filtered velocity)
6. `alpha_dot` (Filtered velocity)
7. `prev_voltage` (Normalized)

### Coordinate System
- **Simulation**: `alpha = 0` is the upright (TOP) position.
- **Hardware**: The encoders are typically reset while the pendulum is hanging down (`alpha = pi`). The deployment scripts handle the mapping to ensure the model sees `0` at the top.

## Safety Features
- **Software Safety Wall**: The motor is automatically cut if `|theta| > 2.3` radians (~130°).
- **Current Protection**: The firmware monitors for stalls and amplifier faults, indicated by a Red LED.
