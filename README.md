# Quanser Qube Servo 2 Control & RL Project

## Introduction to Reinforcement Learning & This Project

### What is Reinforcement Learning?
Reinforcement Learning (RL) is a branch of Machine Learning where an **Agent** learns to make decisions by performing **Actions** in an **Environment** to maximize a cumulative **Reward**. Unlike supervised learning, the agent is not told which actions to take; instead, it must discover which actions yield the most reward by trying them (trial and error).

In this project:
- **The Agent**: The PPO (Proximal Policy Optimization) algorithm.
- **The Environment**: The Quanser Qube Servo 2 (either the physical hardware or the `QubeEnv` simulation).
- **The State**: What the agent "sees" (angles of the arm and pendulum, and their angular velocities).
- **The Action**: The voltage applied to the motor to move the arm.
- **The Reward**: A mathematical signal that tells the agent how well it's doing (e.g., high reward for keeping the pendulum upright, penalties for falling or moving the arm too far).

### The Challenge: The Inverted Pendulum
The inverted pendulum is a classic problem in control theory and robotics. It is **inherently unstable** and **non-linear**. Keeping the pendulum balanced at the top is like balancing a broomstick on your fingertip. It requires:
1. **Swing-up**: Moving the arm back and forth to build enough momentum to lift the pendulum from a hanging position to an upright one.
2. **Balancing**: Once upright, applying tiny, precise corrections to keep it there.

### Why use RL here?
While classical control (like PID) can balance the pendulum once it's already near the top, designing a single controller that can both "swing-up" and "balance" automatically is complex. RL excels at discovering these non-linear strategies through simulation, allowing the agent to learn the physics of the system and "solve" the problem without a human explicitly writing the control equations for every possible state.

---

This project provides a comprehensive framework for controlling and simulating the Quanser Qube Servo 2 (Inverted Pendulum). It features hardware interfacing via Serial, a custom physics simulation, Reinforcement Learning (RL) training, and classical control methods.

## Project Structure

The project is divided into several functional layers:

### 1. Hardware Interface (`QUBE.py`, `QUBE/`)
- **`QUBE.py`**: A Python wrapper for serial communication. It allows reading encoder angles (motor and pendulum), RPM, and current, while sending motor voltage/speed and LED color commands.
- **`QUBE/`**: Contains C++ and Arduino code (`.ino`) that must be uploaded to the microcontroller connected to the Qube hardware.
- **`com.py`**: Configuration for serial communication (port, baudrate) and data packet definitions.

### 2. Simulation & Reinforcement Learning
- **`qube_env.py`**: A custom [Gymnasium](https://gymnasium.farama.org/) environment. It implements the non-linear dynamics of the inverted pendulum using RK4 integration. Includes **Domain Randomization** for better sim-to-real transfer.
- **`train_rl.py`**: Trains a PPO (Proximal Policy Optimization) agent using [Stable Baselines 3](https://stable-baselines3.readthedocs.io/).
- **`test_rl.py`**: Evaluates the trained agent in simulation and generates performance plots (`rl_test_plot.png`).
- **`deploy_qube_serial.py`**: A deployment script specifically designed for serial communication. It includes a **Safety Wall** that cuts power if the arm exceeds ±90°, protecting the hardware.
- **`check_upright.py`**: A quick diagnostic tool that calculates the "upright percentage" of a trained model in simulation.
- **`real_life_deploy.py`**: Script to run the trained RL model on actual hardware using the `quanser_robots` library.

### 3. Classical Control & Tuning
- **`main.py`**: Entry point for live PID tuning. It opens a GUI with real-time plots and sliders to adjust parameters on the fly.
- **`control.py`**: The core control logic used by `main.py`. This is where you implement your own control laws.
- **`PID.py`**: A standard Proportional-Integral-Derivative controller implementation with optional anti-windup.
- **`inverted_pendulum.py`**: Implements an energy-based swing-up controller and a linear balance controller with automatic mode switching.
- **`validate_env.py`**: A script to verify the simulation's physics by running a known good controller.

### 4. Visualization & Logging
- **`liveplot.py` / `gui.py`**: PyQt5 and Tkinter based interfaces for real-time data visualization and hardware interaction.
- **`logger.py`**: Saves all flight/test data (angles, RPM, voltage) into the `Data/` folder as `.csv` files for later analysis.
- **`config.py`**: Central configuration for plotting settings (axes, frequency, data limits).

### 5. Standalone Arduino Examples (`QUBE/examples/`)
The project also includes standalone C++/Arduino examples for users who wish to run control loops directly on the microcontroller without a Python host:
- **`Inverted_Pendulum`**: A complete standalone swing-up and balance controller.
- **`Position_Control`**: Implements basic PID position control for the motor arm.
- **`Velocity_Control`**: Implements PID velocity (RPM) control.
- **`SD_Logging`**: Demonstrates how to log sensor data to an SD card (if hardware is supported).
- **`QUBE_guide`**: A basic example demonstrating how to use the `QUBE.hpp` library functions.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Arduino IDE (for hardware setup)

### Installation
Run the installation script to fetch dependencies:
```bash
python install.py
```
*Note: You may also need to install `gymnasium`, `stable-baselines3`, `torch`, and `numpy` manually if they are not in the install script.*

### Running the Project

#### 1. Real-Time PID Tuning (Hardware)
1. Connect the Qube hardware.
2. Update `COM_PORT` in `control.py` to match your device.
3. Run the main tuner:
   ```bash
   python main.py
   ```

#### 2. Training the AI (Simulation)
To train a new RL agent to balance the pendulum:
```bash
python train_rl.py
```
The model will be saved in `models/qube_ppo_final.zip`.

#### 3. Testing the AI
To see how your trained model performs in the simulator:
```bash
python test_rl.py
```

#### 4. Validating the Simulation
To verify the physics environment is working as expected:
```bash
python validate_env.py
```

---

## Technical Hardware Details

### Sensors & Actuators
- **Encoders**: High-resolution optical encoders with **2048 counts per revolution**.
- **Motor**: 24V DC motor. The control signal is mapped from a digital value of 0-999 to 0-24V.
- **Microcontroller**: Typically a Teensy or similar Arduino-compatible board running the code in `QUBE/examples/`.
- **Communication**: SPI is used between the microcontroller and the QFLEX2 EMBEDDED board at 1MHz.

### Safety Features
The system includes several layers of protection:
1. **Stall Detection**: The C++ library monitors motor current and speed. If a stall is detected (high current, no movement), the LED will turn red.
2. **Amplifier Fault**: Detects hardware-level errors in the motor driver.
3. **Software Safety Wall**: The `deploy_qube_serial.py` script includes a ±90° software limit on the arm's position to prevent the pendulum from hitting the base or entangling wires.
4. **Visual Feedback**: The LED on the Qube is used to signal state:
   - **White**: Initializing/Calibration.
   - **Green**: Active/Normal Operation.
   - **Red/Flashing Red**: Error, Stall, or Hardware Fault.

---

## Troubleshooting

### Serial Connection Issues
- **Permission Denied**: On Linux/Mac, you may need to grant permissions to the serial port: `sudo chmod 666 /dev/ttyACM0` (replace with your port).
- **Wrong Port**: Check your Arduino IDE or use `ls /dev/cu.*` (Mac) or `ls /dev/tty*` (Linux) to find the correct `COM_PORT`. Update this in `control.py` or `com.py`.
- **Busy Port**: Ensure the Arduino Serial Monitor is closed before running the Python scripts.

### Reinforcement Learning
- **No 'models' folder**: The `train_rl.py` script creates this folder automatically. If you get a "File Not Found" error, ensure you have run the training at least once.
- **CUDA/MPS Errors**: The project automatically detects GPU acceleration. If you have issues, you can force CPU usage by changing the `device` variable in `train_rl.py` to `"cpu"`.

### Hardware Safety
- **LED is Red**: This usually indicates a **Stall Error**. The motor has been under high load without moving for too long. Power cycle the Qube or check if the arm is physically blocked.
- **Safety Wall Triggered**: If the arm hits ±90°, the `deploy_qube_serial.py` script will cut the voltage. Simply restart the script and ensure the pendulum is hanging still before starting.

### Dependencies
- If `pip install` fails for `PyQt5`, ensure you have a modern version of `pip` and `setuptools`:
  ```bash
  python -m pip install --upgrade pip setuptools
  ```


### Physics Simulation
The `QubeEnv` uses the following equations of motion to simulate the system:
- **RK4 Integration**: Provides high-accuracy state updates at 50Hz.
- **Domain Randomization**: Randomizes mass, length, and friction parameters during training to make the AI robust to real-world variations.
- **Reward Function**: Penalizes deviation from the upright position ($ \alpha = \pi $) and excessive motor movement or arm deviation ($ \theta $).

### Communication Protocol
The system communicates via a custom binary protocol over Serial (115200 baud). Data is exchanged in fixed-size packets to ensure deterministic timing (50Hz - 300Hz).

#### Python to Arduino (10-byte Control Packet)
| Byte | Description | Data Type |
|---|---|---|
| 0 | Reset Motor Encoder | Boolean (1 = Reset) |
| 1 | Reset Pendulum Encoder | Boolean (1 = Reset) |
| 2-3 | Red LED Intensity | 16-bit Int (0-999) |
| 4-5 | Green LED Intensity | 16-bit Int (0-999) |
| 6-7 | Blue LED Intensity | 16-bit Int (0-999) |
| 8-9 | Motor Command | 16-bit Int (0-1998, Offset 999) |

#### Arduino to Python (12-byte Telemetry Packet)
| Bytes | Description | Format |
|---|---|---|
| 0-3 | Motor Angle | 4 bytes: [Rev MSB (incl. sign), Rev LSB, Ang MSB, Ang LSB] |
| 4-7 | Pendulum Angle | 4 bytes: [Rev MSB (incl. sign), Rev LSB, Ang MSB, Ang LSB] |
| 8-9 | Motor RPM | 2 bytes: [Sign bit + 15-bit value] |
| 10-11 | Motor Current | 2 bytes: 16-bit absolute value (mA) |

---

### Control Strategy
- **Swing-up**: Uses energy-based control to inject just enough energy into the pendulum to reach the top.
- **Balance**: Switches to a PD or LQR controller when the pendulum is within a small angle (e.g., ±20°) of the vertical.
