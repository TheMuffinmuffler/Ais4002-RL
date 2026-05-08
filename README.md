# Quanser Qube Servo 2 Control & RL Project

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
- **`real_life_deploy.py`**: Script to run the trained RL model on actual hardware.

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

## How it Works

### Physics Simulation
The `QubeEnv` uses the following equations of motion to simulate the system:
- **RK4 Integration**: Provides high-accuracy state updates at 50Hz.
- **Domain Randomization**: Randomizes mass, length, and friction parameters during training to make the AI robust to real-world variations.
- **Reward Function**: Penalizes deviation from the upright position ($ \alpha = \pi $) and excessive motor movement or arm deviation ($ \theta $).

### Communication Protocol
Data is sent/received over Serial in 10-byte packets. The Python side handles bit-shifting and scaling to convert raw encoder counts into meaningful degrees and RPM values.

### Control Strategy
- **Swing-up**: Uses energy-based control to inject just enough energy into the pendulum to reach the top.
- **Balance**: Switches to a PD or LQR controller when the pendulum is within a small angle (e.g., ±20°) of the vertical.
