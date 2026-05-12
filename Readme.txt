QUBE SERVO 2 - QUICK START GUIDE
================================

1. PREREQUISITES
   - Python 3.8+
   - Arduino IDE
   - Quanser Qube Servo 2 Hardware

2. INSTALLATION
   - Run 'python install.py' to install required libraries (pyserial, PyQt5, pyqtgraph).
   - Manually install 'stable-baselines3' and 'gymnasium' for Reinforcement Learning.

3. HARDWARE SETUP
   - Open Arduino IDE.
   - Flash 'QUBE/examples/Python_Serial/Python_Serial.ino' to your microcontroller.
   - Update 'COM_PORT' in 'control.py' to match your device (e.g., "COM3" or "/dev/ttyACM0").
4. DIAGNOSTICS (Test your setup)
   - Run 'python check_sensors.py' to verify hardware mapping (TOP should be 0, BOTTOM should be -180).
   - Run 'python gui.py' to see real-time plots and test motor movement.

5. REINFORCEMENT LEARNING
   - TRAIN: Run 'python train_rl_SAC.py' (Now includes "Hit the Pendulum" robustness training).
   - DEPLOY: All 'deploy_qube_serial_*.py' scripts use the verified BOTTOM=0 to Sim-Mapping.

6. SAFETY NOTE
6. CLASSICAL CONTROL (PID)
   - Edit your control logic in 'control.py'.
   - Run 'python main.py' to launch the live tuner GUI.

7. SAFETY NOTE
   - Always ensure the pendulum has enough space to swing.
   - If the LED turns RED, the motor has stalled. Power cycle the device.
   - Software limits will cut power if the arm swings beyond ±130 degrees.
