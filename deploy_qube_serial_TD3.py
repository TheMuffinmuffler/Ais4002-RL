import numpy as np
import time
import torch
from stable_baselines3 import TD3

from QUBE import QUBE
from control import COM_PORT


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def deploy():
    # 1. Load trained TD3 model
    model_path = "models/qube_td3_final.zip"
    device = get_device()

    try:
        model = TD3.load(model_path, device=device)
        print(f"TD3 model loaded from {model_path} on {device}")
    except Exception as e:
        print(f"Error loading TD3 model: {e}")
        return

    # 2. Initialize connection to Teensy / QUBE
    port = COM_PORT
    baudrate = 115200

    try:
        qube = QUBE(port, baudrate)
        print(f"Connected to QUBE on {port}")
    except Exception as e:
        print(f"Could not connect to {port}: {e}")
        return

    # 3. Calibration
    print("Initializing encoders...")
    qube.setRGB(999, 999, 999)  # White

    qube.resetMotorEncoder()
    time.sleep(1.0)

    qube.resetPendulumEncoder()
    time.sleep(0.5)

    qube.update()

    print("--- READY ---")
    print("1. Hold the pendulum hanging down and still.")
    print("2. Starting in 5 seconds...")
    time.sleep(5.0)

    qube.setRGB(0, 999, 0)  # Green

    t_last = time.time()

    qube.update()
    prev_theta = np.deg2rad(qube.getMotorAngle())
    prev_alpha = np.deg2rad(qube.getPendulumAngle())

    max_arm_angle = np.deg2rad(90.0)
    max_voltage = 10.0

    try:
        while True:
            loop_start = time.time()

            # 4. Read sensors
            qube.update()

            theta_deg = qube.getMotorAngle()
            alpha_deg = qube.getPendulumAngle()

            theta = np.deg2rad(theta_deg)
            alpha = np.deg2rad(alpha_deg)

            # 5. Compute dt
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now

            if dt <= 0.0:
                continue

            # 6. Estimate velocities
            theta_dot = (theta - prev_theta) / dt
            alpha_dot = (alpha - prev_alpha) / dt

            prev_theta = theta
            prev_alpha = alpha

            # 7. Build observation vector
            # Must match training environment:
            # [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]
            obs = np.array([
                np.sin(theta),
                np.cos(theta),
                np.sin(alpha),
                np.cos(alpha),
                theta_dot,
                alpha_dot
            ], dtype=np.float32)

            # 8. Predict TD3 action
            action, _ = model.predict(obs, deterministic=True)
            voltage = float(action[0])

            # 9. Clip voltage for safety
            voltage = float(np.clip(voltage, -max_voltage, max_voltage))

            # 10. Safety wall
            if abs(theta) > max_arm_angle:
                print(f"Safety wall hit. Theta = {theta_deg:.1f} deg. Motor stopped.")
                voltage = 0.0
                qube.setRGB(999, 0, 0)  # Red

            # 11. Send voltage
            qube.setMotorVoltage(voltage)

            # 12. Maintain approximately 50 Hz control loop
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, 0.02 - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping... setting motor voltage to 0.")

    finally:
        qube.setMotorVoltage(0.0)
        qube.setRGB(999, 0, 0)  # Red

        if hasattr(qube, "master"):
            qube.master.close()


if __name__ == "__main__":
    deploy()