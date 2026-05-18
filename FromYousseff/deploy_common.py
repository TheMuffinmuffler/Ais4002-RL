import os
import time
from datetime import datetime

import numpy as np
import torch

from compat import apply_compat_shims
from config import (
    ACTION_FILTER,
    ACTION_LIMIT,
    MAX_SAME_SIDE_HITS,
    MOTOR_INVERT,
    PENDULUM_INVERT,
    POWER_GAIN,
    SAFETY_KILL_RAD,
    THETA_INVERT,
    VELOCITY_FILTER,
)
from control import COM_PORT
from logger import AsyncLogger
from QUBE import QUBE

apply_compat_shims()


def _load_model(algo_name, model_path=None):
    if algo_name == "ppo":
        from stable_baselines3 import PPO as Algo
    elif algo_name == "td3":
        from stable_baselines3 import TD3 as Algo
    elif algo_name == "sac":
        from stable_baselines3 import SAC as Algo
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(base_dir, f"models/qube_{algo_name}_final.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Algo.load(model_path, device=device)
    print(f"Loaded {algo_name.upper()} model from {model_path} on {device}")
    return model


def _wrap_deg(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def _shortest_angle_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2.0 * np.pi) - np.pi


def _safe_read_angles(qube):
    """Return angles in the same convention used by qube_env.py."""
    theta_deg = qube.getMotorAngle() * THETA_INVERT
    alpha_deg = _wrap_deg(qube.getPendulumAngle() * PENDULUM_INVERT)
    return theta_deg, alpha_deg


def deploy(algo_name, rgb=(0, 999, 0), model_path=None):
    algo_name = algo_name.lower()
    try:
        model = _load_model(algo_name, model_path=model_path)
    except Exception as exc:
        print(f"Model load failed: {exc}")
        return

    try:
        qube = QUBE(COM_PORT, 115200)
        print(f"Connected to QUBE on {COM_PORT}")
    except Exception as exc:
        print(f"Connection failed: {exc}")
        return

    logger = None
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        print("Calibrating. Pendulum must be hanging down and arm must be centred.")
        qube.setRGB(999, 999, 999)
        qube.resetMotorEncoder()
        time.sleep(0.5)
        qube.resetPendulumEncoder()
        qube.update()

        print("Starting deployment in 3 seconds...")
        time.sleep(3.0)
        qube.setRGB(*rgb)

        log_filename = os.path.join(logs_dir, f"deploy_{algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        logger = AsyncLogger(log_filename)

        qube.update()
        theta_deg, alpha_deg = _safe_read_angles(qube)
        prev_theta = np.deg2rad(theta_deg)
        prev_alpha = np.deg2rad(alpha_deg)
        prev_voltage = 0.0
        th_dot_filt = 0.0
        al_dot_filt = 0.0
        hits_left = 0
        hits_right = 0
        was_left = False
        was_right = False
        t_start = time.time()
        t_last = t_start

        while True:
            loop_start = time.time()
            qube.update()
            theta_deg, alpha_deg = _safe_read_angles(qube)
            theta = np.deg2rad(theta_deg)
            alpha = np.deg2rad(alpha_deg)

            dt = loop_start - t_last
            t_last = loop_start
            if dt <= 0.0 or dt > 0.1:
                dt = 0.02

            th_dot_raw = (theta - prev_theta) / dt
            al_dot_raw = _shortest_angle_diff(alpha, prev_alpha) / dt
            th_dot_filt = VELOCITY_FILTER * th_dot_filt + (1.0 - VELOCITY_FILTER) * th_dot_raw
            al_dot_filt = VELOCITY_FILTER * al_dot_filt + (1.0 - VELOCITY_FILTER) * al_dot_raw

            obs = np.array(
                [
                    np.sin(theta),
                    np.cos(theta),
                    np.sin(alpha),
                    np.cos(alpha),
                    np.clip(th_dot_filt, -60.0, 60.0),
                    np.clip(al_dot_filt, -80.0, 80.0),
                    prev_voltage / ACTION_LIMIT,
                    (loop_start - t_start) / 10.0,  # Normalized time (assuming ~10s episodes)
                    alpha / (2.0 * np.pi),          # Normalized rotation count (9th dim)
                ],
                dtype=np.float32,
            )

            action, _ = model.predict(obs, deterministic=True)
            # Super-Env: action is normalized [-1, 1], scale it to ACTION_LIMIT
            requested_voltage = float(np.asarray(action).reshape(-1)[0]) * ACTION_LIMIT * POWER_GAIN * MOTOR_INVERT
            requested_voltage = float(np.clip(requested_voltage, -ACTION_LIMIT, ACTION_LIMIT))
            voltage = (1.0 - ACTION_FILTER) * prev_voltage + ACTION_FILTER * requested_voltage
            voltage = float(np.clip(voltage, -ACTION_LIMIT, ACTION_LIMIT))

            if abs(theta) > SAFETY_KILL_RAD:
                print(f"\nCRITICAL SAFETY KILL: Arm at {theta_deg:.2f} deg exceeds limit {np.rad2deg(SAFETY_KILL_RAD):.2f} deg.")
                qube.setMotorVoltage(0.0)
                break

            if theta > SAFETY_KILL_RAD:
                if not was_left:
                    hits_left += 1
                    was_left = True
                    print(f"\nLeft safety hit: {hits_left}/{MAX_SAME_SIDE_HITS}")
            else:
                was_left = False

            if theta < -SAFETY_KILL_RAD:
                if not was_right:
                    hits_right += 1
                    was_right = True
                    print(f"\nRight safety hit: {hits_right}/{MAX_SAME_SIDE_HITS}")
            else:
                was_right = False

            if hits_left >= MAX_SAME_SIDE_HITS or hits_right >= MAX_SAME_SIDE_HITS:
                print("\nSafety stop: too many hard-stop hits on one side.")
                break

            qube.setMotorVoltage(voltage)
            prev_theta = theta
            prev_alpha = alpha
            prev_voltage = voltage

            if int(loop_start * 10) != int((loop_start - dt) * 10):
                print(
                    f"{algo_name.upper()} | theta:{theta_deg:7.2f} deg | "
                    f"alpha:{alpha_deg:7.2f} deg | voltage:{voltage:6.2f} V",
                    end="\r",
                )

            logger.log([loop_start - t_start, dt, theta_deg, alpha_deg, th_dot_filt, al_dot_filt, voltage, algo_name.upper()])

            elapsed = time.time() - loop_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

    except KeyboardInterrupt:
        print("\nDeployment stopped by user.")
    finally:
        try:
            qube.setMotorVoltage(0.0)
            qube.setRGB(999, 0, 0)
            if hasattr(qube, "master"):
                qube.master.close()
        finally:
            if logger is not None:
                logger.stop()
