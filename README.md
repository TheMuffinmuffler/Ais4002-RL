# QUBE-Servo 2 Reinforcement Learning Project

This repository contains a high-fidelity Reinforcement Learning environment and training pipeline for the Quanser QUBE-Servo 2. It is designed to train robust agents capable of swinging up and balancing the pendulum from any starting position.

## Folder Structure

- **`/FromYousseff`**: **Primary Development Sandbox.** Contains the most up-to-date environment logic (`qube_env.py`), training scripts (`train_rl_SAC.py`), and evaluation tools.
- **`/FromYousseff/Lasthopemodel`**: **Official Deployment Target.** Contains the final, validated, and smoothed models ready for hardware.
- **`/FromYousseff/models`**: Storage for raw training checkpoints and intermediate weights.
- **`/logs`**: Tensorboard and CSV logs for training runs.
- **`/QUBE`**: Arduino firmware for the QUBE-Servo 2 serial interface.

---

## The Environment (`qube_env.py`)

The environment is a Gymnasium-compatible wrapper around a 4th-order ODE simulation of the QUBE-Servo 2.

### Observation Space (9-D)
1. `sin(theta)`, `cos(theta)` (Arm Angle)
2. `sin(alpha)`, `cos(alpha)` (Pendulum Angle)
3. `theta_dot` (Arm Velocity)
4. `alpha_dot` (Pendulum Velocity)
5. `voltage / 24` (Previous Action)
6. `step_count / 500` (Time progress)
7. `alpha / 2pi` (Normalized rotation count)

### Reward Structure (v6.11 "No Survival" Edition)
To prevent the "Lazy Agent" bug (where the agent earns points by doing nothing at the bottom), the reward function is strictly performance-based:

| Component | Max Value | Description |
| :--- | :--- | :--- |
| **Balance** | +200.0/step | Gaussian reward for being upright and still. |
| **Swing-Up** | +20.0/step | Parabolic reward for pendulum height. |
| **Stillness Jackpot**| +500.0/step | Growing reward for consecutive steps spent upright. |
| **Survival** | **0.0** | Removed to force exploration of swing-up. |

### Penalties
- **Termination:** `-200.0` if the agent crashes (extreme velocity or over-rotation).
- **Centering:** Scaled penalty for being away from the arm's center (0 rad).
- **Boundary:** Massive penalty (`-3000.0`) for moving toward the physical hard stops ($\pm 120^\circ$).

---

## Training Workflow

### 1. Training a New Model
To start a fresh SAC (Soft Actor-Critic) training run with a 512x512x512 architecture:
```bash
python FromYousseff/train_rl_SAC.py --fresh --steps 1000000 --device cuda
```

### 2. Surgical Refinement (Smoothing)
Models trained with high entropy can be "jittery." Before deploying to hardware, run a refinement to calm the control law (Entropy Locked at 0.1):
```bash
python FromYousseff/refine_sac.py --steps 300000 --device cuda
```

---

## Deployment Workflow

### 1. Identify the Final Model
The official deployment model is always located in:
`FromYousseff/Lasthopemodel/qube_sac_refined_FINAL.zip`

### 2. Hardware Deployment
To deploy to the physical QUBE-Servo 2 via serial:
```bash
python FromYousseff/deploy_qube_serial_SAC.py --model_path FromYousseff/Lasthopemodel/qube_sac_refined_FINAL.zip
```

---

## Engineering Standards
- **Sim-to-Real Robustness:** Training uses 2.5% frequency "pokes" and $\pm 12\%$ domain randomization.
- **Mastery:** Models are validated to ensure they hit the ~150k+ reward threshold (Master Balancer).
- **Safety:** Deployment scripts include a `SAFETY_KILL_RAD` to protect the hardware.
