import sys
import os

# --- INLINE COMPATIBILITY SHIM ---
def apply_compat_shims():
    try:
        import numpy.core.numeric as numeric
        sys.modules['numpy._core.numeric'] = numeric
        import numpy.core.multiarray as multiarray
        sys.modules['numpy._core.multiarray'] = multiarray
        import numpy.core.umath as umath
        sys.modules['numpy._core.umath'] = umath
        try:
            import numpy._core
        except ImportError:
            import numpy.core as core
            sys.modules['numpy._core'] = core
    except (ImportError, AttributeError): pass
    try:
        import stable_baselines3.common.utils as sb3_utils
        for name in ["FloatSchedule", "ConstantSchedule", "LinearSchedule"]:
            if not hasattr(sb3_utils, name):
                class DummySchedule:
                    def __init__(self, value=0.0, *args, **kwargs): self.value = value
                    def __call__(self, *args, **kwargs): return self.value
                setattr(sb3_utils, name, DummySchedule)
    except ImportError: pass
    try:
        import numpy.random._pickle as nprp
        for bg_name in ['MT19937', 'PCG64', 'PCG64DXSM', 'Philox', 'SFC64']:
            if hasattr(nprp, bg_name):
                bg_cls = getattr(nprp, bg_name)
                if hasattr(nprp, 'BitGenerators'):
                    nprp.BitGenerators[bg_cls] = bg_cls
    except Exception: pass
    try:
        from gymnasium.spaces.space import Space
        def patched_setstate(self, state):
            if isinstance(state, dict): self.__dict__.update(state)
        Space.__setstate__ = patched_setstate
    except (ImportError, AttributeError): pass

apply_compat_shims()
# --------------------------------

import numpy as np
import torch
from stable_baselines3 import SAC
from qube_env import QubeEnv
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def test():
    os.makedirs("plots", exist_ok=True)
    # Load the trained SAC model
    model_path = "models/qube_sac_final.zip"
    device = get_device()

    try:
        model = SAC.load(model_path, device=device)
        print(f"Loaded SAC model from {model_path} on {device}")
    except Exception as e:
        print(f"Could not load SAC model: {e}")
        return

    env = QubeEnv()
    obs, _ = env.reset()

    history = []
    total_reward = 0.0

    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        # Observation format:
        # [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]
        theta = np.arctan2(obs[0], obs[1])
        alpha = np.arctan2(obs[2], obs[3])

        history.append([
            theta,
            alpha,
            obs[4],
            obs[5],
            float(action[0]),
            reward
        ])

        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps.")
            break

    print(f"Total reward: {total_reward:.2f}")

    if len(history) == 0:
        print("No data collected.")
        return

    history = np.array(history)
    cumulative_reward = np.cumsum(history[:, 5])

    # --- Figure 1: Time Series Dynamics ---
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(np.rad2deg(history[:, 0]), label="Theta (Arm)", color='blue')
    plt.plot(np.rad2deg(history[:, 1]), label="Alpha (Pendulum)", color='red')
    plt.axhline(0, linestyle="--", color='black', alpha=0.3) # 0 is Upright now
    plt.ylabel("Angle (deg)")
    plt.title("QUBE-Servo 2 SAC Performance (0 is Upright)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(np.rad2deg(history[:, 2]), label="Theta dot", color='cyan')
    plt.plot(np.rad2deg(history[:, 3]), label="Alpha dot", color='magenta')
    plt.ylabel("Velocity (deg/s)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.step(range(len(history)), history[:, 4], label="Action (Voltage)", color='green')
    plt.ylabel("Voltage (V)")
    plt.ylim(-11, 11)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(history[:, 5], label="Step Reward", color='orange', alpha=0.6)
    plt.plot(cumulative_reward / (np.arange(len(history)) + 1), label="Avg Reward", color='red')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/sac_dynamics.png", dpi=200)
    print("Dynamics plot saved to plots/sac_dynamics.png")

    # --- Figure 2: Analysis & Phase Space ---
    plt.figure(figsize=(12, 5))

    # Phase Plot: Pendulum
    plt.subplot(1, 2, 1)
    plt.plot(np.rad2deg(history[:, 1]), np.rad2deg(history[:, 3]), color='purple')
    plt.axvline(0, color='red', linestyle='--', alpha=0.5) # 0 is Upright
    plt.xlabel("Alpha (deg)")
    plt.ylabel("Alpha Dot (deg/s)")
    plt.title("Pendulum Phase Space")
    plt.grid(True, alpha=0.3)

    # Cumulative Reward
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward, color='brown', linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Learning Progress (Cumulative)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/sac_analysis.png", dpi=200)
    print("Analysis plot saved to plots/sac_analysis.png")

    env.close()


if __name__ == "__main__":
    test()
