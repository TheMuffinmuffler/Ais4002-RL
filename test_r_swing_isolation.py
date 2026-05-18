import numpy as np
import matplotlib.pyplot as plt

def compute_r_swing_only(alpha):
    """
    Isolated r_swing calculation.
    alpha = 0 is hanging down.
    alpha = pi is upright.
    """
    # Using the exact formula from the v6.1 env
    r_swing = 0.1 * (1.0 - np.cos(alpha))
    return r_swing

def visualize_r_swing():
    # Generate angles from -pi to pi (full circle)
    alphas = np.linspace(-np.pi, np.pi, 500)
    rewards = [compute_r_swing_only(a) for a in alphas]

    plt.figure(figsize=(10, 6))
    plt.plot(np.rad2deg(alphas), rewards, label='r_swing = 0.1 * (1 - cos(alpha))', color='blue', linewidth=2)
    
    # Annotate key points
    plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Hanging Down (0 deg)')
    plt.axvline(180, color='green', linestyle='--', alpha=0.5, label='Upright (180 deg)')
    plt.axvline(-180, color='green', linestyle='--', alpha=0.5)
    
    plt.title("Isolated r_swing Reward Gradient")
    plt.xlabel("Pendulum Angle (alpha) in Degrees")
    plt.ylabel("Reward Value")
    plt.xticks([-180, -90, 0, 90, 180])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the visualization
    save_path = "r_swing_isolation.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_r_swing()
