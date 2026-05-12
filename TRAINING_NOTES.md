# Training Notes & Observations

This file tracks insights, reward analysis, and "lessons learned" during the training of Reinforcement Learning agents for the Qube Servo 2.

## SAC Training Dynamics (v4)

### The "-43,700" Red Flag
A stable reward around **-43,700** strongly suggests the model is **NOT** balancing and has reached a local optimum of "giving up."

#### Reward Breakdown:
*   **Balance Reward**: Perfect balance ($\cos(\alpha) > 0.95$) gives $+20$/step $\approx +10,000$ per episode.
*   **"Hanging Down" Penalty**: When $\cos(\alpha) = -1$, the `dist_upright` penalty is $\approx 80$/step $\approx -40,000$ per episode.
*   **Local Optimum**: Rewards in the $-40k$ to $-45k$ range indicate the model has learned that "doing nothing" is safer than "failing spectacularly." It avoids `safety_penalty` and `jerk_penalty` by letting the pendulum hang still.

### Why Models Get Stuck:
1.  **Aggressive Penalties**: The `energy_error` ($500 \times$) and `dist_upright` are very high. If the model swings and misses, the negative reinforcement may "scare" the policy into staying still.
2.  **Overfitting to Safety**: High gradient update frequencies can cause the model to overfit to a "safe" (but failing) strategy before it has experienced enough successful swing-ups.

### Strategy Recommendations:
*   **Patience**: SAC often has an "Aha!" moment. It may stay at $-40k$ for $300k$ steps and then suddenly "click," with rewards jumping to $+5k$ or higher.
*   **Threshold for Intervention**: If the reward is still $\approx -43k$ at $500,000$ steps, consider:
    *   Increasing `ent_coef` (entropy) to force more exploration.
    *   Reducing the `energy_error` penalty weight to make "swinging" less risky.

## Environment Robustness (Sim-to-Real)

### Randomized Surface Tilt
The environment now includes a randomized base tilt between **0 and 5 degrees** (0.087 rad).
*   **Purpose**: Real-world tables are rarely perfectly level. Training with a randomized tilt forces the agent to learn a robust balance strategy that can "lean" into the tilt, effectively learning to compensate for steady-state bias (integrator-like behavior).
*   **Implementation**: A `self.tilt` parameter is randomized during reset and added to the gravity term in the dynamics: `sin(alpha + self.tilt)`.

## Deployment Troubleshooting

### The "Helicopter Effect" (Continuous Spinning)
If the model swings up successfully but then enters a high-speed spin (helicoptering) rather than balancing:

1.  **Diagnosis**: This is usually caused by **Over-Correction**. The hardware gain is higher than what the model expects, so a small "catch" movement turns into a massive "push" that overshoots the balance point.
2.  **Fix #1: Reduce `POWER_GAIN`**: Lower the gain (e.g., from 1.5 to 1.2). This reduces the strength of the corrections.
3.  **Fix #2: Increase `ACTION_FILTER`**: Higher filtering (e.g., 0.25) smooths out voltage spikes, preventing the sudden "kicks" that trigger the spinning loop.
4.  **Fix #3: Check Polarity**: Ensure `PENDULUM_INVERT` and `MOTOR_INVERT` are correct. If the agent moves *away* from the pendulum at the top, the feedback is positive instead of negative.

---
*Note: Keep an eye on the rewards. A jump toward 0 or positive values indicates the agent has discovered the balance strategy.*
