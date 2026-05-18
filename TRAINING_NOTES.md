# Training Notes & Observations

This file tracks insights, reward analysis, and "lessons learned" during the training of Reinforcement Learning agents for the Qube Servo 2.

## SAC Iterative Refinement (v6.x)

### v6.2: The "Smooth Gradient" Strategy
*   **Change**: Removed `r_height` (all-or-nothing), buffed `r_swing` to **10.0**.
*   **Rationale**: Bridged the gap between exploration and balancing. A global potential energy gradient (0-20 pts) is a better "breadcrumb trail" than a localized funnel.
*   **Result**: Success in discovery, but led to "dragging" (leaning at 40 degrees to harvest the wide jackpot) and oscillations.

### v6.3: The "Precision" Refinement
*   **Change**: Tightened `r_persistence` gate (45° -> **20°**), reduced `r_swing` (10.0 -> **2.5**).
*   **Rationale**: Forced the agent into true balancing. By shrinking the gate, "dragging" becomes non-profitable. Lowering `r_swing` reduced the incentive to oscillate back and forth.
*   **Result**: Improved consistency, but high-frequency wiggling suggested the need for safety limit hardening.

### v6.4: The "Safety-First" Configuration (Current)
*   **Objective**: Prioritize hardware longevity and master the full swing-up sequence.
*   **Reset Curriculum**: Shifted from 55% Bootcamp to **70% Classic (Hanging)**.
    *   *Why*: Forces the agent to master the swing-up from a dead stop rather than over-fitting to stability starts.
*   **Safety Hardening**:
    *   Arm Velocity Limit: **40 rad/s** (was 80).
    *   Pendulum Velocity Limit: **60 rad/s** (was 100).
    *   Helicopter Limit: **360°** (was 400°).
    *   *Why*: Protects the physical motor and arm from high-speed impacts at the hard stops.
*   **Repulsive Boundary**: `out_penalty` increased to **-200.0 * x²**.
    *   *Why*: Creates a massive "repulsive force" that pushes the arm back to center long before it hits the hardware limits.

---
*Note: v6.4 is currently training. Monitoring for breakthrough in swing-up consistency.*


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
