# Implementation Plan - Fixing the "Lazy Agent" Reward Local Minimum

We verified that the agent is currently stuck in a local minimum where it just hangs at the bottom. This implementation plan corrects the reward function in the environment to incentivize swinging up and balancing.

---

## User Review Required

Please review and confirm these simple adjustments before we update the code and restart training.

> [!NOTE]
> **The Problem:** 
> Currently, hanging still at the bottom gives a guaranteed **`+5.0` reward** per step, while swinging up risks a terrifying **`-2,500` crash penalty**. The agent has learned that doing nothing is the safest way to get points.

**Proposed Solution:**
1. **Remove `r_survival` (+5.0):** Hanging still at the bottom will now give **`0.0` points** instead of positive points.
2. **Lower early termination penalty:** Reduce the crash penalty from **`-2,500` to `-200`** so the agent is not afraid to swing the arm fast and explore.
3. **Keep all balance and swing rewards the same:** The balance jackpot and swing-up potential remain fully intact.

---

## Proposed Changes

### [Component Name] FromYousseff Environment

#### [MODIFY] [qube_env.py](file:///E:/Ais4002-RL/FromYousseff/qube_env.py)

We will update two sections in the reward calculation inside `FromYousseff/qube_env.py`:

1. **Remove the survival carrot from the reward sum:**
```diff
-        # 9. Survival Bonus (The Carrot)
-        # Prevents "suicide bug" by making every living step profitable
-        r_survival = 5.0
-        
-        reward = (r_balance + r_swing + r_persistence + r_survival + departure_tax 
-                  - p_center - p_effort - p_smooth - p_boundary)
+        # 9. Survival Bonus (Removed to prevent lazy bottom-hanging)
+        r_survival = 0.0
+        
+        reward = (r_balance + r_swing + r_persistence + r_survival + departure_tax 
-                  - p_center - p_effort - p_smooth - p_boundary)
```

2. **Lower the early termination penalty:**
```diff
-        # Suicide Prevention: If we terminate, we lose the "potential" 2500 points from survival
-        # So we subtract a penalty that offsets the greed for a clean slate.
-        if terminated:
-            reward -= 2500.0
+        # Reduced penalty to encourage high-speed exploration without excessive fear
+        if terminated:
+            reward -= 200.0
```

---

## Verification Plan

### Automated Tests
1. **Diagnostic Check:** Run `FromYousseff/diag_crash.py` to ensure the modified code has no syntax or type errors.
2. **Fresh Retrain:** Kill the current stuck training job and launch a new fresh SAC training run:
   ```powershell
   python FromYousseff/train_rl_SAC.py --fresh
   ```
3. **Reward Curve Monitoring:** Verify that the rollout mean reward grows beyond `2,000` (which is the ceiling for hanging at the bottom) and reaches the expected balanced target range (`>20,000+` cumulative reward).
