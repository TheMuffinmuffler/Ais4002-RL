"""Train all three algorithms.

Default order: SAC first, then TD3, then PPO.  This is intentional: SAC is the
only one worth trusting as the main controller if time is limited.
"""

import argparse
import subprocess
import sys


def run(cmd):
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--sac-steps", type=int, default=350_000)
    parser.add_argument("--td3-steps", type=int, default=350_000)
    parser.add_argument("--ppo-steps", type=int, default=600_000)
    args = parser.parse_args()

    fresh = ["--fresh"] if args.fresh else []
    run([sys.executable, "train_rl_SAC.py", "--steps", str(args.sac_steps), *fresh])
    run([sys.executable, "train_rl_TD3.py", "--steps", str(args.td3_steps), *fresh])
    run([sys.executable, "train_rl.py", "--steps", str(args.ppo_steps), *fresh])


if __name__ == "__main__":
    main()
