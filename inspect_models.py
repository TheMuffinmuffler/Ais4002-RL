from stable_baselines3 import SAC
import os

model_paths = [
    "models/qube_sac_final.zip",
    "FromYousseff/models/qube_sac_final.zip"
]

for path in model_paths:
    if os.path.exists(path):
        print(f"Inspecting {path}...")
        try:
            model = SAC.load(path, device="cpu")
            print(f"  Observation Space: {model.observation_space}")
            print(f"  Action Space: {model.action_space}")
        except Exception as e:
            print(f"  Error loading {path}: {e}")
    else:
        print(f"{path} does not exist.")
