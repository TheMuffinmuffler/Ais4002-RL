import os
import sys
import glob
import time
import argparse

# Add parent directory to sys.path so we can find control.py, QUBE.py, etc.
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy_common import deploy

def select_model(provided_path=None):
    """Search for SAC models or use the provided one."""
    if provided_path:
        if os.path.exists(provided_path):
            return provided_path
        else:
            print(f"Error: Provided path '{provided_path}' does not exist.")
            return None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    # Priority candidates
    candidates = [
        os.path.join(models_dir, "qube_sac_final.zip"),
        os.path.join(models_dir, "qube_sac_refined_final.zip"),
        os.path.join(models_dir, "best_sac/best_model.zip"),
        os.path.join(models_dir, "best_sac_refine/best_model.zip"),
    ]
    
    # Also find all SAC checkpoints
    checkpoints = glob.glob(os.path.join(models_dir, "qube_sac_checkpoint_*.zip"))
    # Sort checkpoints by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    # Build list of unique available paths
    available = []
    seen = set()
    
    for c in candidates:
        if os.path.exists(c):
            norm_c = os.path.normpath(c)
            if norm_c not in seen:
                available.append(c)
                seen.add(norm_c)
                
    for c in checkpoints:
        norm_c = os.path.normpath(c)
        if norm_c not in seen:
            available.append(c)
            seen.add(norm_c)
            
    if not available:
        print("No SAC models found in 'models/' directory.")
        return None
        
    print("\n" + "="*60)
    print(" SAC MODEL SELECTION ")
    print("="*60)
    for i, path in enumerate(available):
        mtime = os.path.getmtime(path)
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        prefix = "-> " if any(p in path for p in ["final", "best"]) else "   "
        print(f"[{i:2}] {prefix}{path:<45} ({mtime_str})")
    print("="*60)
        
    try:
        user_input = input(f"\nSelect model index [0-{len(available)-1}] (default 0): ").strip()
        if user_input == "":
            return available[0]
        idx = int(user_input)
        if 0 <= idx < len(available):
            return available[idx]
    except ValueError:
        pass
        
    return available[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SAC model to QUBE serial.")
    parser.add_argument("--path", type=str, help="Path to the model .zip file")
    args = parser.parse_args()

    print("Starting SAC Deployment Script...")
    chosen_model = select_model(args.path)
    
    if chosen_model:
        print(f"\nPreparing to deploy: {chosen_model}")
        # RGB Green for SAC
        deploy("sac", rgb=(0, 999, 0), model_path=chosen_model)
    else:
        print("Deployment aborted: No model selected.")
