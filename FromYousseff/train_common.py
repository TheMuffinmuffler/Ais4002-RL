import argparse
import os
import torch

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from compat import apply_compat_shims
from config import CHECKPOINT_FREQ, EVAL_FREQ, N_ENVS
from qube_env import QubeEnv

apply_compat_shims()

# Define the base directory for FromYousseff relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    # MPS can be unstable/slower for SB3 on some Macs. CPU is safer.
    return "cpu"


def parse_args(default_steps, default_envs=N_ENVS):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--n-envs", type=int, default=default_envs)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing final model and start from zero.")
    parser.add_argument("--no-randomization", action="store_true", help="Disable domain randomization.")
    return parser.parse_args()


def make_training_env(n_envs, domain_randomization=True):
    return make_vec_env(lambda: QubeEnv(domain_randomization=domain_randomization), n_envs=n_envs)


def make_eval_env():
    return Monitor(QubeEnv(domain_randomization=False))


def setup_dirs(algo_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    algo_log_dir = os.path.join(LOGS_DIR, algo_name)
    os.makedirs(algo_log_dir, exist_ok=True)
    return configure(algo_log_dir, ["stdout", "csv", "tensorboard"])


def callbacks(algo_name, n_envs):
    checkpoint = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // max(n_envs, 1), 1),
        save_path=MODELS_DIR,
        name_prefix=f"qube_{algo_name}_checkpoint",
    )
    eval_callback = EvalCallback(
        make_eval_env(),
        best_model_save_path=os.path.join(MODELS_DIR, f"best_{algo_name}"),
        log_path=os.path.join(LOGS_DIR, f"{algo_name}_eval"),
        eval_freq=max(EVAL_FREQ // max(n_envs, 1), 1),
        deterministic=True,
        render=False,
    )
    return [checkpoint, eval_callback]
