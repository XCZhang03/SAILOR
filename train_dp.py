import argparse
import collections
import contextlib
import gc
import os
import pathlib
import sys
from datetime import datetime
sys.path.append(
    os.path.join(os.getcwd(), "sailor/diffusion")
)  # For diffusion4robotics imports

import numpy as np
from omegaconf import OmegaConf
import torch
from termcolor import cprint

import environments.wrappers as wrappers
import sailor.dreamer.tools as tools
from environments.concurrent_envs import ConcurrentEnvs
from environments.global_utils import save_demo_videos
from sailor.classes.preprocess import Preprocessor
from sailor.classes.resnet_encoder import ResNetEncoder, VQResNetEncoder
from sailor.policies.diffusion_base_policy import DiffusionBasePolicy

from utils import create_datasets_and_envs, get_config, init_dp

# Force EGL rendering in environments
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def train_eval(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()

    # ==================== Logging ====================
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, logdir / "config.yaml")

    config.logdir = logdir
    log_step = 0
    config.scratch_dir = pathlib.Path(config.scratch_dir).expanduser()
    logger = tools.Logger(config) if config.use_wandb else None

    print("---------------------")
    cprint(f"Task: {config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {config.logdir}", "cyan", attrs=["bold"])
    cprint(
        f"Time Limit: {config.time_limit} | Max Env Steps: {config.train_dp_mppi_params['n_env_steps']}",
        "cyan",
        attrs=["bold"],
    )
    if config.visualize_eval:
        cprint(
            f"WARNING: Saving videos of evaluation episodes, please turn off if not needed. High resolution render is {config.high_res_render}",
            "red",
            attrs=["bold"],
        )
    print("---------------------")

    expert_eps, expert_val_eps, envs = create_datasets_and_envs(
        config
    )


    # ============ Diffusion Policy Pretraining ===============
    # If checkpoint is not provided, train the DP
    cprint(
        "----------------No base policy path provided, begin training diffusion policy--------------",
        "yellow",
        attrs=["bold"],
    )
    # Initialize DP
    base_policy = init_dp(logger=logger, config=config)
    # Train it
    expert_dataset_dp = tools.make_dataset(
        expert_eps, batch_length=1, batch_size=config.dp["batch_size"]
    )
    log_step = base_policy.train_base_policy(
        train_dataset=expert_dataset_dp,
        expert_val_eps=expert_val_eps,
        eval_envs=envs,
        log_prefix="dp_pretrain",
    )

    # Store the saved pretrained checkpoint path
    config.dp["pretrained_ckpt"] = os.path.abspath(base_policy.ckpt_file)

    # Cleanup
    OmegaConf.save(config, logdir / "config.yaml")
    del base_policy
    torch.cuda.empty_cache()
    gc.collect()
    envs.close()


if __name__ == "__main__":
    final_config = get_config()
    train_eval(final_config)
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        gc.collect()  # Force garbage collection to run
