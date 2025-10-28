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
from termcolor import cprint

import environments.wrappers as wrappers
from environments.concurrent_envs import ConcurrentEnvs
from environments.global_utils import save_demo_videos
from sailor.classes.preprocess import Preprocessor
from sailor.classes.resnet_encoder import ResNetEncoder, VQResNetEncoder
from sailor.policies.diffusion_base_policy import DiffusionBasePolicy

# Force EGL rendering in environments
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def create_datasets_and_envs(config):
    # ==================== Create Datasets ====================
    # Check if task can be split by __ else raise error
    if "__" not in config.task:
        raise ValueError(f"Task {config.task} must be of form 'env_suite__task'")

    suite, task = config.task.split("__", 1)
    task = task.lower()
    if suite == "robomimic":
        from environments.robomimic.utils import get_train_val_datasets

        expert_eps, expert_val_eps, _, state_dim, action_dim = get_train_val_datasets(
            config
        )
    elif suite == "robocasa":
        from environments.robocasa.utils import get_train_val_datasets

        expert_eps, expert_val_eps, _, state_dim, action_dim = get_train_val_datasets(
            config
        )
    elif suite == "maniskill":
        from environments.maniskill.utils import \
            get_train_val_datasets_maniskill

        expert_eps, expert_val_eps, _, state_dim, action_dim = (
            get_train_val_datasets_maniskill(config)
        )
    elif "libero" in suite:
        from environments.libero.utils import get_train_val_datasets

        expert_eps, expert_val_eps, _, state_dim, action_dim = get_train_val_datasets(
            config
        )
    else:
        raise ValueError(f"Unknown env suite {suite}")
    # Set correct values of state_dim and action_dim
    config.state_dim = int(state_dim)
    config.action_dim = int(action_dim)
    cprint(f"Enviroment State Dim: {state_dim}, Action Dim: {action_dim}", "cyan")

    if config.viz_expert_buffer:
        cprint(
            "-----------------Inspecting Expert Dataset, Saving Videos--------------",
            "yellow",
            attrs=["bold"],
        )
        for id, key in enumerate(expert_eps.keys()):
            frame_successes = np.array(expert_eps[key]["success"]).ravel()
            agent_frames = np.array(expert_eps[key]["agentview_image"])[..., -1]
            robot_frames = np.array(expert_eps[key]["robot0_eye_in_hand_image"])[
                ..., -1
            ]
            save_demo_videos(
                suite=suite,
                task=task,
                id=id,
                frame_successes=frame_successes,
                agent_frames=agent_frames,
                robot_frames=robot_frames,
            )
        exit()

    # ==================== Create Envs ====================
    if suite in ["robomimic", "robocasa"] or "libero" in suite:
        envs = ConcurrentEnvs(
            config=config, env_make=make_env, num_envs=config.num_envs
        )
    elif suite == "maniskill":
        if config.use_cpu_env:
            envs = ConcurrentEnvs(
                config=config, env_make=make_env, num_envs=config.num_envs
            )
        else:
            envs = make_env(config)

    acts = envs.action_space
    print(f"Action Space: {acts}. Low: {acts.low}. High: {acts.high}")
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    return expert_eps, expert_val_eps, envs

def init_dp(config, state_dim=None, action_dim=None, **kwargs):
    # Initialize DP
    preprocessor = Preprocessor(config=config)
    if config.state_only:
        encoder = None
    elif config.dp.get("quantize_image_features", False):
        encoder = VQResNetEncoder(num_cams=config.dp.num_cams)
    else:
        encoder = ResNetEncoder(num_cams=config.dp.num_cams)
    base_policy = DiffusionBasePolicy(
        preprocessor=preprocessor,
        encoder=encoder,
        config=config,
        device=config.device,
        state_dim=state_dim if state_dim is not None else config.get("state_dim", None),
        action_dim=action_dim if action_dim is not None else config.get("action_dim", None),
        name="DP_Pretrain",
        logger=kwargs.pop("logger", None),
        **kwargs
    )
    if config.dp.pretrained_ckpt != "":
        cprint(
            f"Loading pretrained diffusion policy from {config.dp.pretrained_ckpt}",
            "yellow",
            attrs=["bold"],
        )
        base_policy.trainer.load_checkpoint(
            config.dp.pretrained_ckpt
        )
    else:
        cprint(
            "No pretrained diffusion policy checkpoint provided.",
            "yellow",
            attrs=["bold"],
        )
    return base_policy


def make_env(config):
    suite, task = config.task.split("__", 1)
    task = task.lower()
    if suite == "robomimic":
        from environments.robomimic.constants import IMAGE_OBS_KEYS
        from environments.robomimic.env_make import make_env_robomimic
        from environments.robomimic.utils import (
            create_shape_meta, get_robomimic_dataset_path_and_env_meta)

        dataset_path, env_meta = get_robomimic_dataset_path_and_env_meta(
            env_id=task,
            shaped=config.shape_rewards,
            image_size=config.image_size,
            done_mode=config.done_mode,
            datadir=config.datadir,
        )
        shape_meta = create_shape_meta(img_size=config.image_size, include_state=True)

        shape_rewards = config.shape_rewards
        env = make_env_robomimic(
            env_meta,
            IMAGE_OBS_KEYS,
            shape_meta,
            add_state=True,
            reward_shaping=shape_rewards,
            config=config,
            offscreen_render=True,
        )
        env = wrappers.TimeLimit(env, duration=config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)

    elif suite == "robocasa":
        from environments.robocasa.utils import make_env_robocasa

        env = make_env_robocasa(
            config=config,
            task=task,
            suite=suite,
        )
        env = wrappers.TimeLimit(env, duration=config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)

    elif suite == "maniskill":
        from environments.maniskill.utils import make_maniskill_env

        env = make_maniskill_env(config, suite=suite, task=task)
        env = wrappers.UUID(env)

    elif "libero" in suite:
        from environments.libero.utils import make_env_libero

        env = make_env_libero(
            config=config,
            suite=suite,
            task=task,
        )
        env = wrappers.TimeLimit(env, duration=config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)

    else:
        raise ValueError(f"Unknown env suite {suite}")

    return env


def get_config(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=['robomimic', 'debug'])
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--resume_run", type=bool, default=False)
    args, remaining = parser.parse_known_args()

    configs = OmegaConf.load(
        (pathlib.Path(sys.argv[0]).parent / "sailor/configs.yaml")
    )

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]

    defaults = OmegaConf.merge(*[configs.get(name) for name in name_list])

    final_config = OmegaConf.merge(defaults, OmegaConf.from_cli(remaining))

    # Set mppi["horizon"] = pred_horizon and dp["ac_chunk"] = pred_horizon
    final_config.mppi.horizon = final_config.pred_horizon
    final_config.dp.ac_chunk = final_config.pred_horizon

    # Set Wandb Stuff
    exp_name = f"{str(final_config.task).lower()}/{args.exp_name}_demos{final_config.num_exp_trajs}"
    final_config.wandb_exp_name = f"{exp_name}"

    # Set time limit
    suite, task = final_config.task.split("__", 1)
    task = task.lower()

    if suite == "robomimic":
        final_config.time_limit = final_config.env_time_limits[task]

    elif suite == "robocasa":
        final_config.time_limit = final_config.env_time_limits[task]

    elif suite == "maniskill":
        final_config.time_limit = final_config.env_time_limits[task]
    
    elif "libero" in suite:
        final_config.time_limit = 300

    else:
        raise ValueError(f"Unknown env suite {suite}")

    # Set log dir and datadir
    final_config.logdir = (
        f"{final_config.scratch_dir}/logs/{exp_name}/seed{final_config.seed}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    final_config.datadir = os.path.join("datasets", f"{suite}_datasets")

    if final_config.generate_highres_eval:
        final_config.high_res_render = True
    
    return final_config

if __name__ == "__main__":
    final_config = get_config()
    train_eps, val_eps, envs, final_config = create_datasets_and_envs(
        final_config
    )
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        gc.collect()  # Force garbage collection to run
