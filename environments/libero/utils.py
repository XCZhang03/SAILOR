import collections
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robosuite.utils.transform_utils as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from environments.robomimic.utils import add_traj_to_cache, create_shape_meta
from sailor.classes.rollout_utils import get_act_stacked, get_obs_stacked
from sailor.dreamer.tools import add_to_cache

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.utils.utils as libero_utils
from environments.robomimic.robosuite_pose_wrapper import RobosuitePoseWrapper

def parse_task_indices(task):
        """
        Parse task into a list of integer task indices.
        Supports:
          - a single integer or string like "3"
          - a comma-separated list like "1,2,5"
          - ranges with '-' like "2-5" (inclusive) or "5-2" (descending)
          - a mix like "1,3-5,7"
          - a list/tuple of ints
        Returns a deduplicated list preserving the first-seen order.
        """
        tasks = []
        if isinstance(task, (list, tuple)):
            tasks = [int(t) for t in task]
        else:
            task_str = str(task).strip()
            if ',' in task_str or '-' in task_str:
                for part in [p.strip() for p in task_str.split(',') if p.strip()]:
                    if '-' in part:
                        start_s, end_s = [x.strip() for x in part.split('-', 1)]
                        start, end = int(start_s), int(end_s)
                        if start <= end:
                            tasks.extend(range(start, end + 1))
                        else:
                            tasks.extend(range(start, end - 1, -1))
                    else:
                        tasks.append(int(part))
            else:
                tasks = [int(task_str)]

        # Deduplicate while preserving order
        seen = set()
        out = []
        for x in tasks:
            if x not in seen:
                seen.add(x)
                out.append(x)

        if not out:
            raise ValueError(f"Could not parse task: {task!r}")

        return out

def get_env_details(
        config,
        suite,
        task,
):
    """
    Returns the path to the Libero Robomimic dataset and environment metadata.

    Args:
        benchmark_name (str, optional): The name of the benchmark. Defaults to "libero_90".
        task_index (int, optional): The index of the task in the benchmark. Defaults to 0.
        image_size (int, optional): The size of the images in the dataset. Defaults to 128.

    Returns:
        tuple: A tuple containing the dataset path and environment metadata.
    """
    image_size = config.image_size
    benchmark_name = suite
    task_index = int(task)
    assert image_size == 128, "Currently only support image_size 128"
    benchmark = get_benchmark(benchmark_name)()
    task = benchmark.get_task(task_index)
    bddl_file_name = benchmark.get_task_bddl_file_path(task_index)
    demonstration_path = benchmark.get_task_demonstration(task_index)
    demonstration_path = os.path.join(get_libero_path("datasets"), demonstration_path)
    
    env_meta = FileUtils.get_env_metadata_from_dataset(demonstration_path)
    env_meta['bddl_file_name'] = bddl_file_name

    shape_meta = create_shape_meta(image_size, include_state=True)
    return demonstration_path, env_meta, shape_meta

def make_env_libero(
        config,
        suite,
        task,
):
    assert "libero" in suite, f"Only libero is supported, but got {suite}"

    tasks = parse_task_indices(task)

    _, env_meta, shape_meta = get_env_details(
        config,
        suite,
        tasks[0],
    )
    if config.high_res_render:
        camera_shape = config.highres_img_size
    else:
        camera_shape = config.image_size


    env_kwargs = {
        "bddl_file_name": env_meta['bddl_file_name'],
        "camera_heights": camera_shape,
        "camera_widths": camera_shape,
        "camera_segmentations": None,
        "ignore_done": False,
        "hard_reset": False,
    }
    env = OffScreenRenderEnv(**env_kwargs).env

    empty_env_kwargs = env_meta['env_kwargs'].copy()
    empty_env_kwargs['env_name'] = "SingleArmEmptyEnv"
    empty_env_kwargs['hard_reset'] = False
    empty_env_kwargs['has_offscreen_renderer'] = False
    empty_env_kwargs['has_renderer'] = False
    empty_env_kwargs['use_camera_obs'] = False
    empty_env_kwargs['robots'] = [type(robot.robot_model).__name__ for robot in env.robots]

    wrapped_env = RobosuitePoseWrapper(
        empty_env_kwargs,
        env,
        config=config,
        shape_meta=shape_meta,
    )

    return wrapped_env

    
def update_demo_keys(demo) -> Dict[str, Any]:
    KEYS_MAP = {
        "agentview_rgb": "agentview_image",
        "eye_in_hand_rgb": "robot0_eye_in_hand_image",
        "joint_states": "robot0_joint_pos",
        "ee_pos": "robot0_eef_pos",
        "ee_ori": "robot0_eef_ori",
        "gripper_states": "robot0_gripper_qpos",
    }
    new_demo = {}

    new_demo['obs'] = {}
    obs = demo['obs']
    new_obs = new_demo['obs']
    for key in obs.keys():
        if key in KEYS_MAP.values():
            continue
        if key in KEYS_MAP:
            new_key = KEYS_MAP[key]
            new_obs[new_key] = np.array(obs[key])
    new_obs['robot0_eef_quat'] = [T.axisangle2quat(ori) for ori in new_obs['robot0_eef_ori']]
    new_obs['robot0_eef_quat'] = np.array(new_obs['robot0_eef_quat'], dtype=np.float32)
    
    new_demo['rewards'] = np.array(demo['rewards'])
    new_demo['actions'] = np.array(demo['actions'])
    new_demo['dones'] = np.array(demo['dones'])

    return new_demo


def get_train_val_datasets(config):
    num_train_trajs = config.num_exp_trajs
    num_val_trajs = config.num_exp_val_trajs

    suite, task = config.task.split("__", 1)
    tasks = parse_task_indices(task)
    assert "libero" in suite

    train_eps = collections.OrderedDict()
    val_eps = collections.OrderedDict()

    # Load the h5py files
    dataset_paths = []
    shape_meta = None
    for task in tasks:
        dataset_path, _, cur_shape_meta = get_env_details(
            config,
            suite,
            task,
        )
        dataset_paths.append(dataset_path)
        if shape_meta is None:
            shape_meta = cur_shape_meta
        else:
            assert shape_meta == cur_shape_meta, "Shape meta mismatch across tasks"

    new_data_dict = {"data": {}}
    clean_demos = []
    num_demos_per_file = max((num_train_trajs + num_val_trajs) // len(dataset_paths) + 1, 1)
    for dataset_path in dataset_paths:
        if len(clean_demos) >= (num_train_trajs + num_val_trajs):
            break
        print(f"Loading dataset from {dataset_path}", flush=True)

        f = h5py.File(dataset_path, "r")
        demos = list(f["data"].keys())

        # Assert that we have enough data
        assert num_demos_per_file <= len(demos), "Not enough expert data"

        ii = 0
        while ii < num_demos_per_file and len(clean_demos) < (num_train_trajs + num_val_trajs):
            demo = f["data"][demos[ii]]
            demo_updated = update_demo_keys(demo)
            new_demo_name = f"{Path(dataset_path).stem}_{demos[ii]}"
            new_data_dict["data"][new_demo_name] = demo_updated
            clean_demos.append(new_demo_name)
            ii += 1
        f.close()
    np.random.shuffle(clean_demos)
    
    obs_keys = shape_meta["obs"].keys()
    pixel_keys = sorted([key for key in obs_keys if "image" in key])
    state_keys = sorted([key for key in obs_keys if "image" not in key])

    # Initialize norm_dict
    # Read ob_dim and ac_dim from the first datapoint in the first demo
    first_demo = new_data_dict["data"][clean_demos[0]]
    ob_dim = 0
    for key in state_keys:
        ob_dim += np.prod(first_demo["obs"][key].shape[1:])
    ac_dim = first_demo["actions"].shape[1]

    print(f"Initizalizing norm_dict with ob_dim={ob_dim} and ac_dim={ac_dim}")
    norm_dict = {
        "ob_max": -np.inf * np.ones(ob_dim, dtype=np.float32),
        "ob_min": np.inf * np.ones(ob_dim, dtype=np.float32),
        "ac_max": -np.inf * np.ones(ac_dim, dtype=np.float32),
        "ac_min": np.inf * np.ones(ac_dim, dtype=np.float32),
    }

    # Set state_dim and action_dim
    state_dim = 0
    for key in state_keys:
        state_dim += np.prod(first_demo["obs"][key].shape[1:])

    action_dim = first_demo["actions"].shape[1]

    # Fill the Train Dataset
    for ii in range(num_train_trajs):
        demo = clean_demos[ii]
        add_traj_to_cache(
            ii,
            demo,
            train_eps,
            new_data_dict,
            config,
            pixel_keys,
            state_keys,
            norm_dict,
        )

    # Compute average length in data in train_eps
    lengths = [len(ep["state"]) for ep in train_eps.values()]
    print(
        "Min length:",
        min(lengths),
        "Max length:",
        max(lengths),
        "Mean length:",
        np.mean(lengths),
    )
    print("Loaded", len(train_eps.keys()), "training episodes")

    # Fill the Val Dataset
    for ii in range(num_train_trajs, num_train_trajs + num_val_trajs):
        demo = clean_demos[ii]
        add_traj_to_cache(
            ii, demo, val_eps, new_data_dict, config, pixel_keys, state_keys
        )
    print("Loaded", len(val_eps.keys()), "validation episodes")

    return train_eps, val_eps, norm_dict, state_dim, action_dim