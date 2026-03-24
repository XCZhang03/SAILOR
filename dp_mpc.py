from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

import pathlib
import numpy as np
import math
import os
import shutil
import contextlib

from vlm_agent import VLMAgent
from vlm_utils import *

CAMERA_NAMES = ["agentview", "birdview", "robot0_eye_in_hand", "sideview", "canonical_frontview"]
LIBERO_ENV_RESOLUTION = 224
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_names": CAMERA_NAMES}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def _get_empty_env(task, env):
    import robosuite
    dataset_file = os.path.join(get_libero_path("datasets"), f"{task.problem_folder}/{task.name}_demo.hdf5")
    import h5py
    import json
    f = h5py.File(dataset_file, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    f.close()
    empty_env_kwargs = env_meta['env_kwargs'].copy()
    empty_env_kwargs['env_name'] = "SingleArmEmptyEnv"
    empty_env_kwargs['hard_reset'] = False
    empty_env_kwargs['ignore_done'] = True
    empty_env_kwargs['has_offscreen_renderer'] = False
    empty_env_kwargs['has_renderer'] = False
    empty_env_kwargs['use_camera_obs'] = False
    empty_env_kwargs['camera_names'] = CAMERA_NAMES
    empty_env_kwargs['camera_heights'] = LIBERO_ENV_RESOLUTION
    empty_env_kwargs['camera_widths'] = LIBERO_ENV_RESOLUTION
    empty_env_kwargs['robots'] = [type(robot.robot_model).__name__ for robot in env.robots]
    empty_env = robosuite.make(**empty_env_kwargs)
    empty_env.copy_env_model(env)
    return empty_env



def run_mpc(task_id=0, seed=0):
    save_dir = os.path.join("scratch_dir/mpc_data/libero_object/test_agent_proposal", f"task{task_id}", f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    task_suite_name = "libero_object"
    task_id = task_id
    seed = seed

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    empty_env = _get_empty_env(task, env)
    print(f"Task description: {task_description}")

    # ========== ENVIRONMENT INITIALIZATION ==========
    env.reset()
    num_steps = 0
    done = False
    replay_images = []
    obs = env.set_init_state(initial_states[seed])
    for t in range(10):
        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

    # ========== LOAD POLICIES ==========
    from dp_utils import embed_lang
    subtask_description = task_description.replace(" up", "")
    print(f"Subtask description: {subtask_description}")
    subtask_embedding = embed_lang(subtask_description)
        
    from dp_utils import load_checkpoint
    import robosuite.utils.transform_utils as T
    # checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.02.27/09.11.46_train_diffusion_transformer_hybrid_libero_image/checkpoints/epoch=0014-test_mean_score=1.000.ckpt"
    checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.19/11.59.01_train_diffusion_transformer_hybrid_libero_image/checkpoints/epoch=2050-test_mean_score=1.000.ckpt"
    # checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.05/06.14.04_train_diffusion_transformer_hybrid_libero_image/checkpoints/epoch=0022-test_mean_score=1.000.ckpt"
    # checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.08/09.41.02_train_diffusion_transformer_hybrid_libero_image/checkpoints/epoch=0300-test_mean_score=1.000.ckpt"
    policy, cfg = load_checkpoint(checkpoint_path)
    policy = policy.to("cuda")
    import torch
    def to_torch(image):
        image = image_tools.resize_with_pad(image, 128, 128)
        return np.moveaxis(image[::-1], -1, 0) / 255.0
    def policy_fn(obs):
        np_obs_dict = dict(obs)
        if "lang_embed" in cfg.shape_meta.obs:
            np_obs_dict["lang_embed"] = subtask_embedding
        obs_keys = cfg.shape_meta.obs.keys()
        np_obs_dict = {k: np_obs_dict[k] for k in obs_keys}
        np_obs_dict = {k: to_torch(v) if "image" in k else v for k, v in np_obs_dict.items()}
        obs_dict = {k: torch.from_numpy(v).to("cuda").unsqueeze(0).unsqueeze(0) for k, v in np_obs_dict.items()}
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        np_action_dict = {k: v.cpu().numpy() for k, v in action_dict.items()}
        action = np_action_dict['action_pred'][0]
        return action

    # idm_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.02/07.30.54_train_diffusion_unet_lowdim_idm_libero_idm/checkpoints/epoch=0100-val_loss=0.034.ckpt"
    idm_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.17/08.56.15_train_diffusion_unet_lowdim_idm_libero_idm/checkpoints/epoch=0110-val_loss=0.019.ckpt"
    # idm_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.12/01.36.19_train_diffusion_unet_lowdim_idm_libero_idm/checkpoints/epoch=0020-val_loss=0.025.ckpt"
    idm, idm_cfg = load_checkpoint(idm_checkpoint_path)
    idm = idm.to("cuda")
    def idm_fn(obs, target_pos, target_quat=None):
        np_obs_dict = dict(obs)
        obs_keys = idm_cfg.shape_meta.obs.keys()
        np_obs_dict = {k: np_obs_dict[k] for k in obs_keys}
        delta_obs_dict = {"robot0_eef_pos": target_pos - obs['robot0_eef_pos']}
        if target_quat is not None:
            delta_obs_dict['robot0_eef_quat'] = T.quat_distance(target_quat, obs['robot0_eef_quat'])
        obs_dict = {k: torch.from_numpy(v).to("cuda").unsqueeze(0) for k, v in np_obs_dict.items()}
        delta_obs_dict = {k: torch.from_numpy(v).to("cuda").unsqueeze(0) for k, v in delta_obs_dict.items()}
        with torch.no_grad():
            action_dict = idm.predict_action(obs_dict, delta_obs_dict)
        np_pred_action = action_dict['action_pred'].cpu().numpy()[0]
        return np_pred_action

    # idm_2_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.02/07.30.54_train_diffusion_unet_lowdim_idm_libero_idm/checkpoints/epoch=0100-val_loss=0.034.ckpt"
    idm_2_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.11/21.52.52_train_diffusion_unet_lowdim_idm_libero_idm/checkpoints/epoch=0040-val_loss=0.026.ckpt"
    idm_2, idm_2_cfg = load_checkpoint(idm_2_checkpoint_path)
    idm_2 = idm_2.to("cuda")
    def idm_fn_2(obs, target_pos, target_quat=None):
        np_obs_dict = dict(obs)
        obs_keys = idm_2_cfg.shape_meta.obs.keys()
        np_obs_dict = {k: np_obs_dict[k] for k in obs_keys}
        delta_obs_dict = {"robot0_eef_pos": target_pos - obs['robot0_eef_pos']}
        if target_quat is not None:
            delta_obs_dict['robot0_eef_quat'] = T.quat_distance(target_quat, obs['robot0_eef_quat'])
        obs_dict = {k: torch.from_numpy(v).to("cuda").unsqueeze(0) for k, v in np_obs_dict.items()}
        delta_obs_dict = {k: torch.from_numpy(v).to("cuda").unsqueeze(0) for k, v in delta_obs_dict.items()}
        with torch.no_grad():
            action_dict = idm_2.predict_action(obs_dict, delta_obs_dict)
        np_pred_action = action_dict['action_pred'].cpu().numpy()[0]
        return np_pred_action
    

    # ========== EXECUTION PIPELINE ==========
    # Redirect all prints to log file
    log_file_path = os.path.join(save_dir, f"execution_log_task{task_id}_seed{seed}.txt")
    with open(log_file_path, 'w') as log_file:
        with contextlib.redirect_stdout(log_file):
            # Phase 1: Execute policy for subtask 0
            import tqdm
            print("Phase 1: Executing policy")
            pbar = tqdm.tqdm(total=500, desc="Executing policy ")
            while not done and num_steps < 400:
                action_chunk = policy_fn(obs)[:10]
                for action in action_chunk:
                    obs, reward, done, info = env.step(action)
                    replay_images.append(obs["agentview_image"][::-1])
                pbar.update(10)
                num_steps += 10
        
            pbar.close()
            print(f"Execution completed. Total steps: {num_steps}, Done: {done}")

    import imageio
    imageio.mimwrite(os.path.join(save_dir, f"replay_task{task_id}_seed{seed}.mp4"), replay_images, fps=20)
    
    print(f"Execution log saved to: {log_file_path}")
    print(f"Replay saved to: {os.path.join(save_dir, f'replay_task{task_id}_seed{seed}.mp4')}")

    print(f"Success {done}")
    return done


if __name__ == "__main__":
    for task in range(2,10):
        num_success = 0
        for seed in range(10):
            success = run_mpc(task_id=task, seed=seed)
            if success:
                num_success += 1
            print(f"Number of successful runs: {num_success}/{seed+1}")
        with open(os.path.join("scratch_dir/mpc_data/libero_object/test_agent_proposal", f"task{task}", "result.txt"), "w") as f:
            f.write(f"Success rate: {num_success}/10\n")