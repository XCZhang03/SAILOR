from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

import contextlib
import math
import os
import pathlib
import shutil

import imageio
import numpy as np
import robosuite.utils.transform_utils as T
import torch
import tqdm

from dp_utils import embed_lang, load_checkpoint
from vlm_agent import VLMAgent
from vlm_utils import *
from wm_client.wm_env import WMEnv

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
    """Create a headless 'empty' environment that shares the physics model with *env*.

    Used for camera-info queries and world-model simulation without full
    rendering overhead.
    """
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



def run_mpc(task_id=0, seed=0, wm_client=None):
    """Run one episode of VLM-guided MPC with candidate-search optimisation.

    Phases:
        1. Execute diffusion policy for subtask 0 until the handoff condition.
        2. Query VLM agent for a goal point and refine via WM-simulated search
           (midpoint → height → endpoint → local candidates).
        3. Resume diffusion policy for subtask 1 to completion.

    Returns:
        bool: True if the task was completed successfully.
    """
    save_dir = os.path.join("scratch_dir/mpc_data/libero_object/test_agent_search", f"task{task_id}", f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    task_suite_name = "libero_object"
    agent = VLMAgent(task_suite_name, task_id)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    empty_env = _get_empty_env(task, env)
    print(f"Task description: {task_description}")

    # ========== ENVIRONMENT INITIALIZATION ==========
    wm_env = WMEnv(env, empty_env, wm_client)

    wm_env.reset()
    num_steps = 0
    done = False
    replay_images = []
    obs = env.set_init_state(initial_states[seed])
    for t in range(60):
        obs, reward, done, info = wm_env.step(LIBERO_DUMMY_ACTION)
    agent.start_episode(obs)

    # ========== LOAD POLICIES ==========

    # -- Subtask language embeddings --
    subtask_description = task_description.replace(" up", "")
    print(f"Subtask description: {subtask_description}")
    subtask_embedding = embed_lang(subtask_description)
    
    # -- Diffusion policy --
    checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.14/08.49.11_train_diffusion_transformer_hybrid_libero_image/checkpoints/epoch=0460-test_mean_score=0.100.ckpt"
    policy, cfg = load_checkpoint(checkpoint_path)
    policy = policy.to("cuda")

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

    # -- Inverse dynamics models (IDM) --
    idm_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.17/08.56.15_train_diffusion_unet_lowdim_idm_libero_idm/checkpoints/epoch=0110-val_loss=0.019.ckpt"
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
    idm_2_checkpoint_path = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy/data/outputs/2026.03.17/23.50.14_train_diffusion_unet_lowdim_idm_libero_idm/checkpoints/epoch=0060-val_loss=0.020.ckpt"
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
            # Phase 1: First MPC
            print("Phase 1: Executing agent MPC...")

            # position for picking up the object
            agent.start_mpc(obs)
            agent_actions = agent.get_action_proposal()
            gripper_action = -1
            for action_dict in agent_actions:
                if action_dict["action"] == "MOVE":
                    plot_coordinates_on_image(obs, action_dict['parameters'], os.path.join(save_dir, f"pick_proposal.png"))
                    target_point = generate_3d_point(action_dict['parameters'], empty_env.get_camera_info())
                    target_quat = [1, 0, 0, 0]
                    action_chunk = idm_fn(obs, target_point, target_quat)
                    action_chunk = update_gripper_action(action_chunk, gripper_action)
                    break
            # plot_coordinates_on_image(obs, agent_actions, os.path.join(save_dir, f"pick_proposal.png"))
            # target_point = generate_3d_point(agent_actions, empty_env.get_camera_info())
            # target_quat = [1,0,0,0]
            # action_chunk = idm_fn(obs, target_point)
            # action_chunk = update_gripper_action(action_chunk, gripper_action)
            
            # position for placing the object
            final_position = agent.place_proposal()
            final_target_point = generate_3d_point(final_position, empty_env.get_camera_info())
            plot_coordinates_on_image(obs, final_position, os.path.join(save_dir, f"place_proposal.png"))

            # Phase 2: Optimise trajectory to grasp
            for action in action_chunk[:20]:
                obs, reward, done, info = wm_env.step(action)
                replay_images.append(obs["agentview_image"][::-1])
                num_steps += 1
            agent.cache_obs(obs)
            action_chunk = update_gripper_action(idm_fn_2(obs, target_point, target_quat), gripper_action)

            # -- Step 1: Optimise endpoint position --
            with wm_env.simulation():
                pred_obs = wm_env.simulate(action_chunk)
                wm_agent_obs = pred_obs['future_obs']
            imageio.mimwrite(os.path.join(save_dir, 'test_dp_output_wm_1.mp4'), pred_obs['WMPredictionOutput'].full_video, fps=20)
            endpoint_response = agent.optimize_endpoint(wm_agent_obs)
            target_point += optimize_endpoint(endpoint_response, scale=0.05)

            # target_point += np.array([0, 0, 0.03])  # lift up for better clearance

            # -- Step 2: Local candidate search --
            candidate_obs = []
            candidate_actions = []
            candidate_points = generate_candidates(target_point, scale=0.05)
            for i, candidate_point in enumerate(candidate_points):
                with wm_env.simulation():
                    candidate_action = update_gripper_action(idm_fn_2(obs, candidate_point, target_quat), gripper_action)
                    candidate_actions.append(candidate_action)
                    next_obs = wm_env.simulate(candidate_action)
                    # next_action_chunk = policy_fn(next_obs['future_obs'][-1], subtask_id=1)[:20]
                    # next_obs = wm_env.simulate(next_action_chunk)
                imageio.mimwrite(os.path.join(save_dir, f'dp_position_round0_candidate{i}.mp4'), next_obs['WMPredictionOutput'].full_video, fps=20)
                candidate_obs.append(next_obs['future_obs'])
            wristview_ranking = agent.rank_images_wristview(candidate_obs)
            # frontview_ranking = agent.rank_images_frontview(candidate_obs)
            action_chunk = candidate_actions[wristview_ranking[0]]
            target_point = candidate_points[wristview_ranking[0]]

            # # # second round of local candidate search
            # candidate_obs = []
            # candidate_actions = []
            # candidate_points = generate_candidates(target_point, scale=0.04)
            # for i, candidate_point in enumerate(candidate_points):
            #     with wm_env.simulation():
            #         candidate_action = update_gripper_action(idm_fn_2(obs, candidate_point, target_quat), gripper_action)
            #         candidate_actions.append(candidate_action)
            #         next_obs = wm_env.simulate(candidate_action)
            #         # next_action_chunk = policy_fn(next_obs['future_obs'][-1], subtask_id=1)[:20]
            #         # next_obs = wm_env.simulate(next_action_chunk)
            #     imageio.mimwrite(os.path.join(save_dir, f'dp_position_round1_candidate{i}.mp4'), next_obs['WMPredictionOutput'].full_video, fps=20)
            #     candidate_obs.append(next_obs['future_obs'])
            # wristview_ranking = agent.rank_images_wristview(candidate_obs)
            # # frontview_ranking = agent.rank_images_frontview(candidate_obs)
            # action_chunk = candidate_actions[wristview_ranking[0]]
            # target_point = candidate_points[wristview_ranking[0]]

            # -- Execute optimised trajectory --
            for action in action_chunk:
                obs, reward, done, info = wm_env.step(action)
                replay_images.append(obs["agentview_image"][::-1])
            num_steps += len(action_chunk)

            # -- Execute policy for grasp -- 
            for _ in range(8):
                action_chunk = policy_fn(obs)[:10]
                for action in (action_chunk):
                    obs, reward, done, info = wm_env.step(action)
                    replay_images.append(obs["agentview_image"][::-1])
                num_steps += 10

            # Phase 2: Place the object after grasp
            print("Phase 4: Placing the object with agent MPC...")
            gripper_action = 1
            action_chunk = update_gripper_action(idm_fn(obs, final_target_point), gripper_action)
            for action in action_chunk[:20]:
                obs, reward, done, info = wm_env.step(action)
                replay_images.append(obs["agentview_image"][::-1])
                num_steps += 1
            action_chunk = update_gripper_action(idm_fn_2(obs, final_target_point), gripper_action)
            with wm_env.simulation():
                pred_obs = wm_env.simulate(action_chunk)
                wm_agent_obs = pred_obs['future_obs']
            imageio.mimwrite(os.path.join(save_dir, 'test_dp_output_wm_2.mp4'), pred_obs['WMPredictionOutput'].full_video, fps=20)
            placement_response = agent.optimize_placement(wm_agent_obs)
            final_target_point += optimize_endpoint(placement_response, scale=0.05)
            action_chunk = update_gripper_action(idm_fn_2(obs, final_target_point), gripper_action)

            # -- Step 2: Local candidate search --
            candidate_obs = []
            candidate_actions = []
            candidate_points = generate_candidates(final_target_point, scale=0.05)
            for i, candidate_point in enumerate(candidate_points):
                with wm_env.simulation():
                    candidate_action = update_gripper_action(idm_fn_2(obs, candidate_point, target_quat), gripper_action)
                    candidate_actions.append(candidate_action)
                    next_obs = wm_env.simulate(candidate_action)
                    # next_action_chunk = policy_fn(next_obs['future_obs'][-1], subtask_id=1)[:20]
                    # next_obs = wm_env.simulate(next_action_chunk)
                imageio.mimwrite(os.path.join(save_dir, f'dp_placement_round0_candidate{i}.mp4'), next_obs['WMPredictionOutput'].full_video, fps=20)
                candidate_obs.append(next_obs['future_obs'])
            frontview_ranking = agent.rank_placement_frontview(candidate_obs)
            sideview_ranking = agent.rank_placement_sideview(candidate_obs)
            best_candidate = combine_rankings(frontview_ranking, sideview_ranking)
            print(best_candidate)
            action_chunk = candidate_actions[best_candidate]
            print(action_chunk)
            for action in action_chunk[:20]:
                obs, reward, done, info = wm_env.step(action)
                replay_images.append(obs["agentview_image"][::-1])
                num_steps += 1
            for _ in range(20):
                obs, reward, done, info = wm_env.step(LIBERO_DUMMY_ACTION)
                # replay_images.append(obs["agentview_image"][::-1])
                num_steps += 1

            print(f"Execution completed. Total steps: {num_steps}, Done: {done}")

    imageio.mimwrite(os.path.join(save_dir, f"replay_task{task_id}_seed{seed}.mp4"), replay_images, fps=20)
    
    print(f"Execution log saved to: {log_file_path}")
    print(f"Replay saved to: {os.path.join(save_dir, f'replay_task{task_id}_seed{seed}.mp4')}")

    print(f"Success {done}")
    return done


if __name__ == "__main__":

    from wm_client.client import WMClient
    host = "0.0.0.0"
    port = 7880
    wm_client = WMClient(host, port)

    for task in [1]:
        num_success = 0
        for seed in range(1, 10):
            success = run_mpc(task_id=task, seed=seed, wm_client=wm_client)
            if success:
                num_success += 1
            print(f"Number of successful runs: {num_success}/{seed+1}")
        with open(os.path.join("scratch_dir/mpc_data/libero_object/test_agent_search", f"task{task}", "result.txt"), "w") as f:
            f.write(f"Success rate: {num_success}/10\n")