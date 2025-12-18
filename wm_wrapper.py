import torch
import os
import imageio
from PIL import Image
from copy import deepcopy
from omegaconf import OmegaConf
import numpy as np
from collections.abc import Iterable
from collections import defaultdict, deque
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.gym_util.multistep_wrapper import stack_last_n_obs

# libero
# CKPT_PATH = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/dfot/outputs/2025-11-10/05-00-18/checkpoints/epoch=337-step=352196.ckpt"


# panel
# CKPT_PATH = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/dfot/outputs/2025-10-27/10-37-55/checkpoints/epoch=199-step=104200.ckpt"
# CKPT_PATH = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/dfot/outputs/2025-11-25/08-08-05/checkpoints/epoch=49-step=20850.ckpt"
CKPT_PATH = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/dfot/outputs/2025-11-30/00-13-38/checkpoints/epoch=90-step=464942.ckpt"

# robomimic pretrained
# CKPT_PATH = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/dfot/outputs/2025-09-26/22-35-12/checkpoints/last.ckpt"

def load_cfg(cfg_path, overrides=None):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)                   # resolve interpolations
    # Optional: safely set _name fields if your experiment code expects them
    cfg_choice = {
        'experiment': 'video_generation',
        'dataset': 'robomimic_pose',
        'algorithm': 'dfot_video_control'
    }
    cfg.experiment._name = cfg_choice["experiment"]
    cfg.dataset._name = cfg_choice["dataset"]
    cfg.algorithm._name = cfg_choice["algorithm"]
    return cfg

def load_dfot(ckpt_path=CKPT_PATH):
    experiment = load_experiment(ckpt_path=ckpt_path)
    algo = experiment._build_algo(checkpoint_path=ckpt_path).eval()
    return algo


def load_experiment(ckpt_path=CKPT_PATH):
    from experiments import build_experiment
    cfg_path = os.path.dirname(os.path.dirname(ckpt_path)) + "/.hydra/config.yaml"
    cfg = load_cfg(cfg_path)
    experiment = build_experiment(cfg, ckpt_path=ckpt_path, load_model_only=True)
    return experiment

def stack_obs_venv(obs_venv_candidates, len_context, obs_key):
    '''
    obs_venv_candidates: [[List({obs_key: [...]} x len_context) x num_envs] x num_candidates]
    '''
    key_obs_stacked = []  # list of arrays with shape [num_envs, len_context, ...]
    for obs_venv in obs_venv_candidates:
        # obs_venv: list (num_envs) of lists of frame dicts
        key_obs_stacked_venv = []  # list of arrays with shape [len_context, ...]
        for obs in obs_venv:
            key_obs_stacked_venv.append(
                stack_last_n_obs([frame[obs_key] for frame in obs], len_context)
            )
        # Stack across environments -> [num_envs, len_context, ...]
        key_obs_stacked.append(np.stack(key_obs_stacked_venv, axis=0))

    # Concatenate candidates along the first axis to match original behavior
    # Result shape: list(np.array[num_envs, len_context, ...] * num_candidates)
    return key_obs_stacked

def stack_imagine_obs_venv_dict(obs_venv_candidates, simulation_obs_venv_candidates=None, n_obs_steps=None):
    stacked_obs_dict_list = [ {} for _ in range(len(obs_venv_candidates))]
    if n_obs_steps is None:
        n_obs_steps = 1
    elif isinstance(n_obs_steps, Iterable):
        n_obs_steps = n_obs_steps[0]
    
    # override simulation obs (proprio states) with real obs if available
    if simulation_obs_venv_candidates is not None:
        for simulation_obs_key in simulation_obs_venv_candidates[0][0][0].keys():
            if simulation_obs_key not in simulation_obs_venv_candidates[0][0][-1].keys():
                continue
            for i in range(len(simulation_obs_venv_candidates)):
                stacked_obs_dict_list[i][simulation_obs_key] = stack_obs_venv(
                    [simulation_obs_venv_candidates[i]], n_obs_steps, obs_key=simulation_obs_key,
                )[0]
    for obs_key in obs_venv_candidates[0][0][0].keys():
        if obs_key not in obs_venv_candidates[0][0][-1].keys():
            continue
        for i in range(len(obs_venv_candidates)):
            stacked_obs_dict_list[i][obs_key] = stack_obs_venv(
                [obs_venv_candidates[i]], n_obs_steps, obs_key=obs_key,
            )[0]
    return stacked_obs_dict_list

def to_torch(np_array, device, dtype=None):
    return torch.from_numpy(np_array).to(device=device, dtype=dtype)

def vis_simulation_buffer(simulation_buffer_venv, vis_key="multiview_image"):
    image_files = []
    for i in range(len(simulation_buffer_venv)):
        image = simulation_buffer_venv[i][-1][vis_key]
        img = Image.fromarray((image * 255).astype(np.uint8).transpose(1,2,0))
        img.save(f'test_simulation_{i}.png')
        image_files.append(f'test_simulation_{i}.png')
    return image_files

def vis_imagine_buffer(obs_venv_candidates, vis_key="agentview_image", context_len=4):
    video_files = []
    final_frames = [[] for env_idx in range(len(obs_venv_candidates[0]))]
    for candidate_idx in range(len(obs_venv_candidates)):
        for env_idx in range(len(obs_venv_candidates[candidate_idx])):
            video = (np.stack([obs_frame[vis_key] for obs_frame in obs_venv_candidates[candidate_idx][env_idx][context_len:]], axis=0).transpose(0,2,3,1) * 255.0).astype(np.uint8)
            # pose_video = (np.stack([obs_frame[vis_key] for obs_frame in simulation_buffer_venv_candidates[candidate_idx][env_idx][-n_action_steps:]], axis=0).transpose(0,2,3,1) * 255.0).astype(np.uint8)
            # concat_video = np.concatenate([video, pose_video], axis=2)
            imageio.mimwrite(f'scratch_dir/test/libero_imagine_candidate{candidate_idx}_env{env_idx}.mp4', video, fps=8)
            video_files.append(f'scratch_dir/test/libero_imagine_candidate{candidate_idx}_env{env_idx}.mp4')
            final_frames[env_idx].append(video[-1])
    print(video_files)
    return final_frames

def process_obs_dict(obs_dict, render_key="multiview_image"):
    if render_key == "multiview_image":
        image_list = [obs_dict['agentview_image'], obs_dict['birdview_image'], obs_dict['frontview_image'], obs_dict['sideview_image']]
        top = np.concatenate([image_list[0], image_list[1]], axis=2)
        bottom = np.concatenate([image_list[2], image_list[3]], axis=2)
        obs_dict['multiview_image'] = np.concatenate([top, bottom], axis=1)
    return obs_dict

def create_obs_dict_from_video_frame(video_frame, render_key="multiview_image"):
    if render_key == "multiview_image":
        c, h, w = video_frame.shape
        h_half = h // 2
        w_half = w // 2
        obs_dict = {
            'agentview_image': video_frame[:, 0:h_half, 0:w_half],
            'birdview_image': video_frame[:, 0:h_half, w_half:w],
            'frontview_image': video_frame[:, h_half:h, 0:w_half],
            'sideview_image': video_frame[:, h_half:h, w_half:w],
            'multiview_image': video_frame,
        }
    else:
        obs_dict = {
            render_key: video_frame,
        }
    return obs_dict




class DFoTWrapper:
    def __init__(
        self,
        algo,
        device="cuda",
    ):
        self.algo = algo
        self.algo.to(device)
        self.device = device
        self.context_len = self.algo.n_context_tokens
        self.key_frame_density = 1/2  # 1/2 means every 2 frames is a key frame
        self.algo.cfg.tasks.prediction.keyframe_density = self.key_frame_density

        self.imagine_horizon = 100
        # self.render_key = "agentview_image"
        self.render_key = "multiview_image"
        self.vis_key = "frontview_image"

        self.reset_buffer()

    def reset_buffer(self):
        self.simulation_buffer_venv = None

    def simulate_actions(self, actions, env, robot_state=None, simulation_buffer_venv=None):
        if robot_state is not None:
            env.call_each('set_robot', 
                args_list=[(robot_state[i],) for i in range(env.num_envs)])
        else:
            env.call("set_robot")

        # lazy buffer init
        if simulation_buffer_venv is None:
            simulation_buffer_venv = self.simulation_buffer_venv
        if simulation_buffer_venv is None:
            self.simulation_buffer_venv = simulation_buffer_venv = [deque(maxlen=self.context_len) for _ in range(env.num_envs)]

        for step_index in range(actions.shape[1]):
            action_step = actions[:, step_index]
            simulation_obs_venv = env.call_each(
                'simulation_step', 
                args_list=[(action_step[i],) for i in range(env.num_envs)]
            )
            for i in range(len(simulation_obs_venv)):
                simulation_buffer_venv[i].append(process_obs_dict(simulation_obs_venv[i], render_key=self.render_key))
        robot_state_venv = env.call("get_simulation_robot_state")
        return robot_state_venv


    def imagine(self, action_candidates, env):
        obs_venv = env.call("obs")
        self.n_action_steps = action_candidates[0].shape[1]
        # remove dequeue length constraints
        self.obs_venv_candidates = [[list([process_obs_dict(obs_frame, render_key=self.render_key) for obs_frame in obs])[-self.context_len:] for obs in obs_venv] for _ in range(len(action_candidates))]
        self.simulation_buffer_venv_candidates = [[list(simulation_buffer)[-self.context_len:] for simulation_buffer in self.simulation_buffer_venv] for _ in range(len(action_candidates))]

        for action_index, action_candidate in enumerate(action_candidates):
            self.simulate_actions(
                action_candidate,
                env,
                simulation_buffer_venv=self.simulation_buffer_venv_candidates[action_index]
            )

        self.predict_videos()
        return vis_imagine_buffer(self.obs_venv_candidates, vis_key=self.vis_key, context_len=self.context_len)

    def imagine_policy(self, policy_candidates, env, n_imagine_steps=2):
        obs_venv = env.call("obs")
        # remove dequeue length constraints
        self.obs_venv_candidates = [[list([process_obs_dict(obs_frame, render_key=self.render_key) for obs_frame in obs])[-self.context_len:] for obs in obs_venv] for _ in range(len(policy_candidates))]
        self.simulation_buffer_venv_candidates = [[list(simulation_buffer)[-self.context_len:] for simulation_buffer in self.simulation_buffer_venv] for _ in range(len(policy_candidates))]

        simulation_state_venv_candidates = [None] * len(policy_candidates)
        for step_idx in range(n_imagine_steps):
            obs_dict_list = stack_imagine_obs_venv_dict(self.obs_venv_candidates, self.simulation_buffer_venv_candidates, n_obs_steps=env.call("n_obs_steps"))
            action_candidates = [policy(obs_dict, step_idx=step_idx) for policy, obs_dict in zip(policy_candidates, obs_dict_list)]
            self.n_action_steps = action_candidates[0].shape[1]
            for candidate_index, action_candidate in enumerate(action_candidates):
                simulation_state_venv = self.simulate_actions(
                    action_candidate,
                    env,
                    simulation_buffer_venv=self.simulation_buffer_venv_candidates[candidate_index],
                    robot_state=simulation_state_venv_candidates[candidate_index],
                )
                simulation_state_venv_candidates[candidate_index] = simulation_state_venv
            self.predict_videos()

        return vis_imagine_buffer(self.obs_venv_candidates, vis_key=self.vis_key, context_len=self.context_len)


    def predict_videos(self):
        video_context = to_torch(
            np.concatenate(stack_obs_venv(         
                self.obs_venv_candidates, self.context_len, obs_key=self.render_key,
            ), axis=0), device=self.algo.device, dtype=self.algo.dtype
        )
        simulation_context = to_torch(
            np.concatenate(stack_obs_venv(
                self.simulation_buffer_venv_candidates, self.context_len + self.n_action_steps, obs_key=self.render_key,
            ), axis=0), device=self.algo.device, dtype=self.algo.dtype
        )
        video = torch.cat([video_context, torch.zeros(video_context.size(0), self.n_action_steps, *video_context.shape[2:], device=video_context.device, dtype=video_context.dtype)], dim=1)
        cond = simulation_context

        assert video.shape == cond.shape, f"Video shape {video.shape} and cond shape {cond.shape} do not match."
        batch = {
            'videos': video,
            'conds': cond,
        }
        batch = self.algo.on_after_batch_transfer(batch, dataloader_idx=0)
        with torch.no_grad():
            videos = self.algo._sample_all_videos(batch, batch_idx=0)['prediction'].clamp(0, 1)
        
        video_candidates = torch.chunk(videos, len(self.obs_venv_candidates), dim=0)
        video_venv_candidates = [video.unbind() for video in video_candidates]
        for candidate_idx in range(len(video_venv_candidates)):
            for env_idx in range(len(video_venv_candidates[candidate_idx])):
                video = video_venv_candidates[candidate_idx][env_idx][-self.n_action_steps:].cpu().numpy()  # [n_action_steps, C, H, W]
                obs_frame = [create_obs_dict_from_video_frame(video_frame, render_key=self.render_key) for video_frame in video]
                self.obs_venv_candidates[candidate_idx][env_idx].extend(obs_frame)
        return 


if __name__ == "__main__":
    device = "cuda"
    algo = load_dfot()
    wm = DFoTWrapper(
        algo=algo,
        device=device,
    )
    

