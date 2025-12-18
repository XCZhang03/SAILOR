import os
import pathlib
import hydra
import torch
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.libero_utils import get_env_details, create_env


def load_checkpoint(checkpoint: str):
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.eval()

    return policy, cfg

def load_env(cfg, benchmark_name, task_indices, n_envs=1, output_dir="./scratch_dir/test"):
    cfg.task.env_runner._target_ = "diffusion_policy.env_runner.libero_image_runner.LIBEROImageRunner"
    cfg.task.env_runner.benchmark_name = benchmark_name
    cfg.task.env_runner.task_indices = task_indices
    cfg.task.env_runner.n_envs = n_envs

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )

    return env_runner.env, env_runner.env_init_fn_dills

from diffusion_policy.common.pytorch_util import dict_apply
import numpy as np

from diffusion_policy.common.libero_utils import LANG_EMBED_CACHE_FILE
def embed_lang(instruction: str) -> np.ndarray:
    lang_embed_cache = dict(np.load(LANG_EMBED_CACHE_FILE)) if os.path.exists(LANG_EMBED_CACHE_FILE) else dict()
    lang_embed = lang_embed_cache.get(instruction, None)
    # if lang_embed is not None:
    #     print('Loaded language embed from cache.')
    if lang_embed is None:
        raise NotImplementedError
        # from diffusion_policy.model.vision.model_getter import get_language_model
        # lang_encode_fn = get_language_model()
        # lang_embed = lang_encode_fn(instruction).astype(np.float32)
        # lang_embed_cache[instruction] = lang_embed
        # np.savez_compressed(LANG_EMBED_CACHE_FILE, **lang_embed_cache)
    return lang_embed


def slice_last_obs(x):
    if len(x.shape) == 2:
        return x
    elif len(x.shape) ==3:
        return x[:, -1]

class ProposalPolicy:
    def __init__(
            self,
            device = "cuda",
    ):
        self.device = torch.device(device)
        self.grasp_policy_dict = {}
        self.idm_policy = None

    def setup_grasp_policy(self, prompt: str, checkpoint: str):
        policy, cfg = load_checkpoint(checkpoint)
        policy = policy.to(self.device)
        policy.reset()
        obs_keys = cfg.shape_meta.obs.keys()
        def policy_fn(obs):
            np_obs_dict = dict(obs)
            shape = obs['agentview_image'].shape[:2]
            if 'lang_embed' in obs_keys:
                np_obs_dict['lang_embed'] = np.tile(embed_lang(prompt)[None, None, :], (*shape, *[1]*(embed_lang(prompt)[None, None, :].ndim-2)))
            np_obs_dict = {k: np_obs_dict[k] for k in obs_keys}
            obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=self.device))
            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action']
            return action
        self.grasp_policy_dict[prompt] = policy_fn

    def setup_idm_policy(self, checkpoint: str):
        policy, cfg = load_checkpoint(checkpoint)
        policy = policy.to(self.device)
        policy.reset()
        obs_keys = cfg.shape_meta.obs.keys()
        def policy_fn(obs, delta_obs_dict):
            np_obs_dict = dict(obs)
            np_obs_dict = {k: np_obs_dict[k] for k in obs_keys}
            obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(slice_last_obs(x)).to(
                        device=self.device))
            delta_obs_dict = dict_apply(delta_obs_dict,
                    lambda x: torch.from_numpy(x).to(
                        device=self.device))
            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict, delta_obs_dict)
            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy()) 
            action = np_action_dict['action']
            return action
        self.idm_policy = policy_fn

    def get_grasp_policy(self, prompt: str):
        return self.grasp_policy_dict[prompt]

    def get_idm_policy(self):
        return self.idm_policy
    
    def close_gripper(self):
        delta_obs_dict = {"robot0_gripper_qpos": np.array([[-1, 1]]) * 0.1}

    def follow_traj(self, trajectory):

        def policy_fn(obs, step_idx):
            cur_pos = slice_last_obs(obs['robot0_eef_pos'])
            target_pos = trajectory[step_idx]
            delta_obs_dict = {"robot0_eef_pos": target_pos - cur_pos}
            action = self.idm_policy(obs, delta_obs_dict)
            return action
        return policy_fn
            
            