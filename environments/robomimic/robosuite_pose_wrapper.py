from collections import OrderedDict

import numpy as np
import robosuite as suite
from gym import spaces
from robosuite.wrappers import GymWrapper
from termcolor import cprint
from environments.robomimic.robosuite_image_wrapper import RobosuiteImageWrapper


class RobosuitePoseWrapper(RobosuiteImageWrapper):
    """
    A modified version of the GymWrapper class from the Robosuite library.
    This wrapper is specifically designed for handling image observations in Robosuite environments.

    Args:
        env (gym.Env): The underlying Robosuite environment.
        shape_meta (dict): A dictionary containing shape information for the observations.
        keys (list, optional): A list of observation keys to include in the wrapper. Defaults to None.
        add_state (bool, optional): Whether to include the state information in the observations. Defaults to True.
            If true, all non-image observation keys are concatenated into a single value labelled by the "state"
            key in the observation dictionary.

    Attributes:
        action_space (gym.Space): The action space of the environment.
        observation_space (gym.Space): The observation space of the environment.
        render_cache (numpy.ndarray): The last rendered image.
        render_obs_key (str): The key of the observation to be used for rendering.

    Note:
        Both the reset() and step() functions follow the Gym API.

    Raises:
        RuntimeError: If an unsupported observation type is encountered.

    """

    def __init__(
        self, env_kwargs, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        empty_env_kwargs = env_kwargs.copy()
        empty_env_kwargs['env_name'] = "EmptyEnv"
        empty_env_kwargs['hard_reset'] = False
        empty_env_kwargs['has_offscreen_renderer'] = False
        empty_env_kwargs['has_renderer'] = False
        empty_env_kwargs['use_camera_obs'] = False
        self.empty_env = suite.make(**empty_env_kwargs)
        self.empty_env.copy_env_model(self.env)
        self.reset()
        
        
        self.camera_names = []
        for key in self.observation_space.keys():
            if "image" in key and 'robot' not in key:
                self.camera_names.append(key.replace("_image", ""))
                
    def set_robot(self):
        return {"qpos": self.empty_env.copy_robot_state(self.env)}
    
    def render_action_pose(self, actions, set_robot=False):
        if set_robot:
            self.set_robot()
        self.simulation_step(actions)
        return self.render_simulation_pose()

    def render_simulation_pose(self):
        action_poses = {}
        res = (self.env.camera_heights[0], self.env.camera_widths[0])
        for cam_name in self.camera_names:
            camera_transform = self.empty_env.get_camera_transform(camera_name=cam_name, camera_width=res[1], camera_height=res[0])
            pose_image = self.empty_env.plot_pose(camera_transform, height=res[0], width=res[1])
            action_poses[f"{cam_name}_image"] = pose_image
        return action_poses

    def reset(self, **kwargs):
        self.empty_env.reset()
        returns = super().reset(**kwargs)
        self.set_robot()
        return returns
    
    def simulation_step(self, actions):
        if len(actions.shape) == 1:
            actions = actions[None, :]
        for action in actions:
            self.empty_env.step(action)
    
    def step(self, action, step_simulation=False):
        returns = super().step(action)
        if step_simulation:
            self.set_robot()
        return returns