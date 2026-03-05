import h5py
import numpy as np
import os
import io
import pathlib
from PIL import Image
from google.genai import types

from libero.libero import benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from vlm_api import call_api
from vlm_utils import *

prompt_base = """- Part 0: Instruction
You are a robotics expert, and you are here given a robot manipulation task.
Please analyze the task and given images, understand the task, and provide correct action proposals or help identify the right actioin.
"""

prompt_proposal = """- Part 4: Action Proposal
    Now, here comes your job. The current robot state is shown in the frontview, topview and sideview images below. 
    The robot is currently stuck and needs your help to propose the next actions. Here are the guidelines for analyzing the current state and proposing the next action:
    ### Guideline for analysis:
    - step 1: identify the task instruction and target objects based on the demonstration images and instruction in Part 1, and identify the position of the target objects in our current episode based on the current episode images in Part 2, refrencing the demonstration images in Part 1.
    - step 2: identify the state of the task, for example, which objects are already being moved and what should we do next. Use the observation history in Part 3 as a refrence to help understand the progress so far.
    ### **REMINDER** The observation history may be before the current observation, so when planning the next stap, use the current observation images as your starting point. 
    - step 3: identify the position of the target object in the images to operate next, follow the order of objects in the task description if there are multiple subtasks.
    - step 4: identify the target position of the robot gripper, where the robot should be ready to execute the actions to complete the next subtask.
    - step 5: To move the gripper from the current position to the target position, generate a mid-waypoint to avoid collision, so the gripper can follow the current position - mid-waypoint - target position trajectory to reach the target position safely without collision. 
    ### Guideline for action proposal:
    - return the target position of the gripper in the format of (x, y) coordinates of the image, normalized to 0-1000. 
    - return the coordinates of mid-waypoint and target position in json format for each image, for example:
    ```json
    [
      {
        "frontview": {"x": 500, "y": 300},
        "topview": {"x": 450, "y": 350},
        "sideview": {"x": 480, "y": 320}
        }, # mid-waypoint
      {
        "frontview": {"x": 600, "y": 400},
        "topview": {"x": 550, "y": 450},
        "sideview": {"x": 580, "y": 420}
        } # target position
        ...
    ]
    ```
    - If the object is being occluded in any of the views, either give the coordinates based on the other views, or skip the view in the output. We at least need to views and their target position coordinates.
    """

class VLMAgent:
    def __init__(self, 
                 task_suite_name, 
                 task_id,
                 obs_history_interval=4
                 ):
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.get_task_description()

        self.obs_history_interval = obs_history_interval

    def get_task_description(self):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        task = task_suite.get_task(self.task_id)

        def _get_libero_env(task, resolution, seed):
            """Initializes and returns the LIBERO environment, along with the task description."""
            task_description = task.language
            CAMERA_NAMES = ["agentview", "birdview", "robot0_eye_in_hand", "sideview", "canonical_frontview"]
            task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_names": CAMERA_NAMES}
            env = OffScreenRenderEnv(**env_args)
            env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
            return env, task_description
        env, self.task_description = _get_libero_env(task, resolution=256, seed=0)
        demonstration_path = os.path.join(get_libero_path("datasets"), task_suite.get_task_demonstration(self.task_id))
        f = h5py.File(demonstration_path, 'r')
        demo = f['data']['demo_0']
        states = np.array(demo['states'])
        env.reset()

        # get start image
        start_obs = env.set_init_state(states[5])
        start_image_agentview = start_obs['agentview_image'][::-1]
        start_image_topview = start_obs['birdview_image'][::-1]
        # get end image
        end_obs = env.set_init_state(states[-1])
        end_image_agentview = end_obs['agentview_image'][::-1]
        end_image_topview = end_obs['birdview_image'][::-1]
        
        # get asset images
        assets_dir = get_libero_path("assets")
        print(assets_dir)
        asset_contents = []
        for object_name, object_item in env.env.objects_dict.items():
            object_name = " ".join(object_name.split("_")[:-1])  # Remove the trailing number from the object name
            if object_name in self.task_description:
                object_name = object_name.replace(" ", "_")  # Replace spaces with underscores to match the asset file names
                object_class = str(object_item.__class__)
                if "hope_objects" in object_class:
                    asset_file = os.path.join(assets_dir, "stable_hope_objects", f"{object_name}", "texture_map.png")
                    assert os.path.isfile(asset_file), f"Asset file not found: {asset_file}"
                    print(f"Found asset file: {asset_file}")
                elif "google_scanned_objects" in object_class:
                    asset_file = os.path.join(assets_dir, "stable_scanned_objects", f"{object_name}", "texture.png")
                    assert os.path.isfile(asset_file), f"Asset file not found: {asset_file}"
                    print(f"Found asset file: {asset_file}")
                else:
                    asset_contents = []
                    break
                with open(asset_file, 'rb') as f:
                    image_bytes = f.read()
                asset_contents.extend([f"Here is the object texture image of {object_name.replace('_', ' ')} in the task description, where you can idetify the color of the object", types.Part.from_bytes(data=image_bytes, mime_type='image/png')])

        # close env
        env.close()
        del env

        start_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(start_image_agentview), mime_type='image/jpeg')
        start_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(start_image_topview), mime_type='image/jpeg')
        end_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(end_image_agentview), mime_type='image/jpeg')
        end_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(end_image_topview), mime_type='image/jpeg')

        prompt = f"""- Part 1: Task Description
    task instruction: {self.task_description}
    Here is the start frontview image and top view image, and the end frontview image and top view image of the task demonstration. Please identify the target objects in the task description, and analyze how to achieve the task goal based on the start and end images.
    ### CAUTION: do not identify the object completely based on the text description, look at the end image to see which object is being moved and how it is being operated.
    ### REMINDER: in real tasks below, the arrangements of the objects may be slightly different from the demonstration. So remember the actual shape and color of the target object, and do not solely rely on the relative position between objects in the demonstration. The target object may be partially occluded in the start image, so please analyze multiview images if needed.
        """
        self.task_prompt = [prompt_base+prompt, "Here is the frontview and topview images of the start state of demonstration", start_image_agentview_part, start_image_topview_part, "Here is the frontview and topview images of the end state of demonstration", end_image_agentview_part, end_image_topview_part] + asset_contents

    def start_episode(self, obs):
        self.obs_cache = []
        episode_start_image_topview = obs['birdview_image'][::-1]
        episode_start_image_agentview = obs['agentview_image'][::-1]
        episode_start_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(episode_start_image_agentview), mime_type='image/jpeg')
        episode_start_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(episode_start_image_topview), mime_type='image/jpeg')
        prompt = f"""- Part 2: Current Episode Observation
        Here are the frontview and topview start state images of the current episode of the task in Part 1. The arrangements of the objects may be slightly different from the demonstration, but the target object and instruction should be the same as in the demonstration.
        Please analyze the current images and identify the target objects, and understand how to achieve the task goal based on the current images and the task description you have already seen.
        """
        self.current_episode_prompt = [prompt, "Here is the frontview and topview START state images of our episode", episode_start_image_agentview_part, episode_start_image_topview_part]

    def cache_obs(self, obs):
        self.obs_cache.append(types.Part.from_bytes(data=numpy_to_jpeg_bytes(obs['agentview_image'][::-1]), mime_type='image/jpeg'))
    
    def verify_task_progress(self, interval=1, num_hist=1):
        prompt = """- Part 3: Task Progress Verification
        The robot has been executing actions to complete the task.
        The frontview image of the past step(s) and the frontview image of the current step is shown below. 
        Please analyze the images, and verify whether the robot has made progress in the current step compared with previous step(s).
        If there is progress, please return 1. If there is no progress or wrong progress, where the gripper is stucked in the same place, or the gripper is not moving towards the target object, return 0
        ### **REMINDER** The gripper movements may be slower when grasping/droping an object, so as long the robot is gasping or releasing the correct target object return 1.
        Also, sometimes when placing objects into a tight space, the object orientation must be aligned with the openings. Return 0 if the object orientation is wrong.
        ### **CAUTION** The gripper may reach for the **WRONG** object instead of the target object in task instruction as in Part 1 and Part 2. Please verify carefully if the gripper is moving towards the right target object. If not, return 0.
        return in json format, for example:
        ```json
        {
        "progress": 1
        }
        ```
        """
        content = self.task_prompt + self.current_episode_prompt + [prompt, "Here is the frontview image of the past steps",] + self.obs_cache[slice(-num_hist * interval - 1, -1, interval)] + ["Here is the frontview image of the current step", self.obs_cache[-1]] + \
            ["Reason about the task progress based on the images first following the previous instructions and analyze the gripper movements and object status (if object is grasped), and then return the result in json format."]
        response = call_api(content)
        print("API response for task progress verification:", response)
        return get_json(response)["progress"]

    def verify_subtask_completion(self, next_subtask_instruction):
        prompt = f"""- Part 4: Subtask Completion Verification
        The robot has been executing actions so far. Here is the next subtask instruction: {next_subtask_instruction}. 
        Please analyze whether the robot has completed the **previous** subtasks and is ready to **start** the next subtask, based on the observation history in Part 3. 
        Return 1 if the subtask is completed, 0 otherwise.""" + """
        Example:
        ```json
        {"start_subtask": 1}
        ```
        ## **REMINDER** The question is whether the robot has completed all previous subtasks and ready ro proceed to the mentioned subtask.
        """
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + [prompt]
        response = call_api(content)
        # print("API response for subtask completion verification:", response)
        return get_json(response)["start_subtask"]

    def reflect_on_obs_history(self):
        prompt = f"""- Part 3: Observations History Reflection
        The robot has been executing actions to complete the task, and the key frames of frontview images of the robot's observations during the execution are shown below. Please analyze the observation history and reflect on the task progress, and identify which step of the task we are currently at, and what is the next step to achieve the task goal.
        ### REMINDER: the observation images are only key frames, and there may have been missing frames between consecutive images.
        """
        self.obs_history_prompt = [prompt] + self.obs_cache[::self.obs_history_interval]

    def start_mpc(self, obs):
        current_image_frontview = obs['agentview_image'][::-1]
        current_image_topview = obs['birdview_image'][::-1]
        current_image_sideview = obs['sideview_image'][::-1]
        current_image_frontview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_frontview), mime_type='image/jpeg')
        current_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_topview), mime_type='image/jpeg')
        current_image_sideview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_sideview), mime_type='image/jpeg')
        self.mpc_obs = ["Here is the frontview image of the CURRENT state", current_image_frontview_part, "Here is the topview image of the CURRENT state", current_image_topview_part, "Here is the sideview image of the CURRENT state", current_image_sideview_part]

    def get_action_proposal(self):
        self.reflect_on_obs_history()
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + [prompt_proposal] + self.mpc_obs + \
            ["First reason from the multiview images and analyze following the guidelines, and then return the json output."]
        response = call_api(content)
        print("API response:", response)
        output = get_json(response)
        return output


if __name__ == "__main__":
    task_suite_name = "libero_10"
    task_id = 0
    seed = 0
    
    agent = VLMAgent(task_suite_name, task_id)