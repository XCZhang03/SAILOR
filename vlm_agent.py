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
Please analyze the task and given images, understand the task, and provide correct action proposals or help identify the right action.
"""

prompt_proposal = """- Part 4: Action Proposal
    Now, please propose the next actions for the robot to complete the task. You should complete the task following the order in the instruction. 
    The available atomic actions are:
    1. **MOVE** Here you move the gripper to a target position, and you should point it out in the multiview state images provided below. 
    The position should be represented by x,y pixel coordinates normalized to 0-1000. 
    **REMINDER** To grasp an object, simply move towards it.
    Examples:
    {
    "action": "MOVE",
    "parameters": {
    "frontview": {"x": 500, "y": 300},
    "topview": {"x": 450, "y": 350},
    "sideview": {"x": 480, "y": 320}
    }
    2. **ROTATION** Here you rotate the gripper, and you return the rotation in Euler angles [delta_roll, delta_pitch, delta_yaw] in degrees. 
    ### **REMINDER** 
    # a. The coordinate system and axis are defined as follows: from the frontview camera perspective,
    - The x axis is pointing towards the camera, with away from camera being negative x and towards the camera being positive x.
    - The y axis is pointing to the right, with left being negative y and right being positive y.
    - The z axis is pointing upwards, with down being negative z and upwards being positive z.
    # b. Reason carefully about the rotation direction, especially if we are about to place an object into a small space, and we have to make sure the orientation of the object matches the opening space.
    Example:
    {
    "action": "ROTATE",
    "parameters": {
    "delta_roll": 0,
    "delta_pitch": 15,
    "delta_yaw": 0
    }
    }
    3. **RELEASE** Here you release the object by opening the gripper.
    Example:
    {
    "action": "RELEASE",
    "parameters": {}
    }
    Finally, return a list of actions in the order of execution. For example, 
    [
    {
    "action": "MOVE",
    "parameters": {
    "frontview": {"x": 500, "y": 300},
    "topview": {"x": 450, "y": 350},
    "sideview": {"x": 480, "y": 320}
    }
    },
    {
    "action": "ROTATION",
    "parameters": {
    "delta_roll": 0,
    "delta_pitch": 15,
    "delta_yaw": 0
    }
    },
    {
    "action": "RELEASE",
    "parameters": {}
    }
    ]
    **REMINDER** You should not return a single release acion in the final action list.
    ```
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
        # close env
        env.close()
        del env

        start_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(start_image_agentview), mime_type='image/jpeg')
        start_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(start_image_topview), mime_type='image/jpeg')
        end_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(end_image_agentview), mime_type='image/jpeg')
        end_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(end_image_topview), mime_type='image/jpeg')

        prompt = f"""- Part 1: Task Description
    task instruction: {self.task_description}
        """
        self.task_prompt = [prompt_base+prompt, "Here is the frontview and topview images of the start state of demonstration", start_image_agentview_part, start_image_topview_part, "Here is the frontview and topview images of the end state of demonstration", end_image_agentview_part, end_image_topview_part]

    def start_episode(self, obs):
        self.obs_cache = []
        episode_start_image_topview = obs['birdview_image'][::-1]
        episode_start_image_agentview = obs['agentview_image'][::-1]
        episode_start_image_agentview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(episode_start_image_agentview), mime_type='image/jpeg')
        episode_start_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(episode_start_image_topview), mime_type='image/jpeg')
        prompt = f"""- Part 2: Current Episode Observation
        Here are the frontview start state images of the current episode of the task in Part 1.
        Please understand how to achieve the task goal based on the task description you have already seen.
        """
        self.current_episode_prompt = [prompt, "Here is the frontview start state images of our episode", episode_start_image_agentview_part]

    def cache_obs(self, obs):
        self.obs_cache.append(types.Part.from_bytes(data=numpy_to_jpeg_bytes(obs['agentview_image'][::-1]), mime_type='image/jpeg'))
    
    def verify_task_progress(self, interval=1, num_hist=1):
        prompt = """- Part 3: Task Progress Verification
        The robot has been executing actions to complete the task.
        The frontview image of the past step(s) and the frontview image of the current step is shown below. 
        Please analyze the images, and verify whether the robot has made progress in the current step compared with previous step(s).
        If there is progress, please return 1. If there is no progress or wrong progress, where the gripper is stucked in the same place, or the gripper is not moving towards the target position, return 0
        return in json format, for example:
        ```json
        {
        "progress": 1
        }
        ```
        """
        content = self.task_prompt + self.current_episode_prompt + [prompt, "Here is the frontview image of the past steps",] + self.obs_cache[slice(max(-num_hist * interval - 1, -len(self.obs_cache)), -1, interval)] + ["Here is the frontview image of the current step", self.obs_cache[-1]] + \
            ["Reason about the task progress based on the images first following the previous instructions and analyze the gripper movements and object status (if object is grasped), and then return the result in json format."]
        response = call_api(content, thinking="low")
        print("API response for task progress verification:", response)
        return get_json(response)["progress"]

    def verify_subtask_completion(self, subtasks, obs):
        current_frontview_image = obs['agentview_image'][::-1]
        current_frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_frontview_image), mime_type='image/jpeg')
        prompt = f"""- Part 4: Subtask Completion Verification
        Here we have decomposed the task instruction in Part 1 into two subtasks {subtasks[0]} and {subtasks[1]}, where {subtasks[0]} should be completed before {subtasks[1]}.
        The robot has been executing actions, and I need you to verify in the last step whether the first subtask has completed and we can move on to the second subtask, or we are still in the first subtask and need to keep working on it.
         Please analyze the observation history and the current images, and verify whether the first subtask has been completed. If the first subtask has been completed and we can move on to the second subtask, please return 1. If we are still in the first subtask and need to keep working on it, please return 0. 
        return in json format, for example:
        ```json
        [0]
        ```
        """
        content = self.task_prompt + self.current_episode_prompt + [prompt] + ["Here is the frontview images of the past history steps"] + self.obs_cache[::self.obs_history_interval] + ["Here is the frontview image of the current step", current_frontview_image_part] + \
            ["Reason about the subtask completion based on the task instructions and current image, and then return the result in json format."]
        response = call_api(content, thinking="low")
        print("API response for subtask completion verification:", response)
        return get_json(response)[0]

    def reflect_on_obs_history(self):
        prompt = f"""- Part 3: Observations History Reflection
        The robot has been executing actions to complete the task, and the key frames of frontview images of the robot's observations during the execution are shown below.
        """
        self.obs_history_prompt = [prompt] + self.obs_cache[::self.obs_history_interval]

    def start_mpc(self, obs):
        current_image_frontview = obs['agentview_image'][::-1]
        current_image_topview = obs['birdview_image'][::-1]
        current_image_sideview = obs['sideview_image'][::-1]
        current_image_wristview = obs['robot0_eye_in_hand_image'][::-1]
        current_image_frontview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_frontview), mime_type='image/jpeg')
        current_image_topview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_topview), mime_type='image/jpeg')
        current_image_sideview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_sideview), mime_type='image/jpeg')
        current_image_wristview_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(current_image_wristview), mime_type='image/jpeg')
        self.mpc_obs = ["Here is the frontview image of the current state", current_image_frontview_part, "Here is the topview image of the current state", current_image_topview_part, "Here is the sideview image of the current state", current_image_sideview_part, "Here is the wristview image of the current state", current_image_wristview_part]
    
    def get_action_proposal(self):
        self.reflect_on_obs_history()
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + [prompt_proposal] + self.mpc_obs + \
            ["Please analyze the previous history, understand the state of the task right now, and return the proposed action sequence in json format."]
        response = call_api(content, thinking=None)
        print("API response:", response)
        output = get_json(response)
        return output

    def verify_proposal(self, obs_list):
        prompt = """- Part 4: Verify trajectory
        Here we provide a robot trajectory trying to comlete the task, and I need you to verify wether it is safe and successfully completes actions.
        Here is the guidelines of verification:
        1. Reflect on the task history in Part 3 and understand what to do next, and see if the overall direction of the trajectory is correct, for example, whether the gripper is moving towards the target object or moving the object towards the target location.
        2. Analyze the trajectory frame by frame, and inspect whether there may be potential collitions with any object along the trajectory. If there is collition, return 0.
        **REMINDER** Since the frames are from simulation, some objects may appear blurry when collides. Infer from the gripper positions whether there may be potential collition.
        3. Analyze whether the trajectory has completed the subgoal. In partivular, when we need to grasp an object, whether we have successfully grasped it and lifted it up. When we are trying to place an object, whether the object is aligned with the openings or target regions.
        If the gripper is smoothly operating without collision and also correctly completing the subgoal or making correct progress, return 1. Else return 0.
        Please return your verification result in json format, for example:
        ```json
        {"verified": 1}
        ```
        """
        frontview_image_part_list = []
        for obs in obs_list:
            frontview_image = obs['agentview_image'][::-1]
            frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(frontview_image), mime_type='image/jpeg')
            frontview_image_part_list.append(frontview_image_part)
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + prompt + ["Here is the frontview images of the trajectory to be verified"] + frontview_image_part_list[::4] + \
            ["Please analyze the trajectory based on the guidelines and images step by step, and then return the verification result in json format."]
        response = call_api(content, thinking=None)
        print("API response for trajectory verification:", response)
        return get_json(response)["verified"]
    
    def optimize_trajectory(self, obs_list):
        prompt = """- Part 4: Optimize trajectory
        Here we provide a robot trajectory trying to comlete the task, and I need you to optimize the trajectory to make it safer and more successful in completing the task.
        I want you to return the direction to move the gripper so that it can be collision-free and ready to complete the task.
        Return the gripper trajectory adjustments in x y z directions:
        The coordinate system and axis are defined as follows: from the frontview camera perspective,
        - The x axis is pointing towards the camera, with away from camera being negative x and towards the camera being positive x.
        - The y axis is pointing to the right, with left being negative y and right being positive y.
        - The z axis is pointing upwards, with down being negative z and upwards being positive z.
        For example, if you need to lift the gripper up to avoid collision, return 1 in z direction and 0 in x and y direction; if you need to move the gripper right, return 1 in y direction and 0 in x and z direction; if you need to move the gripper towards the camera, return 1 in x direction and 0 in y and z direction.
        Please return your optimization direction in json format, for example:
        ```json
        {
        "x": 0,
        "y": -1,
        "z": 1}
        ```
         where x, y, z can only be -1, 0 or 1, with 0 being no movement in that direction, 1 being move towards positive direction and -1 being move towards negative direction.
        Here is the guidelines for analyzing the trajectory::
        1. Analyze the trajectory frame by frame, and inspect whether there may be potential collitions with any object along the trajectory. 
        2. If there is collision in the middle of the trajectory, adjust the gripper and return the result. Make sure the gripper is lifted and cleared of all objects. If not, make your adjustments and return them.
        """
        frontview_image_part_list = []
        for obs in obs_list:
            frontview_image = obs['agentview_image'][::-1]
            frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(frontview_image), mime_type='image/jpeg')
            frontview_image_part_list.append(frontview_image_part)
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + prompt + ["Here is the frontview images of the trajectory to be optimized"] + frontview_image_part_list[::2] + \
            ["Please analyze the trajectory based on the guidelines and images step by step, and then return the optimization result in json format."]
        response = call_api(content, thinking=None)
        print("API response for trajectory optimization:", response)
        return get_json(response)

    def optimize_endpoint(self, obs_list):
        obs = obs_list[-1]
        frontview_image = obs['agentview_image'][::-1]
        frontview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(frontview_image), mime_type='image/jpeg')
        sideview_image = obs['sideview_image'][::-1]
        sideview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(sideview_image), mime_type='image/jpeg')
        topview_image = obs['birdview_image'][::-1]
        topview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(topview_image), mime_type='image/jpeg')
        wristview_image = obs['robot0_eye_in_hand_image'][::-1]
        wristview_image_part = types.Part.from_bytes(data=numpy_to_jpeg_bytes(wristview_image), mime_type='image/jpeg')
        prompt = """- Part 4: Optimize gripper position
        Here is the gripper position of our next robot action, and I want you to look carefully and analyze the position of the gripper and the object, and optimize the gripper position following the guidelines below.
        ## **Format**: Return the gripper trajectory adjustments in x y z directions:
        The coordinate system and axis are defined as follows: from the frontview camera perspective,
        - The x axis is pointing towards the camera, with away from camera being negative x and towards the camera being positive x.
        - The y axis is pointing to the right, with left being negative y and right being positive y.
        - The z axis is pointing upwards, with down being negative z and upwards being positive z.
        For example, if you need to lift the gripper up to avoid collision, return 1 in z direction and 0 in x and y direction; if you need to move the gripper right, return 1 in y direction and 0 in x and z direction; if you need to move the gripper towards the camera, return 1 in x direction and 0 in y and z direction.
        Please return your optimization direction in json format, for example:
        ```json
        {
        "x": 0,
        "y": -1,
        "z": 1}
        ```
        where x, y, z can only be -1, 0 or 1, with 0 being no movement in that direction, 1 being move towards positive direction and -1 being move towards negative direction.
        ## **Guidelines**:
        0. Understand what the gripper is trying to do, based on your task understandings and the history.
        1. If the gripper is about to grasp an object, make sure the gripper is above and well aligned with the object.
        Zoom in on the wristview image to see if the object is between the jaws of the gripper. If not, move the gripper so that it is above the object and well aligned with the object.
        Zoom in on the frontview images, and see if the gripper is directly above the object. If the gripper is below and can not grasp, return 1 in z directioin.
        2. If the gripper is about to place an object, make sure the object is aligned with the target region or openings.
        Zoom in on the images, and return the adjustments if needed.
        """
        content = self.task_prompt + self.current_episode_prompt + self.obs_history_prompt + [prompt] + ["Here is the frontview image of the current state", frontview_image_part, "Here is the sideview image of the current state", sideview_image_part, "Here is the topview image of the current state", topview_image_part, "Here is the wristview image of the current state", wristview_image_part] + \
            ["Please analyze the gripper position based on the guidelines and images step by step, and then return the optimization result in json format."]
        response = call_api(content, thinking=None)
        print("API response for endpoint optimization:", response)
        return get_json(response)

                                                                              

if __name__ == "__main__":
    task_suite_name = "libero_10"
    task_id = 0
    seed = 0
    
    agent = VLMAgent(task_suite_name, task_id)