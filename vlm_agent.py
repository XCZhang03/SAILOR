import h5py
import numpy as np
import os
import io
import pathlib

from tornado.process import task_id

from libero.libero import benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from vlm_api import call_api
from vlm_utils import *
from test_sam import *

prompt_base = """- Part 0: Instruction
You are a robotics expert, and you are here given a robot manipulation task.
Please analyze the task and given images, understand the task, and provide correct action proposals or help identify the right action.
"""

prompt_proposal = """- Part 3: Action Proposal
    Now, please propose the next actions for the robot to complete the task.
    You have already grasped the object, and now you need to move the object and place it into the basket.
    Output you action sequence composed of the atomic actions below, and follow the format strictly as in the example.
    The available atomic actions are:
    1. **MOVE** Here you move the gripper to a target position, and you should point it out in the multiview state images provided below. 
    The position should be represented by x,y pixel coordinates normalized to 0-999. 
    Examples:
    {
    "action": "MOVE",
    "parameters": {
    "frontview": {"x": 500, "y": 300},
    "topview": {"x": 450, "y": 350},
    "sideview": {"x": 400, "y": 250}
    }
    2. **ROTATION** Here you rotate the gripper, and you return the rotation in Euler angles [delta_roll, delta_pitch, delta_yaw] in degrees. 
    ### **REMINDER** 
    # a. The coordinate system and axis are defined as follows: from the frontview camera perspective,
    - The x axis is pointing towards the camera, with away from camera being negative x and towards the camera being positive x.
    - The y axis is pointing to the right, with left being negative y and right being positive y.
    - The z axis is pointing upwards, with down being negative z and upwards being positive z.
    # b. Reason carefully about the rotation direction, especially if we are about to grasp an object from the side, or release an object from the side.
    # c. If we are going to grasp an object and the gripper jaws are not aligned with the object, you may adjust the delta_yaw.
    # d. Grasping the **pot** from the **handle** is a typical case where we need to rotate the gripper to the side to grasp. When rotating the gripper.
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
    4. **GRASP** Here you grasp the object by closing the gripper.
    Example:
    {
    "action": "GRASP",
    "parameters": {}
    }
    **REMINDER** You should not return a single release acion in the final action list.
    Finally, return a list of actions in the order of execution.
    For example, 
    [
    {
    "action": "MOVE",
    "parameters": {
    "frontview": {"x": 500, "y": 300},
    "topview": {"x": 450, "y": 350},
    "sideview": {"x": 400, "y": 250}
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
    ```
    """

prompt_object = """- Part 2: Identify target object.
Based on the task instruction and target object images, please identify the position of the target object in the multiview images of the initial state.
 The position should be represented by x,y pixel coordinates normalized to 0-1000. 
 Return your result in json format, for example:
 ```json
    {
    "frontview": {"x": 500, "y": 300},
    "topview": {"x": 450, "y": 350},
    "sideview": {"x": 400, "y": 250}
    }
```
*** Make sure you are pointing at the same correct target object across all view images.
"""



image_ranking_frontview = """ - Part 3: Trajectory Images Ranking
Now,  I have a set of candidate images that shows the frontview of the robot state, grasping an object. 
I want you to rank them from best to worst in terms of steadly and cleanly grasping the object. First, reflect on the task instructions to determine the target object, then analyze following the guidelines below:
Guidelines for ranking the images:
**Zoom in on the target object and the robot gripper** 
1. When grasping a box, the best image should show the box is between the jaws of the robot gripper, not beside or in front of it.
2. Check the depth of the object and the robot gripper. If the gripper is behind the object it can not grasp the object firmly, thus the image should be ranked lower. The best image should show the gripper is at the center of the object to grasp the object firmly.
3. Check the position of the object. If the object is to the left or right of the gripper and not between the gripper, it can not grasp.
## **REMINDER**:
1. You should rank all the images following the same standard. If none of them is perfect, you should rank them by which one is the closest.
2. When ranking the later images, refer and reflect the previous candidates to rank them faithfully. For example, if the first image is blurry and the second image is clear, then the second image should be ranked higher than the first image.
3. Focus only on the target object which is being grasped.
Return the ranking result in json format, for example:
```json
[0, 2, 3, 1, 4, 5]
```
where the ids in the list are the image ids ranked from best to worst, with the first being the best. The range of the ids should be from 0 to N-1, where N is the total number of candidate images.
"""

image_ranking_wristview = """ - Part 3: Trajectory Images Ranking
Now, I have a set of wristview images showing the robot trying to grasp an object. I want you to rank them from best to worst in terms of whether the grasp is firm and clear. The best image should be a clear and firm grasp at the right place.
Here are more detailed guidelines:
1. Reflect on the trajectory history and task instructions, and understand which object is the gripper trying to grasp.
2. Identify the position of the jaws of the gripper in the wristview image, which is at the bottom. Verify whether the object is being grasped between the jaws clearly.
3. If we are grasping a box, we should have the entire box directly between our gripper jaws, which means **half** of the box should be visible at the bottom of the writview image.
4. If the entire box is visible in the wristview image, that means the gripper is too behind and thus not a firm grasp at the middle. **REMINDER** The bottom edge of the box should not be visible in the wristview image, otherwise the gripper is behind.
## **REMINDER**:
1. You should rank all the images following the same standard. If none of them is perfect, you should rank them by which one is closest.
2. When ranking the later images, refer and reflect the previous candidates to rank them faithfully. For example, if the first image is blurry and the second image is clear, then the second image should be ranked higher than the first image.
Return the ranking result in json format, for example:
```json
[0, 2, 3, 1, 4, 5]
```
where the ids in the list are the image ids ranked from best to worst, with the first being the best. The range of the ids should be from 0 to N-1, where N is the total number of candidate images.
"""

class VLMAgent:
    def __init__(self, 
                 task_suite_name, 
                 task_id,
                 ):
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.get_task_description()

        self.sideview = self.task_id in view_config['sideview']
        self.wristview = self.task_id in view_config['wristview']

    def get_task_description(self):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        task = task_suite.get_task(self.task_id)
        self.task_description = task.language
        self.target_object_prompt = target_object[self.task_id]
        self.object_image = Image.open(os.path.join(get_libero_path("assets"), "object_images", self.task_suite_name, f"{self.task_id}.png")).convert("RGB")
        self.object_image_byte = numpy_to_jpeg_bytes(np.array(self.object_image)[::-1])

    def start_episode(self, obs):
        self.obs_cache = []
        self.episode_start_image_topview = obs['birdview_image'][::-1]
        self.episode_start_image_agentview = obs['agentview_image'][::-1]
        self.episode_start_image_sideview = obs['sideview_image'][::-1]

        self.episode_start_image_topview_byte = numpy_to_jpeg_bytes(self.episode_start_image_topview)
        self.episode_start_image_agentview_byte = numpy_to_jpeg_bytes(self.episode_start_image_agentview)
        self.episode_start_image_sideview_byte = numpy_to_jpeg_bytes(self.episode_start_image_sideview)


        self.task_prompt = [f"""
        - Part 1: Task Description
        Here is the task_desciption: {self.task_description}
        and here is a close-up image for the target object:
        """] + [self.object_image_byte]

        self.episode_obs = ["Here is the frontview image of the initial state"] + [self.episode_start_image_agentview_byte] + ["Here is the topview image of the initial state"] + [self.episode_start_image_topview_byte] + ["Here is the sideview image of the initial state"] + [self.episode_start_image_sideview_byte]




    def identify_target_object(self):
        object_pixels = {}

        frontview_segmentation_results = segment_image(Image.fromarray(self.episode_start_image_agentview), self.target_object_prompt)
        if len(frontview_segmentation_results["masks"]) == 0:
            print("No object found in the frontview image for the prompt:", self.target_object_prompt)
        else:
            # Assuming the first mask is the one we want
            mask = frontview_segmentation_results["masks"][np.argmax(frontview_segmentation_results["scores"])]
            object_pixels['frontview'] = {
                "x": get_mask_center_pixel(mask)[0],
                "y": get_mask_center_pixel(mask)[1]
            }

        topview_segmentation_results = segment_image(Image.fromarray(self.episode_start_image_topview), self.target_object_prompt)
        if len(topview_segmentation_results["masks"]) == 0:
            print("No object found in the topview image for the prompt:", self.target_object_prompt)
        else:
            # Assuming the first mask is the one we want
            mask = topview_segmentation_results["masks"][np.argmax(topview_segmentation_results["scores"])]
            object_pixels['topview'] = {
                "x": get_mask_center_pixel(mask)[0],
                "y": get_mask_center_pixel(mask)[1]
            }

        sideview_segmentation_results = segment_image(Image.fromarray(self.episode_start_image_sideview), self.target_object_prompt)
        if len(sideview_segmentation_results["masks"]) == 0:
            print("No object found in the sideview image for the prompt:", self.target_object_prompt)
        else:
            # Assuming the first mask is the one we want
            mask = sideview_segmentation_results["masks"][np.argmax(sideview_segmentation_results["scores"])]
            object_pixels['sideview'] = {
                "x": get_mask_center_pixel(mask)[0],
                "y": get_mask_center_pixel(mask)[1]
            }

        return object_pixels

    def start_mpc(self, obs):
        frontview_image = obs['agentview_image'][::-1]
        topview_image = obs['birdview_image'][::-1]
        sideview_image = obs['sideview_image'][::-1]

        frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
        topview_image_byte = numpy_to_jpeg_bytes(topview_image)
        sideview_image_byte = numpy_to_jpeg_bytes(sideview_image)

        self.mpc_obs = ["Here is the frontview image of the current state"] + [frontview_image_byte] + ["Here is the topview image of the current state"] + [topview_image_byte] + ["Here is the sideview image of the current state"] + [sideview_image_byte]

    def get_action_proposal(self):
        content = self.task_prompt + [prompt_proposal] + self.mpc_obs + \
            ["Please analyze images, and return the proposed actions in json format as in the example."]
        response = call_api(content, thinking="low")
        print("API response:", response)
        output = get_json(response)
        return output

    def place_proposal(self):
        prompt = """- Part 1: Target position for placement
        Here, I want you to identify the target position for the gripper to prepare for placement. 
        You should point out the target position of the robot gripper in the multiview images provided below, so that the target object can be placed into the basket from there. 
        The position should be represented by x,y pixel coordinates normalized to 0-999.
        Format: Please return the target position in json format, for example:
```json
{
"frontview": {"x": 500, "y": 300},
"topview": {"x": 450, "y": 350},
"sideview": {"x": 400, "y": 250}
}
```
        """
        content = [prompt] + self.mpc_obs + \
            ["Please analyze the images, think step by step following the guidelines to identify the target position for placement, and return the result in json format."]
        response = call_api(content, thinking="low")
        print("API response for placement proposal:", response)
        return get_json(response)


    def optimize_trajectory(self, obs_list):
        prompt = """- Part 4: Optimize trajectory
        Here we provide a robot trajectory trying to comlete the task, and I need you to optimize the trajectory to make it safer and more successful in completing the task.
        I want you to return the direction to move the gripper so that it can be collision-free with objects and ready to complete the task.
        Return the gripper trajectory adjustments in x y z directions:
        The coordinate system and axis are defined as follows: from the frontview camera perspective,
        - The x axis is pointing towards the camera, with away from camera (towards the robot body) being negative x and towards the camera being positive x.
        - The y axis is pointing to the right, with left being negative y and right being positive y.
        - The z axis is pointing upwards, with down being negative z and upwards being positive z.
        For example, if you need to lift the gripper up to avoid collision, return 1 in z direction and 0 in x and y direction.
        Please return your adjustments direction in json format, for example:
        ```json
        {
          "delta_x": 0,
          "delta_y": 0,
          "delta_z": 1
        }
        ```
         where delta_x, delta_y, delta_z can only be -1, 0 or 1, with 0 being no movement in that direction, 1 being move towards positive direction and -1 being move towards negative direction.
        Here are the guidelines for optimization:
        0. Refelect on the task instruction and understand what the trajectory is trying to do.
        1. Analyze the trajectory frame by frame, and inspect whether there may be potential collisions with any object along the trajectory. 
            - Make sure that the gripper had cleared any previous objects such as backet rims during the movement, especially at the start of the trajectory.
            - If there is any potential collision with rims and objects, adjust the gripper direction, so that the trajectory is fully clear of any obstacles.
        2. The images are from simulation thus not perfectly realistic, if you find the trajectory is very close to the rim in the images, you should also adjust the gripper to be safer.
        """
        frontview_image_byte_list = []
        for obs in obs_list:
            frontview_image = obs['agentview_image'][::-1]
            frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
            frontview_image_byte_list.append(frontview_image_byte)
        content = self.task_prompt + [prompt] + ["Here is the frontview images of the trajectory to be optimized"] + frontview_image_byte_list[::2] + \
            ["Please analyze the trajectory based on the guidelines and images step by step, and then return the optimization result in json format."]
        response = call_api(content, thinking="low")
        print("API response for trajectory optimization:", response)
        return get_json(response)

    
    def optimize_height(self, obs_list):
        prompt = """- Part 3: Optimize gripper height
        Here I will give you a series of images showing the trajectory of the gripper approaching and trying to place an object.
        I want you to identify whether the gripper height need to be adjusted by lifting to avoid collision with objects.
        Please analyze following the guidelines below:
        1. First, reflect on the task instruction and understand what the gripper is trying to do.
        2. The gripper should be slightly above the object top to ensure enough room for descending and grasping, which is approximately half the object height.
        3. If the gripper jaws are almost touching the object in the final images and there is potential risk of collision, you should lift the gripper by returning 1
        4. If the gripper is at a safe height, you should return 0.
        Format:
        Please return your gripper height adjustment in json format, for example:
        ```json
        {
            "z": 0
        }
        ```
        where z means you are adjusting along the z axis, with 1 being upwards (lifting) and 0 being no adjustment.
        you should return 0 or 1 in the answer.
        """
        frontview_image_byte_list = []
        for obs in obs_list:
            frontview_image = obs['agentview_image'][::-1]
            frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
            frontview_image_byte_list.append(frontview_image_byte)
        content = self.task_prompt + [prompt] + frontview_image_byte_list[-4:] + \
        ["Please analyze the trajectory based on the guidelines and images, reason carefully step by step, and then return the result in json format."]
        response = call_api(content, thinking=None)
        print("API response for gripper height optimization:", response)
        return get_json(response)['z']
    
    def optimize_height_sideview(self, obs_list):
        prompt = """- Part 3: Optimize gripper height
        Here I will give you a image of the gripper approaching and trying to grasp an object from the frontview and sideview.
        I want you to identify whether the gripper height need to be adjusted by lifting to avoid collision with the object and ensure a safe grasp.
        Please analyze following the guidelines below:
        1. First, reflect on the task instruction and target object, and understand what the gripper is trying to grasp.
        2. The gripper should be above the object top to ensure enough room for descending and grasping, which is approximately half the object height.
        3. If the gripper jaws are almost touching the top of the object in the frontview images and there is potential risk of collision, you should lift the gripper by returning 1
        4. From the sideview image, zoom in on the gripper and the object, make sure the gripper jaws are above the top of the object.
        4. If the gripper is at a safe height clearly above the object, you should return 0.
        Format:
        Please return your gripper height adjustment in json format, for example:
        ```json
        {
            "z": 0
        }
        ```
        where z means you are adjusting along the z axis, with 1 being upwards (lifting) and 0 being no adjustment.
        you should return 0 or 1 in the answer.
        """
        frontview_image = obs_list[-1]['agentview_image'][::-1]
        frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
        sideview_image = obs_list[-1]['sideview_image'][::-1]
        sideview_image_byte = numpy_to_jpeg_bytes(sideview_image)
        content = self.task_prompt + self.current_episode_prompt + [prompt] + ["Here is the frontview image of the gripper approaching the object", frontview_image_byte, "Here is the sideview image of the gripper approaching the object", sideview_image_byte] + \
        ["Please analyze the images based on the guidelines, reason carefully step by step, and then return the result in json format."]
        response = call_api(content, thinking=None)
        print("API response for gripper height optimization:", response)
        return get_json(response)['z']

    def optimize_height_place(self, obs_list):
        prompt = """- Part 3: Optimize gripper height
        Here I will give you a series of images showing the trajectory of the gripper approaching and trying to place an object.
        I want you to identify whether the gripper height need to be adjusted by lifting to avoid collision with the object and ensure dropping from a safe height.
        Please analyze following the guidelines below:
        1. First, reflect on the task instruction, and understand what the gripper is trying to do. In this case, the gripper is trying to place an object into the basket, so the target object is the object in the gripper and the target region is the basket.
        2. The gripper and the object should be safely above the basket to drop and place.
        3. If the object is almost touching the rim in the final images and there is potential risk of collision, you should lift the gripper by returning 1
        4. If the gripper is at a safe height throughout the trajectory, you should return 0.
        Format:
        Please return your gripper height adjustment in json format, for example:
        ```json
        {
            "z": 0
        }
        ```
        where z means you are adjusting along the z axis, with 1 being upwards (lifting) and 0 being no adjustment.
        you should return 0 or 1 in the answer.
        """
        frontview_image_byte_list = []
        for obs in obs_list:
            frontview_image = obs['agentview_image'][::-1]
            frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
            frontview_image_byte_list.append(frontview_image_byte)
        content = self.task_prompt + self.current_episode_prompt + [prompt] + frontview_image_byte_list[-20:] + \
        ["Please analyze the trajectory based on the guidelines and images, reason carefully step by step, and then return the result in json format."]
        response = call_api(content, thinking="low")
        print("API response for gripper height optimization:", response)
        return get_json(response)['z']


    def optimize_endpoint(self, obs_list):
        obs = obs_list[-1]
        frontview_image = obs['agentview_image'][::-1]
        frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
        sideview_image = obs['sideview_image'][::-1]
        sideview_image_byte = numpy_to_jpeg_bytes(sideview_image)
        topview_image = obs['birdview_image'][::-1]
        topview_image_byte = numpy_to_jpeg_bytes(topview_image)
        wristview_image = obs['robot0_eye_in_hand_image'][::-1]
        wristview_image_byte = numpy_to_jpeg_bytes(wristview_image)
        prompt = f"""- Part 4: Optimize gripper position
        Here is the gripper position of our next robot action, and I want you to look carefully and analyze the position of the gripper and the object, and optimize the gripper position following the guidelines below.
        Return the gripper position adjustments in x y z directions:
        The coordinate system and axis are defined as follows: from the **frontview** camera perspective,
        - The x axis is pointing towards the camera, with away from camera being negative x and towards the camera being positive x.
        - The y axis is pointing to the right, with left being negative y and right being positive y.
        - The z axis is pointing upwards, with down being negative z and upwards being positive z.
        For example, if you need to lift the gripper up to avoid collision, return 1 in z direction and 0 in x and y direction; if you need to move the gripper to the right, return 1 in y direction and 0 in x and z direction; if you need to move the gripper towards the camera, return 1 in x direction and 0 in y and z direction.
        ## **Guidelines**:
        0. Understand what the gripper is trying to do, based on your task instruction understandings.
        1. If the gripper is about to grasp an object, the gripper should be aligned and directly above the object.
        2. Adjust the gripper if it is clearly misaligned with the object, with both gripper jaws outside the object
        Zoom in on the frontview image, and see if the gripper is below or above the object. If the gripper is clearly below and can not grasp, return 1 in z direction. Also, check whether the gripper is **clearly** to the left or right of the object with both **jaws** outside the object, and return the adjustment in y direction.
        {"Zoom in on the wristview image to see if the object is between the jaws of the gripper. For the wrist view image, the left in wristview is the right from the frontview which is +y, so if the object is in the right of the wristview you should move left in the frontview (right in wristview) which is the -y direction, and vice versa. The up in the wristview image is towards the camera, which is +x. The gripper jaws are at the bottom edge of the wristview image, and the object should appear in the bottom part of the wristview image, so if the object is at the upper part of the image, you should move +x and vice versa." if self.wristview else ""}
        {"Zoom in on the sideview image to see whetherthe gripper jaws is directly abovethe object. The left inthe sideview image is towards the camera, thus +x. Ifthe gripper jaws are positioned entirely to the right ofthe object, move in the +x direction, and vice versa. Ifthe gripper partially overlaps withthe object, do not adjustthe x-direction." if self.sideview else ""}
        3. When grasping a cup you should grasp bythe rim, sothe gripper should be placed above the rim instead ofthe body center. Do not adjustthe y direction unless both jaws are outside ofthe cup.
        Similarly, when grasping a box, if one ofthe grippers is abovethe box, do not adjust it. Adjustthe direction only if both gripper jaws are outside the object.
        4. Reason carefully about the object positions, make sure you are looking at the right object, and point to them before reasoning about the spatial relationships.
        {"Cross validate you output from multiple views to ensure the correctness of the directions" if self.sideview or self.wristview else ""}
        """
        format = """
        ## **Format**:
        Please return your optimization direction in json format, for example:
        ```json
        {
        "x": 0,
        "y": -1,
        "z": 1}
        ```
        where x, y, z can only be -1, 0 or 1, with 0 being no movement in that direction, 1 being move towards positive direction and -1 being move towards negative direction.
        Only adjust if the gripper is clearly misaligned, such as both the jaws are beside the object. Return all 0 if the object is mostly below the object.
        """
        content = self.task_prompt + self.current_episode_prompt + [prompt+format] + ["Here is the frontview image of the current state", frontview_image_byte,] + (["Here is the sideview image of the current state", sideview_image_byte] if self.sideview else []) + (["Here is the wristview image of the current state", wristview_image_byte] if self.wristview else []) + \
            ["Please analyze the gripper position based on the guidelines and images step by step, output your thought process, and then return the optimization result in json format as the example above."]
        response = call_api(content, thinking="low")
        print("API response for endpoint optimization:", response)
        return get_json(response)
    
    def verify_rotation(self, candidate_obs_list):
        prompt = """- Part 4: Verify gripper rotation
        Here I give you a set of candidate images that shows the frontview of the robot state, grasping an object with different gripper orientations. I want you to analyze the images and verify which image shows the best gripper orientation for a firm and clear grasp. Please return the id of the best image in json format, for example:
        ```json
        {
        "best_image_id": 0
        }
        ```
        where the best_image_id is the id of the image that shows the best gripper orientation ready for a firm and clear grasp, ranging from 0 to N-1, where N is the total number of candidate images.
        ## Guidelines:
        1. The position of the gripper may be mis-aligned with the target object, so only care about the rotation of the gripper.
        2. When grasping the box, the gripper should be horizontal and grasping the long edges.
        """
        frontview_image_byte_list = []
        for obs_list in candidate_obs_list:
            frontview_image = obs_list[-1]['agentview_image'][::-1]
            frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
            frontview_image_byte_list.append(frontview_image_byte)
        content = self.task_prompt + self.current_episode_prompt + [prompt] + ["Here is the frontview images of the candidates"] + frontview_image_byte_list + \
            ["Please analyze the gripper orientations in the images based on the guidelines and images step by step, and then return the verification result in json format as the example above."]
        response = call_api(content, thinking=None)
        print("API response for rotation verification:", response)
        return get_json(response)["best_image_id"]

    def rank_images_frontview(self, candidate_obs_list):
        frontview_image_byte_list = []
        for obs_list in candidate_obs_list:
            frontview_image = obs_list[-1]['agentview_image'][::-1]
            frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
            frontview_image_byte_list.append(frontview_image_byte)
        content = self.task_prompt + self.current_episode_prompt + [image_ranking_frontview] + frontview_image_byte_list + \
            ["Please reason carefully about the robot and object states in the images one by one, following the guidelines step by step, output your thought process, and rank them from best to worst in json format as example above"]
        response = call_api(content, thinking="low")
        print("API response for image ranking:", response)
        return get_json(response)

    def rank_images_wristview(self, candidate_obs_list):
        wristview_image_byte_list = []
        for obs_list in candidate_obs_list:
            wristview_image = obs_list[-1]['robot0_eye_in_hand_image'][::-1]
            wristview_image_byte = numpy_to_jpeg_bytes(wristview_image)
            wristview_image_byte_list.append(wristview_image_byte)
        content = self.task_prompt + self.current_episode_prompt + [image_ranking_wristview] + wristview_image_byte_list + \
            ["Please reason carefully about the gripper and object states in the images one by one, following the guidelines step by step, output your thought process, and then rank them from best to worst in json format as example above"]
        response = call_api(content, thinking="low")
        print("API response for wristview image ranking:", response)
        return get_json(response)
    
    def rank_placement_wristview(self, candidate_obs_list):
        prompt = """
- Part 4: Rank placement candidates
Here I give you a set of candidate images that shows the wristview of the robot state, above the basket and trying to drop an object into the basket.
 I want you to analyze the images and rank the position of the gripper:
- The gripper should be directly above the center of the basket, so in the wristview image, the jaws at the bottom of the image should be at the center of the basket.
- Only the upper half of the basket should be visible in the wristview image, and it should be centered.
Please return the ranking of the images in json format, for example:
```json
[0, 1, 2, 5, 4]
```
The first image in the list is the best one with the gripper most properly placed at the center.
The range of the ranking is from 0 to N-1, where N is the total number of candidate images.
 """
        wristview_image_byte_list = []
        for obs_list in candidate_obs_list:
            wristview_image = obs_list[-1]['robot0_eye_in_hand_image'][::-1]
            wristview_image_byte = numpy_to_jpeg_bytes(wristview_image)
            wristview_image_byte_list.append(wristview_image_byte)
        content = self.task_prompt + self.current_episode_prompt + [prompt] + wristview_image_byte_list + \
            ["Please reason carefully about the placement status of the object in the images one by one, following the guidelines step by step, output your thought process, and then rank the placements from best to worst in json format as example above"]
        response = call_api(content, thinking="low")
        print("API response for wristview placement ranking:", response)
        return get_json(response)
    
    def rank_placement_frontview(self, candidate_obs_list):
        prompt = """
- Part 4: Rank placement candidates
Here I give you a set of candidate images that shows the frontview of the robot state, above the basket and trying to drop an object into the basket.
 I want you to analyze the images and rank the position of the gripper:
- The gripper should be directly above the center of the basket, so in the frontview image, the gripper should be above the basket and at the middle of the basket.
- The gripper should be at a safe height above the basket, so the object can be dropped from a safe height without collision with the basket rim.
Please return the ranking of the images in json format, for example:
```json
[0, 1, 2, 5, 4]
```
The first image in the list is the best one with the gripper most properly placed at the center and at a safe height.
The range of the ranking is from 0 to N-1, where N is the total number of candidate images.
 """
        frontview_image_byte_list = []
        for obs_list in candidate_obs_list:
            frontview_image = obs_list[-1]['agentview_image'][::-1]
            frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
            frontview_image_byte_list.append(frontview_image_byte)
        content = self.task_prompt + self.current_episode_prompt + [prompt] + frontview_image_byte_list + \
            ["Please reason carefully about the placement status of the object in the images one by one, following the guidelines step by step, output your thought process, and then rank the placements from best to worst in json format as example above"]
        response = call_api(content, thinking="low")
        print("API response for frontview placement ranking:", response)
        return get_json(response)
    def rank_placement_sideview(self, candidate_obs_list):
        prompt = """
- Part 4: Rank placement candidates
Here I give you a set of candidate images that shows the sideview of the robot state, above the basket and trying to drop an object into the basket.
 I want you to analyze the images and rank the position of the gripper:
- The gripper should be directly above the center of the basket, so in the sideview image, the gripper should be above the basket and at the middle of the basket.
- If the gripper is to the left rim or to the right rim of the basket, and when opening the gripper, the grasped object may be droped out of the basket, so the position is bad. 
The gripper should be well aligned with the basket center, so that the object can be safely dropped into the basket.
Please return the ranking of the images in json format, for example:
```json
[0, 1, 2, 5, 4]
``` 
The first image in the list is the best one with the gripper most properly placed at the center and at a safe height.
The range of the ranking is from 0 to N-1, where N is the total number of candidate images.
 """
        sideview_image_byte_list = []
        for obs_list in candidate_obs_list:
            sideview_image = obs_list[-1]['sideview_image'][::-1]
            sideview_image_byte = numpy_to_jpeg_bytes(sideview_image)
            sideview_image_byte_list.append(sideview_image_byte)
        content = self.task_prompt + [prompt] + sideview_image_byte_list + \
            ["Please reason carefully about the placement status of the object in the images one by one, following the guidelines step by step, output your thought process, and then rank the positions from best to worst in json format as example above"]
        response = call_api(content, thinking="low")
        print("API response for sideview placement ranking:", response)
        return get_json(response)
    
    def optimize_placement(self, obs_list):
        prompt = """- Part 4: Optimize placement position
Here is the multiview images showing the gripper position above the basket, trying to drop the object directly into the basket.
I need you to optimize the positions of the gripper so that it is above the center of the basket, and the object can be droped safely inside.
 The coordinate system and axis are defined as follows: from the **frontview** camera perspective,
        - The x axis is pointing towards the camera, with away from camera being negative x and towards the camera being positive x.
        - The y axis is pointing to the right, with left being negative y and right being positive y.
        - The z axis is pointing upwards, with down being negative z and upwards being positive z.
        For example, if you need to lift the gripper up to avoid collision, return 1 in z direction and 0 in x and y direction; if you need to move the gripper to the right, return 1 in y direction and 0 in x and z direction; if you need to move the gripper towards the camera, return 1 in x direction and 0 in y and z direction.
## Guidelines:
1. Zoom in on the frontview image, and see if the gripper and the object is at a safe height well above the basket. If the object is too close to the basket rim and may collide with the rim when dropping, you should lift the gripper by returning 1 in z direction.
2. Zoom in on the frontview image, and see if the gripper is inside the basket. If the gripper is to the right rim, you should move left in the -y direction, and vice versa. If the gripper is inside between the left and right rim, you should return 0 in y direction without adjustment.
3. Zoom in on the sideview image, and see if the gripper is inside the basket. The right in the sideview image is away from the camera, thus -x, and the left in the sideview image is towards the camera, thus +x. If the gripper is to the right side of the basket near the right rim, you should move towards the camera in +x direction, and vice versa. If the gripper is inside between the left and right rim, you should return 0 in x direction without adjustment.
 ## **Format**:
        Please return your optimization direction in json format, for example:
        ```json
        {
        "x": 0,
        "y": -1,
        "z": 1}
        ```
        where x, y, z can only be -1, 0 or 1, with 0 being no movement in that direction, 1 being move towards positive direction and -1 being move towards negative direction.
        """
        obs = obs_list[-1]
        frontview_image = obs['agentview_image'][::-1]
        frontview_image_byte = numpy_to_jpeg_bytes(frontview_image)
        sideview_image = obs['sideview_image'][::-1]
        sideview_image_byte = numpy_to_jpeg_bytes(sideview_image)
        content = self.task_prompt + self.current_episode_prompt + [prompt] + ["Here is the frontview image of the current state", frontview_image_byte, "Here is the sideview image of the current state", sideview_image_byte] + \
            ["Please analyze the gripper position based on the guidelines and images step by step, output your thought process, and then return the optimization result in json format as the example above."]
        response = call_api(content, thinking="low")
        print("API response for placement optimization:", response)
        return get_json(response)


        


                                                                              

if __name__ == "__main__":
    task_suite_name = "libero_10"
    task_id = 0
    seed = 0
    
    agent = VLMAgent(task_suite_name, task_id)