import io
import json
import re
from PIL import Image
import numpy as np
from typing import List
from google.genai import types 

from env_repos.robosuite.robosuite.models.tasks import task

view_name_mapping = {
    "frontview": "agentview",
    "topview": "birdview",
    "sideview": "sideview"
}

def numpy_to_jpeg_bytes(img_array):
    """Convert numpy image array to JPEG bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format='JPEG', quality=90)
    return buf.getvalue()

def get_json(response_text):
    # Use regular expression to extract the JSON part from the response text
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            output = json.loads(json_str)
            return output
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
            return None

    # Fallback: try to extract a raw JSON object or array
    raw_pattern = r'(\{.*\}|\[.*\])'
    match = re.search(raw_pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            output = json.loads(json_str)
            return output
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
            return None

    print("No JSON found in the response.")
    return None


# ------------------------------------------------------------
# Helper: Triangulate a single 3D point from multiple views
# ------------------------------------------------------------
def triangulate_multiview(uv_list, P_list):
    """
    uv_list:  list of (u, v) pixel coords for each camera
    P_list:   list of 3x4 projection matrices

    Returns: 3D point in world coordinates
    """
    A = []

    for (u, v), P in zip(uv_list, P_list):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])

    A = np.vstack(A)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]          # last row
    X = X_h[:3] / X_h[3]  # convert from homogeneous
    return X

def plot_coordinates_on_image(obs, coordinates, save_path=None):
    from PIL import ImageDraw, Image
    for view in coordinates:
        if view == "frontview":
            image = obs['agentview_image'][::-1]
        elif view == "topview":
            image = obs['birdview_image'][::-1]
        elif view == "sideview":
            image = obs['sideview_image'][::-1]
        else:
            continue
        
        x = int(coordinates[view]["x"] / 1000 * image.shape[1])
        y = int(coordinates[view]["y"] / 1000 * image.shape[0])
        
        # Plot a red dot on the image at the (x, y) coordinates
        image_with_dot = Image.fromarray(image)
        draw = ImageDraw.Draw(image_with_dot)
        draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='red')
        
        # Save or display the image with the plotted coordinates
        if save_path:
            image_with_dot.save(f"{save_path}_{view}_with_coordinates.jpg")
        else:
            image_with_dot.save(f"test_dp_{view}_with_coordinates.jpg")

def generate_3d_point(coordinates, camera_info):
    uv_list = []
    p_list = []
    for view in coordinates:
        if view not in view_name_mapping:
            continue
        _view = view_name_mapping[view]
        x = int(coordinates[view]["x"] / 1000 * camera_info[_view]["camera_width"])
        y = int(coordinates[view]["y"] / 1000 * camera_info[_view]["camera_height"])
        uv_list.append((x, y))
        p_list.append(camera_info[_view]["camera_transform"])

    return triangulate_multiview(uv_list, p_list)

def is_noop(action, obs, threshold=9e-2):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    gripper_position = obs['robot0_gripper_qpos'][0]
    moving = (np.sign(gripper_action) == np.sign(gripper_position - 0.02))
    # print("Norm", np.linalg.norm(action[:-1]))
    # print("gripper", gripper_action, prev_gripper_action)
    return np.linalg.norm(action[:-1]) < threshold and not moving


def optimize_trajectory(traj_response, scale=0.06):
    if traj_response['delta_z'] < 0:
        traj_response['delta_z'] = 0
    adjustment = np.array([traj_response['delta_x'] * 0.01, traj_response['delta_y'] * 0.01, traj_response['delta_z'] * scale])
    return adjustment

def optimize_endpoint(endpoint_response, scale=0.02):
    adjustment = np.array([scale * endpoint_response['x'], scale * endpoint_response['y'], scale * endpoint_response['z']])
    return adjustment

def optimize_rotation(action_chunk, rotation_response):
    euler_angles = [rotation_response['delta_roll'], rotation_response['delta_pitch'], rotation_response['delta_yaw']]
    axis_angles = np.radians(euler_angles)  # Convert degrees to radians
    length = len(action_chunk)
    action_chunk += (np.array([[0,0,0,axis_angles[0], axis_angles[1], axis_angles[2],0]]*length)/length)
    return action_chunk

def generate_rotation_candidates():
    candidates = [[1.0, 0.0, 0.0, 0.0], None]
    return candidates

def update_gripper_action(action_chunk, gripper_action):
    if gripper_action == -1:
        action_chunk[:, -1] = -1
    elif gripper_action == 1:
        action_chunk[:, -1] = 1
    else:
        raise ValueError("Invalid gripper action. Expected -1 or 1.")
    return action_chunk
    

def generate_candidates(target_point, gaussian=False, scale=0.05):
    candidates = [target_point]
    if not gaussian:
        noise = [[scale,0,0], [-scale,0,0], [0,scale,0], [0,-scale,0]]
        for n in noise:
            noised_point = target_point + np.array(n)
            candidates.append(noised_point)
    else:
        for _ in range(3):
            noised_point = target_point + np.random.normal(0, scale, size=3)
            candidates.append(noised_point)
    return candidates

def combine_rankings(ranking_1, ranking_2):
    combined_scores = {}
    
    # Assign score based on position in the ranking (lower score = better ranking)
    for i, candidate in enumerate(ranking_1):
        combined_scores[candidate] = combined_scores.get(candidate, 0) + i
    for i, candidate in enumerate(ranking_2):
        combined_scores[candidate] = combined_scores.get(candidate, 0) + i
    
    # Sort candidates by their combined scores (lowest score is the best)
    best_candidate = min(combined_scores, key=combined_scores.get)
    return best_candidate


view_config = {
    "topview": [1],
    "sideview": [0,2,7],
    "wristview": [1,3,4,5,6,8,9],
}
subtask_steps = {
    0: 100,
    4: 70,
    6: 100,
    7: 110,
    8: 150,
}
libero10_subtask_map = {
    0: [50, 54],
    1: [47, 51],
    2: [20, 19],
    3: [24, 22],
    4: [67, 68],
    5: [77],
    6: [72, 70],
    7: [46, 47],
    8: [38, 19],
    9: [-1, 33]
}
subtask_scales = {
    0: 0.5,
    4: 1.0,
    6: 0.5,
    7: 1.0

}

target_object = {
    0: "a blue round can",
    1: "a blue box",
    4: "a red ketchup bottle with silver cap",
    8: "a brown rectangle box"
}


if __name__ == "__main__":
    pass