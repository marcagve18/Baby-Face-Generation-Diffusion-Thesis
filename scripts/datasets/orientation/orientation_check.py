import json
import os
import numpy as np
import cv2


DATASET_PATH = f"{os.environ.get('DATASETS_PATH')}/Datasets_adults/FaceScape"

subject_number = 1
expression = "1_neutral"

def getImagesPath(subject_number, expression="1_neutral"):
    return f"{DATASET_PATH}/{subject_number}/{expression}"

def determine_photo_orientation_opencv(R):
    # Convert rotation matrix to rotation vector
    R = R[:, :3]
    rot_vec, _ = cv2.Rodrigues(R)
    rot_vec = rot_vec.flatten()
    
    # Normalize the rotation vector to get the camera's forward direction
    forward_direction = rot_vec / np.linalg.norm(rot_vec)
    
    # Reference frontal direction vector in the world coordinate system for OpenCV (positive Z-axis)
    frontal_direction = np.array([0, 0, 1])
    
    # Calculate the cosine of the angle between the camera's forward and the world's frontal direction
    cos_angle = np.dot(forward_direction, frontal_direction)
    
    # Calculate the angle in degrees
    angle = np.arccos(cos_angle) * (180 / np.pi)
    
    print(angle)
    # Determine orientation based on angle
    if angle < 45:
        return "Frontal"
    elif angle > 135:
        return "Rear"
    else:
        return "Lateral"


def load_parameters(path):
    with open(f"{path}/params.json", 'r') as f:
        params = json.load(f) # read parameters
        return params
    
def get_extrinsic_matrix(img_number, parameters):
    return np.array(parameters['%d_Rt' % img_number])

parameters = load_parameters(getImagesPath(subject_number, expression))

outcome = determine_photo_orientation_opencv(get_extrinsic_matrix(24, parameters))

print(outcome)

