import os
import sys
from os.path import join as pjoin
import json
from PIL import Image
import numpy as np
import pickle


# TODO to modify!
DATASET_PATH = '/data/chengyang/data/chengyang/akb48_all_annotated'

SAVE_PATH = './example_rendered'

VISU_SAVE_PATH = './visu'

ID_PATH = './akb48_all_id_list.txt'

TARGET_GAPARTS = [
    'line_fixed_handle', 'round_fixed_handle', 'slider_button', 'hinge_door', 'slider_drawer',
    'slider_lid', 'hinge_lid', 'hinge_knob', 'hinge_handle'
]

OBJECT_CATEGORIES = [
    'Box', 'TrashCan', 'Bucket', 'Drawer'
]

CAMERA_POSITION_RANGE = {
    'Box': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 0.35,
        'distance_max': 0.55
    }],
    'TrashCan': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 0.35,
        'distance_max': 0.55
    }],
    'Bucket': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 0.45,
        'distance_max': 0.6
    }],
    'Drawer': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 0.55,
        'distance_max': 0.8
    }]
}

BACKGROUND_RGB = np.array([216, 206, 189], dtype=np.uint8)
