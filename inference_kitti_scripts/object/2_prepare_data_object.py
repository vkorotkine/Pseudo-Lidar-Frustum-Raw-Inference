from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import fr_kitti_inference_lib.kitti_utils.kitti_util as utils
import pickle as pickle
from fr_kitti_inference_lib.kitti_utils.kitti_object import *
import argparse

from tqdm import tqdm
import pykitti

from fr_kitti_inference_lib.data_preparation import (
    ms_coco_classnames_dict,
    extract_frustum_data_rgb_detection,
    create_rgb_detection_file_object,
)


class_dict = ms_coco_classnames_dict()

type_whitelist = ["Car", "Pedestrian", "Cyclist"] + [
    class_dict[lv1] for lv1 in range(80)
]

type = "pseudo_lidar"
rgb_or_normal = "rgb"


base_dir = "./results/frustum_prepared_data/"
test_train = "testing"
for type in ["lidar", "pseudo_lidar"]:
    file_dir = os.path.join(base_dir, f"frustum_all_object_{type}_{rgb_or_normal}")
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    create_rgb_detection_file_object(test_train)
    extract_frustum_data_rgb_detection(
        output_filename=os.path.join(file_dir, f"{test_train}.pickle"),
        viz=False,
        type_whitelist=type_whitelist,
        type=type,
        kitti_type="object",
    )
