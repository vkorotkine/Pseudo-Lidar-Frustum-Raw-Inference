from __future__ import print_function

import os
import pickle as pickle
from fr_kitti_inference_lib.kitti_utils.kitti_object import *
from fr_kitti_inference_lib.data_preparation import (
    ms_coco_classnames_dict,
    extract_frustum_data_rgb_detection,
    create_rgb_detection_file,
)


class_dict = ms_coco_classnames_dict()

type_whitelist = ["Car", "Pedestrian", "Cyclist"]
date = "2011_09_26"
# drive = "0001"
drive_list = [
    "0001",
    "0002",
    "0005",
    # "0009",
    # "0011",
    # "0013",
    # "0014",
    # "0015",
    # "0017",
    # "0018",
    # "0048",
    # "0051",
    # "0056",
]


base_dir = "./results/frustum_prepared_data/"

for drive in drive_list:
    for type in ["pseudo_lidar"]:
        # for type in ["lidar"]:
        file_dir = os.path.join(base_dir, f"frustum_all_raw_{type}")
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)

        for drive in drive_list:
            create_rgb_detection_file(date, drive)
            extract_frustum_data_rgb_detection(
                output_filename=os.path.join(file_dir, f"{date}_{drive}.pickle"),
                viz=False,
                type_whitelist=type_whitelist,
                date=date,
                drive=drive,
                type=type,
            )
