from __future__ import print_function

import pickle as pickle
import fr_kitti_inference_lib.train.provider as provider

from fr_kitti_inference_lib.test import (
    test_from_rgb_detection,
)

NUM_POINT = 1024

type = "pseudo_lidar"
date = "2011_09_26"
drive_list = [
    "0001",
    "0002",
    "0005",
    # "0009",
    "0011",
    "0013",
    "0014",
    "0015",
    "0017",
    "0018",
    "0048",
    "0051",
    "0056",
]  # , "0009", "0011", "0013", "0014", "0015", "0017", "0018", "0048", "0051", "0056"]
# drive_list = ["0013"]  # , "0014", "0015", "0017", "0018", "0048", "0051", "0056"]

for type in ["pseudo_lidar"]:
    for drive in drive_list:
        output_dir = (
            f"results/detection_results/detection_results_{date}_{drive}_{type}"
        )
        output_filename = (
            "./dataset/KITTI/raw/object_predictions/{type}/{date}/{drive}.pickle"
        )
        frustum_data_path = f"./results/frustum_prepared_data/frustum_all_raw_{type}/{date}_{drive}.pickle"
        # Load Frustum Datasets.
        frustum_dataset = provider.FrustumDataset(
            npoints=NUM_POINT,
            split=None,
            rotate_to_center=True,
            overwritten_data_path=frustum_data_path,
            from_rgb_detection=True,
            one_hot=True,
        )

        test_from_rgb_detection(
            output_dir + ".pickle", output_dir, "raw", frustum_dataset
        )
