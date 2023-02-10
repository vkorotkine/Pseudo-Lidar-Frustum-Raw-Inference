import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import pickle as pickle
from tqdm import tqdm
from fr_kitti_inference_lib.models.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import fr_kitti_inference_lib.train.provider as provider
from fr_kitti_inference_lib.train.train_util import get_batch

from fr_kitti_inference_lib.test import (
    get_session_and_ops,
    softmax,
    inference,
    write_detection_results,
    fill_files,
    test_from_rgb_detection,
)

NUM_POINT = 1024

type = "pseudo_lidar"
train_test = "testing"
from_rgb_detection = True
if from_rgb_detection:
    rgb_id = "rgb"
else:
    rgb_id = "normal"
for type in ["lidar", "pseudo_lidar"]:
    output_dir = f"./results/detection_results/detection_results_{train_test}_{type}"
    output_filename = (
        f"./dataset/KITTI/object/object_predictions/{train_test}_{type}.pickle"
    )
    frustum_data_path = f"./results/frustum_prepared_data/frustum_all_object_{type}_{rgb_id}/{train_test}.pickle"
    # Load Frustum Datasets.
    frustum_dataset = provider.FrustumDataset(
        npoints=NUM_POINT,
        split=None,
        rotate_to_center=True,
        overwritten_data_path=frustum_data_path,
        from_rgb_detection=from_rgb_detection,
        one_hot=True,
    )

    test_from_rgb_detection(
        output_dir + ".pickle", output_dir, "object", frustum_dataset
    )
