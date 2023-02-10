""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import fr_kitti_inference_lib.kitti_utils.kitti_util as utils
from fr_kitti_inference_lib.kitti_utils.kitti_object import *


def show_image_with_boxes(
    img, objects, calib, show3d=False, box_color=(255, 255, 255), thickness=2
):
    """Show image with 2D bounding boxes"""
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == "DontCare":
            continue
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img2 = utils.draw_projected_box3d(
            img2, box3d_pts_2d, color=box_color, thickness=thickness
        )
    # Image.fromarray(img1).show()
    if show3d:
        Image.fromarray(img2).show()
    return img2


basedir = "./dataset/KITTI/object/"

type_list = ["pseudo_lidar"]


for type in type_list:

    results_dir = f"./results/detection_results/detection_results_testing_{type}"

    n_images = len(
        [
            entry
            for entry in os.listdir(results_dir)
            if os.path.isfile(os.path.join(results_dir, entry))
        ]
    )

    basedir_labels = results_dir

    dataset: kitti_object = kitti_object(
        os.path.join("./dataset/KITTI/object"), "testing"
    )
    for data_idx in range(n_images):
        try:
            label_filename = os.path.join(
                basedir_labels, f"{str(data_idx).zfill(6)}.txt"
            )

            objects_pseudo_lidar = utils.read_label(label_filename)

            img = np.array(dataset.get_image(data_idx))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width, img_channel = img.shape

            calib = dataset.get_calibration(data_idx)

            img = img.copy()
            for lv1, obj in enumerate(objects_pseudo_lidar):
                #  print(obj)
                if obj.type == "DontCare":
                    continue
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                if box3d_pts_2d is not None:
                    img = utils.draw_projected_box3d(img, box3d_pts_2d)

            cv2.imshow("Tracked Image", img)
            cv2.waitKey(0) & 0xFF

        except:
            continue
