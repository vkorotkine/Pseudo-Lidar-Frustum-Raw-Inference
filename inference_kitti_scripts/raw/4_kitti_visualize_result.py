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

import pykitti
from matplotlib import pyplot as plt
from fr_kitti_inference_lib.kitti_utils.viz_utils import show_image_with_boxes


basedir = "./dataset/KITTI/raw/"

date = "2011_09_26"
drive_list = ["0002", "0005", "0013"]  # "0002", "0005"]
type_list = ["pseudo_lidar"]
basedir_video = f"./results/video_results/"

debug = False
for drive in drive_list:
    for type in type_list:
        results_dir = (
            f"./results/detection_results/detection_results_{date}_{drive}_{type}"
        )

        n_images = len(
            [
                entry
                for entry in os.listdir(results_dir)
                if os.path.isfile(os.path.join(results_dir, entry))
            ]
        )

        basedir_labels = results_dir

        dataset: pykitti.raw = pykitti.raw(basedir, date, drive)
        if not debug:
            img = np.array(dataset.get_cam2(0))
            height, width, layers = img.shape
            size = (width, height)
            out = cv2.VideoWriter(
                f"{basedir_video}detections_{date}_{drive}_{type}.avi",
                cv2.VideoWriter_fourcc(*"DIVX"),
                2,
                size,
            )

        for data_idx in range(n_images):

            label_filename = os.path.join(
                basedir_labels, f"{str(data_idx).zfill(10)}.txt"
            )

            objects_pseudo_lidar = utils.read_label(label_filename)

            img = np.array(dataset.get_cam2(data_idx))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width, img_channel = img.shape

            calib = utils.CalibrationRaw(dataset)

            img = img.copy()
            for lv1, obj in enumerate(objects_pseudo_lidar):
                #  print(obj)
                if obj.type == "DontCare":
                    continue
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                if box3d_pts_2d is not None:
                    img = utils.draw_projected_box3d(img, box3d_pts_2d)
            if not debug:
                out.write(img)

            if debug:
                cv2.imshow("Tracked Image", img)
                cv2.waitKey(0) & 0xFF
        if not debug:
            out.release()
