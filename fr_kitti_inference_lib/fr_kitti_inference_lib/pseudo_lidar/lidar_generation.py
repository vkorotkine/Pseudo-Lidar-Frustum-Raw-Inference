import argparse
import os

import numpy as np
import scipy.misc as ssc
import fr_kitti_inference_lib.pseudo_lidar.preprocessing.kitti_util as kitti_util
from tqdm import tqdm
import pykitti
import imageio

import fr_kitti_inference_lib.pseudo_lidar.preprocessing.kitti_util as kitti_util


def project_disp_to_points(calib: kitti_util.Calibration, disp, max_high):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1.0 - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def lidar_from_disp(disp_map, is_depth, calib):
    parser = argparse.ArgumentParser(description="Generate Libar")
    parser.add_argument("--max_high", type=int, default=1)
    args = parser.parse_args()

    if not is_depth:
        disp_map = (disp_map * 256).astype(np.uint16) / 256.0
        lidar = project_disp_to_points(calib, disp_map, args.max_high)
    else:
        disp_map = (disp_map).astype(np.float32) / 256.0
        lidar = project_depth_to_points(calib, disp_map, args.max_high)
    # pad 1 in the indensity dimension
    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
    lidar = lidar.astype(np.float32)
    return lidar
