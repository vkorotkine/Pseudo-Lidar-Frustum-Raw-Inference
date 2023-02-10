import argparse
import os

import numpy as np
import scipy.misc as ssc
import fr_kitti_inference_lib.pseudo_lidar.preprocessing.kitti_util as kitti_util
from fr_kitti_inference_lib.pseudo_lidar.lidar_generation import (
    project_depth_to_points,
    project_disp_to_points,
)
from tqdm import tqdm
import pykitti
import imageio
from fr_kitti_inference_lib.pseudo_lidar.lidar_generation import lidar_from_disp

if __name__ == "__main__":

    is_depth = False
    basedir = "./dataset/KITTI/raw/"

    date = "2011_09_26"

    drive_list = [
        # "0001",
        # "0002",
        # "0005",
        "0009",
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

    for drive in drive_list:
        dataset: pykitti.raw = pykitti.raw(basedir, date, drive)
        calib = kitti_util.CalibrationRaw(dataset)
        disparity_dir = f"./results/predict_disparity/raw/{date}/{drive}"
        savedir = f"./dataset/KITTI/raw/pseudo-lidar_velodyne/{date}/{drive}"

        assert os.path.isdir(disparity_dir)

        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        disps = [x for x in os.listdir(disparity_dir) if x[-3:] == "npy"]
        disps = sorted(disps)

        for fn in tqdm(disps):
            disp_map = np.load(disparity_dir + "/" + fn)

            lidar = lidar_from_disp(disp_map, is_depth, calib)
            predix = fn[:-4]
            lidar.tofile("{}/{}.bin".format(savedir, predix))
