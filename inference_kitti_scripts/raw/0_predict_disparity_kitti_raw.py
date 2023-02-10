from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math

from fr_kitti_inference_lib.pseudo_lidar.psmnet.utils import preprocess
from fr_kitti_inference_lib.pseudo_lidar.psmnet.models import *
from fr_kitti_inference_lib.pseudo_lidar.psmnet.dataloader import KITTI_raw_loader as DA
from tqdm import tqdm

from fr_kitti_inference_lib.pseudo_lidar.disparity_prediction import predict_disparity

parser = argparse.ArgumentParser(description="PSMNet")
parser.add_argument(
    "--save_figure",
    default=True,
    action="store_true",
    help="if true, save the numpy file, not the png file",
)
args = parser.parse_args()


def main():

    date = "2011_09_26"

    datapath = "./dataset/KITTI/raw/"

    drive_list = [
        "0001",
        "0002",
        "0005",
        "0009",
        "0011",
        # "0013",
        "0014",
        "0015",
        "0017",
        "0018",
        "0048",
        "0051",
        "0056",
    ]

    for drive in drive_list:
        savepath = f"./results/predict_disparity/raw/{date}/{drive}"

        processed = preprocess.get_transform(augment=False)
        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        test_left_img, test_right_img = DA.dataloader(datapath, date, drive)
        for inx in tqdm(range(len(test_left_img))):
            img = predict_disparity(test_left_img[inx], test_right_img[inx])
            np.save(savepath + "/" + test_left_img[inx].split("/")[-1][:-4], img)


if __name__ == "__main__":
    main()
