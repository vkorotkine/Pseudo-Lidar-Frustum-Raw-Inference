""" Prepare KITTI data for 3D object detection.

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
import pickle as pickle
from fr_kitti_inference_lib.kitti_utils.kitti_object import *
import argparse

from tqdm import tqdm
import pykitti


def ms_coco_classnames_dict():
    dict = {
        0: "__background__",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        12: "stop sign",
        13: "parking meter",
        14: "bench",
        15: "bird",
        16: "cat",
        17: "dog",
        18: "horse",
        19: "sheep",
        20: "cow",
        21: "elephant",
        22: "bear",
        23: "zebra",
        24: "giraffe",
        25: "backpack",
        26: "umbrella",
        27: "handbag",
        28: "tie",
        29: "suitcase",
        30: "frisbee",
        31: "skis",
        32: "snowboard",
        33: "sports ball",
        34: "kite",
        35: "baseball bat",
        36: "baseball glove",
        37: "skateboard",
        38: "surfboard",
        39: "tennis racket",
        40: "bottle",
        41: "wine glass",
        42: "cup",
        43: "fork",
        44: "knife",
        45: "spoon",
        46: "bowl",
        47: "banana",
        48: "apple",
        49: "sandwich",
        50: "orange",
        51: "broccoli",
        52: "carrot",
        53: "hot dog",
        54: "pizza",
        55: "donut",
        56: "cake",
        57: "chair",
        58: "couch",
        59: "potted plant",
        60: "bed",
        61: "dining table",
        62: "toilet",
        63: "tv",
        64: "laptop",
        65: "mouse",
        66: "remote",
        67: "keyboard",
        68: "cell phone",
        69: "microwave",
        70: "oven",
        71: "toaster",
        72: "sink",
        73: "refrigerator",
        74: "book",
        75: "clock",
        76: "vase",
        77: "scissors",
        78: "teddy bear",
        79: "hair drier",
        80: "toothbrush",
    }

    return dict


def in_hull(p, hull):
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    """pc: (N,3), box3d: (8,3)"""
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    """pc: (N,2), box2d: (xmin,ymin,xmax,ymax)"""
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def random_shift_box2d(box2d, shift_ratio=0.1):
    """Randomly shift box center, randomly scale width and height"""
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    return np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])


def get_label_objects_raw(basedir, drive, date, idx):
    label_dir = basedir + f"2d_detections/{date}/{drive}/image_02/"
    label_filename = os.path.join(label_dir, str(idx).zfill(10) + ".txt")
    return utils.read_label(label_filename)


def get_lidar_raw(basedir, drive, date, idx, type="lidar"):
    if type == "pseudo_lidar":
        lidar_dir = basedir + f"pseudo-lidar_velodyne/{date}/{drive}/"
    if type == "lidar":
        lidar_dir = f"./dataset/KITTI/raw/{date}/2011_09_26_drive_{drive}_sync/velodyne_points/data"

    lidar_filename = os.path.join(lidar_dir, str(idx).zfill(10) + ".bin")
    return utils.load_velo_scan(lidar_filename)


def get_box3d_dim_statistics(idx_filename):
    """Collect and dump 3D bounding box statistics"""
    dataset = kitti_object(os.path.join(ROOT_DIR, "dataset/KITTI/object"))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print("------------- ", data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type == "DontCare":
                continue
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

    with open("box3d_dimensions.pickle", "wb") as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)


def read_det_file(det_filename):
    """Parse lines in 2D detection output files"""
    det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, "r"):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip(".png")))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list


# TODO: Spagehittificaiton is occuridng here. Need to separate out the object vs raw dataset at a higher level.
# These if statements are unaccepatble.
def extract_frustum_data_rgb_detection(
    output_filename,
    viz=False,
    type_whitelist=["Car"],
    img_height_threshold=25,
    lidar_point_threshold=5,
    date=None,
    drive=None,
    split="training",
    type=None,
    kitti_type="raw",
):
    """Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    """
    if kitti_type == "raw":
        basedir = "./dataset/KITTI/raw/"
        dataset: pykitti.raw = pykitti.raw(basedir, date, drive)
        det_filename = f"{basedir}/rgb_2d_detections_det_formatted/{date}_{drive}.txt"

    if kitti_type == "object":
        basedir = "./dataset/KITTI/object/"
        dataset: kitti_object = kitti_object(
            os.path.join("./dataset/KITTI/object"), split
        )
        det_filename = f"{basedir}rgb_2d_detections_det_formatted/{split}.txt"

    det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_file(
        det_filename
    )
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    for det_idx in tqdm(range(len(det_id_list))):
        data_idx = det_id_list[det_idx]
        # print('det idx: %d/%d, data idx: %d' % \
        #   (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            if kitti_type == "raw":
                calib = utils.CalibrationRaw(dataset)
                pc_velo = get_lidar_raw(basedir, drive, date, data_idx, type=type)
            if kitti_type == "object":
                calib = dataset.get_calibration(data_idx)
                pc_velo = dataset.get_lidar(data_idx, type=type)

            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]
            if kitti_type == "raw":
                img = np.array(dataset.get_cam2(data_idx))
            if kitti_type == "object":
                img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True
            )
            cache = [calib, pc_rect, pc_image_coord, img_fov_inds]
            cache_id = data_idx
        else:
            calib, pc_rect, pc_image_coord, img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist:
            continue

        # 2D BOX: Get pts rect backprojected
        xmin, ymin, xmax, ymax = det_box2d_list[det_idx]
        box_fov_inds = (
            (pc_image_coord[:, 0] < xmax)
            & (pc_image_coord[:, 0] >= xmin)
            & (pc_image_coord[:, 1] < ymax)
            & (pc_image_coord[:, 1] >= ymin)
        )
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(
            box2d_center_rect[0, 2], box2d_center_rect[0, 0]
        )

        # Pass objects that are too small
        if (
            ymax - ymin < img_height_threshold
            or len(pc_in_box_fov) < lidar_point_threshold
        ):
            continue

        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)

    with open(output_filename, "wb") as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)


def create_rgb_detection_file(date, drive, image_idx_list=None):
    # Create the det file the way rgb_detections likes it. Concatenates all the
    # object detections and throws them into one file.
    # This is kind of abominable but we are just trying to get it into the format this thing likes.
    base_dir_images_to_write_in_file = (
        "dataset/KITTI/raw/"
        + date
        + "/"
        + date
        + "_drive_"
        + drive
        + "_sync/image_02/data/"
    )
    bounding_box_dir = f"./dataset/KITTI/raw/2d_detections/{date}/{drive}/image_02/"
    output_dir = f"./dataset/KITTI/raw/rgb_2d_detections_det_formatted/"

    det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    det_str2id = dict((v, k) for k, v in det_id2str.items())

    n_images = len(
        [
            entry
            for entry in os.listdir(bounding_box_dir)
            if os.path.isfile(os.path.join(bounding_box_dir, entry))
        ]
    )

    list_of_strings = []
    if image_idx_list == None:
        image_idx_list = range(n_images)
    for image_idx in image_idx_list:
        label_filename = f"./dataset/KITTI/raw/2d_detections/{date}/{drive}/image_02/{str(image_idx).zfill(10)}.txt"
        image_filename = (
            f"{base_dir_images_to_write_in_file}{str(image_idx).zfill(10)}.png"
        )
        lines = [line.rstrip() for line in open(label_filename)]
        for label_line in lines:
            data = label_line.split(" ")

            data_to_write = [None for lv1 in range(8)]
            data_to_write[0] = image_filename
            data_to_write[1] = det_str2id[data[0]]
            data_to_write[2] = data[15]
            data_to_write[3:7] = [int(float(loc)) for loc in data[4:8]]
            data_to_write[7] = "\n"
            list_of_strings.append(" ".join(map(str, data_to_write)))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fname = f"{output_dir}{date}_{drive}.txt"
    f = open(fname, "w")
    f.writelines(list_of_strings)
    f.close()


# TODO: Lots of code duplication with above function. Fix...
def create_rgb_detection_file_object(test_train, image_idx_list=None):
    # Create the det file the way rgb_detections likes it. Concatenates all the
    # object detections and throws them into one file.
    # This is kind of abominable but we are just trying to get it into the format this thing likes.

    base_dir_images_to_write_in_file = f"dataset/KITTI/object/{test_train}/image_2"

    bounding_box_dir = f"./dataset/KITTI/object/2d_detections/{test_train}/image_2"
    output_dir = f"./dataset/KITTI/object/rgb_2d_detections_det_formatted/"

    det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    det_str2id = dict((v, k) for k, v in det_id2str.items())

    n_images = len(
        [
            entry
            for entry in os.listdir(bounding_box_dir)
            if os.path.isfile(os.path.join(bounding_box_dir, entry))
        ]
    )

    list_of_strings = []

    if image_idx_list is None:
        image_idx_list = range(n_images)

    for image_idx in image_idx_list:
        label_filename = f"./dataset/KITTI/object/2d_detections/{test_train}/image_2/{str(image_idx).zfill(10)}.txt"
        image_filename = (
            f"{base_dir_images_to_write_in_file}/{str(image_idx).zfill(6)}.png"
        )
        lines = [line.rstrip() for line in open(label_filename)]
        for label_line in lines:
            data = label_line.split(" ")

            data_to_write = [None for lv1 in range(8)]
            data_to_write[0] = image_filename
            data_to_write[1] = det_str2id[data[0]]
            data_to_write[2] = data[15]
            data_to_write[3:7] = [int(float(loc)) for loc in data[4:8]]
            data_to_write[7] = "\n"
            list_of_strings.append(" ".join(map(str, data_to_write)))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fname = f"{output_dir}/{test_train}.txt"
    f = open(fname, "w")
    f.writelines(list_of_strings)
    f.close()
