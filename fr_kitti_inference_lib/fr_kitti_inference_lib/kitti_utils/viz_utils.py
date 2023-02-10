from __future__ import print_function
import numpy as np
import cv2
from PIL import Image
import fr_kitti_inference_lib.kitti_utils.kitti_util as utils


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
