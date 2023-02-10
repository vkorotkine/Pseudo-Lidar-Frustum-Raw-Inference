import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, date, sequence):

    # left_fold = "image_02/data/"
    # right_fold = "image_03/data/"

    # left_fold = "image_02/data/"
    # right_fold = "image_03/data/"

    left_fold = "image_02/data/"
    right_fold = "image_03/data/"

    # left_fold  = 'image_2/data/'
    # right_fold = 'image_3/data/'

    sequence_path = date + "/" + date + "_drive_" + sequence + "_sync/"
    # image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
    image = [img for img in os.listdir(filepath + sequence_path + left_fold)]
    image = sorted(image)

    left_test = [filepath + sequence_path + left_fold + img for img in image]
    right_test = [filepath + sequence_path + right_fold + img for img in image]

    return left_test, right_test
