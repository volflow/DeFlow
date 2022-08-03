
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import numpy as np
import PIL.Image as Image
import cv2
import os

im_path1 = 'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/canon/16x/'
im_path2 = 'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/iphone/4x/'

paths = ['D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/canon/1x/',
         'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/canon/4x/',
         'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/canon/16x/',
         'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/iphone/1x/',
         'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/iphone/4x/']

# im1 = cv2.imread(im_path1)
# im2 = cv2.imread(im_path2)


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    print(err)


def check_img_names(path):
    names =[]
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            names.append(filename)
    return names


def compare_img_lists(path1, path2):
    l1 = check_img_names(path1)
    l2 = check_img_names(path2)
    none_common = set(l1) ^ set(l2)
    return none_common


def delete_none_common(paths, differences):
    for path in paths:
        for filename in differences:
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                os.remove(f)


if __name__ == "__main__":
    difference_list = compare_img_lists(im_path1, im_path2)
    with open(r'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/comparison.txt', 'w') as fp:
        for item in difference_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done finding differences')

    delete_none_common(paths, difference_list)
    print("deleted differences")

    new_difference_list = compare_img_lists(im_path1, im_path2)
    print(new_difference_list)       # should be empty

    # mse(im1, im2)
    # s = ssim(im1, im2)
    # print(s)