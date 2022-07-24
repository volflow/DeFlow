import os
import argparse
import sys

from tqdm import tqdm
import numpy as np
import imageio

sys.path.insert(0, '../codes')
from data.util import is_image_file, load_at_multiple_scales

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-source_dir', required=True, help='path to directory containing HR images')
    # parser.add_argument('-target_dir', required=True, help='path to target directory')
    # parser.add_argument('-scales', nargs='+',  type=int, default=[1, 4], help='scales to downsample to')
    # args = parser.parse_args()

    # source_dir = args.source_dir
    source_dir = 'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/validation/iphone/'
    target_dir = source_dir
    scales = [1, 4]
    scale_dirs = []
    for scale in scales:
        dir = os.path.join(target_dir, f'{scale}x')
        scale_dirs.append(dir)
        os.makedirs(dir, exist_ok=True)

    for fn in tqdm(sorted(os.listdir(source_dir))):
        if not is_image_file(fn):
            continue

        images = load_at_multiple_scales(source_dir + fn, scales=scales)

        for img, dir in zip(images, scale_dirs):
            imageio.imwrite(os.path.join(dir, fn), img)
