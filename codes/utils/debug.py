import imageio
import traceback
import datetime
import numpy as np
import os
from inspect import getframeinfo, stack
import uuid

def time_stamp():
    return datetime.datetime.now().strftime('%y%m%d_%H%M%S_%f')


base_dir = "/tmp/debug_" + time_stamp() + '_' + str(uuid.uuid4())
os.makedirs(base_dir)

counter = {}


def p(s, freq=1):
    caller = getframeinfo(stack()[1][0])
    loc_string = "{}:{}".format(caller.filename, caller.lineno)
    if not loc_string in counter.keys():
        counter[loc_string] = 0

    if counter[loc_string] % freq == 0:
        print("{} - {}".format(loc_string, s))

    counter[loc_string] += 1


def imwrite(name, img):
    path = os.path.join(base_dir, name + "_" + time_stamp() + ".png")

    img_min = img.min()
    img_max = img.max()
    img_mean = img.mean()
    img_shape = img.shape

    if hasattr(img, 'detach'):
        img = img.detach()
    if hasattr(img, 'cpu'):
        img = img.cpu()
    if hasattr(img, 'numpy'):
        img = img.numpy()

    # Select first of Batch
    if len(img.shape) == 4:
        img = img[0]

    # RGB Images
    if img.shape[0] == 3:
        img = img.transpose([1, 2, 0])
    # Features
    elif img.shape[1] == img.shape[2]:
        n = img.shape[0]
        width = img.shape[1]
        height = img.shape[2]

        n_width = int(np.ceil(np.sqrt(n)))
        n_height = int(np.ceil(n / n_width))

        out = np.zeros((width * n_height, height * n_height))

        for i in range(n_height):
            for j in range(n_width):
                idx = i * n_height + j
                if idx >= n:
                    break
                height_beg = (i + 0) * height
                height_end = (i + 1) * height
                width_beg = (j + 0) * width
                width_end = (j + 1) * width
                out[height_beg:height_end, width_beg:width_end] = img[idx, :, :]

        img = out
    else:
        raise RuntimeError("Not recognized: {}".format(img.shape))

    try:
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        imageio.imwrite(path, img)
    except:
        traceback.print_exc()

    caller = getframeinfo(stack()[1][0])
    print("{}:{} - {}, mean:{:.2E}, [{:.2E},{:.2E}] {}".
          format(caller.filename, caller.lineno,
                 img_shape, img_mean, img_min, img_max, path))
