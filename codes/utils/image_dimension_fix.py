from PIL import Image
import os


# use on 4x data
def fix_dimensions_4x(path, save_path):

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            im = Image.open(f)
            width, height = im.size
            if width/4 % 2 != 0:
                im = im.crop((0, 0, width - 4, height))
            if height/4 % 2 != 0:
                im = im.crop((0, 0, width, height - 4))
            save_f = os.path.join(save_path, filename)
            im.save(save_f, quality=95, subsampling=0)


# use on 16x data
def fix_dimensions_16x(path, save_path):

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            im = Image.open(f)
            width, height = im.size
            if width % 2 != 0:
                im = im.crop((0, 0, width - 1, height))
            if height % 2 != 0:
                im = im.crop((0, 0, width, height - 1))
            save_f = os.path.join(save_path, filename)
            im.save(save_f, quality=95, subsampling=0)


if __name__ == "__main__":

    path = '../../datasets/DPED-RO/DPEDiphone-val-y/16x/'
    save_path = '../../datasets/DPED-RO/DPEDiphone-val-y/16x_reshaped'

    fix_dimensions_16x(path, save_path)
