import numpy as np
import torch
import torch.utils.data as data
import data.util as util
from data import random_crop, center_crop, random_flip, random_rotation, imread
import matlablike_resize

def getEnv(name): import os; return True if name in os.environ.keys() else False

class LQGTMulticlassDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTMulticlassDataset, self).__init__()
        self.opt = opt
        self.crop_size = opt.get("GT_size", None)
        self.alignment = opt.get("alignment", None) # aligns random crops
        self.target_scale = opt.get("target_scale", None)
        self.scale = None
        self.random_scale_list = [1]

        if 'normalize' in opt:
            self.normalize = True
            self.mean_clean_hr = torch.tensor(opt['normalize']['mean_clean_hr']).reshape(3,1,1)/255
            self.mean_noisy_hr = torch.tensor(opt['normalize']['mean_noisy_hr']).reshape(3,1,1)/255
            self.std_clean_hr = torch.tensor(opt['normalize']['std_clean_hr']).reshape(3,1,1)/255
            self.std_noisy_hr = torch.tensor(opt['normalize']['std_noisy_hr']).reshape(3,1,1)/255

            self.mean_clean_lr = torch.tensor(opt['normalize']['mean_clean_lr']).reshape(3,1,1)/255
            self.mean_noisy_lr = torch.tensor(opt['normalize']['mean_noisy_lr']).reshape(3,1,1)/255
            self.std_clean_lr = torch.tensor(opt['normalize']['std_clean_lr']).reshape(3,1,1)/255
            self.std_noisy_lr = torch.tensor(opt['normalize']['std_noisy_lr']).reshape(3,1,1)/255
        else:
            self.normalize = False

        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        dataroot_GT = opt['dataroot_GT']
        dataroot_LQ = opt['dataroot_LQ']

        assert type(dataroot_GT) == list
        assert type(dataroot_LQ) == list

        gpu = True
        augment = True

        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        # y_labels_file_path  = opt['dataroot_y_labels']

        self.dataroot_GT = dataroot_GT
        self.dataroot_LQ = dataroot_LQ

        self.paths_GT, self.sizes_GT = [], []
        self.paths_LQ, self.sizes_LQ = [], []

        self.y_labels = []

        for class_i, (root_GT, root_LQ) in enumerate(zip(dataroot_GT, dataroot_LQ)):
            paths_GT, sizes_GT = util.get_image_paths(self.data_type, root_GT)
            paths_LQ, sizes_LQ = util.get_image_paths(self.data_type, root_LQ)
            
            assert paths_GT, 'Error: GT path is empty.'
            assert paths_LQ, 'Error: LQ path is empty on the fly downsampling not yet supported.'
            # print(len(paths_GT), self.paths_GT[:10])
            if paths_LQ and paths_GT:
                assert len(paths_LQ) == len(paths_GT), 'GT and LQ datasets have different number of images - LQ: {}, GT: {}'.format(len(paths_LQ), len(paths_GT))

            # limit to n_max images per class if specified
            n_max = opt.get('n_max', None)
            if n_max is not None:
                if n_max > 0:
                    paths_GT = paths_GT[:n_max]
                    paths_LQ = paths_LQ[:n_max]
                elif n_max < 0:
                    paths_GT = paths_GT[n_max:]
                    paths_LQ = paths_LQ[n_max:]
                else:
                    raise RuntimeError("Not implemented")

            self.paths_GT += paths_GT
            if sizes_GT is not None:
                self.sizes_GT += sizes_GT
            self.paths_LQ += paths_LQ
            if sizes_LQ is not None:
                self.sizes_LQ += sizes_LQ

            self.y_labels += [class_i]*len(paths_GT)
        
        if not self.sizes_GT:
            self.sizes_GT = None
        if not self.sizes_LQ:
            self.sizes_LQ = None

        assert self.paths_GT, 'Error: GT path is empty.'
        # print(len(self.paths_GT), self.paths_GT[:10])
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

        # preload data to RAM
        if opt.get('preload', True):
            print('preloading data...')
            self.preloaded = True
            self.GT_imgs = []
            
            # get GT image
            for GT_path in self.paths_GT:
                self.GT_imgs.append(imread(GT_path))
            
            if self.paths_LQ:
                self.LQ_imgs = []
                for LQ_path in self.paths_LQ:
                    self.LQ_imgs.append(imread(LQ_path))
        else: 
            self.preloaded = False

        self.gpu = gpu
        self.augment = augment

        self.measures = None

        self.label_hist = {}

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()

        # get GT image
        GT_path = self.paths_GT[index]
        if self.preloaded: # use preloaded images
            hr = self.GT_imgs[index]
        else:
            hr = imread(GT_path)

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.preloaded: # use preloaded images
                lr = self.LQ_imgs[index]
            else:
                lr = imread(LQ_path)
        else:  # downsampling on-the-fly
            LQ_path = None
            raise RuntimeError("on-the-fly downsampling not implemented")

        if self.scale == None:
            self.scale = hr.shape[1] // lr.shape[1]
        
        assert hr.shape[1] == self.scale * lr.shape[1], ('non-fractional ratio or incorrect scale', lr.shape, hr.shape)

        if self.alignment is None:
            self.alignment = self.target_scale if self.target_scale else self.scale

        if self.use_crop:
            hr, lr = random_crop(hr, lr, self.crop_size, self.scale, self.use_crop, alignment=self.alignment)

        if self.center_crop_hr_size:
            hr, lr = center_crop(hr, self.center_crop_hr_size, alignment=self.alignment), center_crop(lr, self.center_crop_hr_size//self.scale, alignment=self.alignment//self.scale)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        if self.target_scale is not None and self.scale != self.target_scale:
            # make size of lr is divible by recaling factor
            # otherwise matlablike_resize gives non-multiple sizes
            rescale = self.target_scale//self.scale
            lr = lr[:, :(lr.shape[1]//rescale)*rescale, :(lr.shape[2]//rescale)*rescale]
            hr = hr[:, :(hr.shape[1]//self.target_scale)*self.target_scale, :(hr.shape[2]//self.target_scale)*self.target_scale]

            # use matlablike_resize for consitency
            lr = np.transpose(lr, [1, 2, 0])
            lr = matlablike_resize.imresize(lr, 1/rescale)
            lr = np.transpose(lr, [2, 0, 1])

            assert hr.shape[1] == self.target_scale * lr.shape[1], ('non-fractional ratio', lr.shape, hr.shape)

        hr = hr / 255.0
        lr = lr / 255.0

        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)

        if self.y_labels is not None:
            y_label = self.y_labels[index]

            if self.normalize and y_label == 1:
                hr = torch.clamp(((hr - self.mean_noisy_hr)/self.std_noisy_hr)*self.std_clean_hr + self.mean_clean_hr, 0, 1)
                lr = torch.clamp(((lr - self.mean_noisy_lr)/self.std_noisy_lr)*self.std_clean_lr + self.mean_clean_lr, 0, 1)

            return {'LQ': lr, 
                    'GT': hr, 
                    'y_label': y_label,
                    'LQ_path': LQ_path, 
                    'GT_path': GT_path,
                }

        return {'LQ': lr, 'GT': hr, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)

