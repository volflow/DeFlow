import os.path as osp
import cv2
import os
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np
import torch

#### options
from utils.ImageSplitter import ImageSplitter

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []

    for data in test_loader:

        n_splits = opt.get('n_splits', 1)
        margin = opt.get('split_margin', 50)
        splits_LQ = ImageSplitter(data['LQ'], n_splits, margin=margin)
        splits_GT = ImageSplitter(data['GT'], n_splits, margin=margin)
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        img_name = data['LQ_path'][0]

        model.feed_data(data)

        nll = model.test()
        if nll is None:
            nll = 0
        print("nll: {}".format(nll))

        visuals = model.get_current_visuals()
        img_dir = dataset_dir

        sr_img = None
        # Save SR images for reference
        if hasattr(model, 'heats'):
            for heat in model.heats:
                for i in range(model.n_sample):
                    sr_img = util.tensor2img(visuals['SR', heat, i])  # uint8
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_h{:03d}_s{:d}.png'.format(img_name,
                                                                                        int(heat * 100), i))
                    util.save_img(sr_img, save_img_path)
        else:
            sr_img = util.tensor2img(visuals['SR'])  # uint8
            save_img_path = os.path.join(img_dir,
                                         '{:s}.png'.format(img_name))
            util.save_img(sr_img, save_img_path)
        assert sr_img is not None

        # Save LQ images for reference
        save_img_path_lq = os.path.join(img_dir,
                                        '{:s}_LQ.png'.format(img_name))
        if not os.path.isfile(save_img_path_lq):
            lq_img = util.tensor2img(visuals['LQ'])  # uint8
            util.save_img(
                cv2.resize(lq_img, dsize=None, fx=opt['scale'], fy=opt['scale'],
                           interpolation=cv2.INTER_NEAREST),
                save_img_path_lq)

        # Save GT images for reference
        gt_img = util.tensor2img(visuals['GT'])  # uint8
        save_img_path_gt = os.path.join(img_dir,
                                        '{:s}_GT.png'.format(img_name))
        if not os.path.isfile(save_img_path_gt):
            util.save_img(gt_img, save_img_path_gt)

        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]


        # calculate PSNR and SSIM
        if need_GT:
            gt_img = util.tensor2img(visuals['GT'])
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.

            crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            test_results['psnr'].append(psnr)

            logger.info('{:20s} - PSNR: {:.6f} dB.'.format(img_name, psnr))
        else:
            logger.info(img_name)

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info(
            '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB\n'.format(
                test_set_name, ave_psnr))
