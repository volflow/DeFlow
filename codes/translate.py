import argparse
import os

from tqdm import tqdm
import torch
import numpy as np
import imageio

import options.options as option
import utils.util as util
from data.util import is_image_file, load_at_multiple_scales
from models import create_model
from validate import rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-opt', required=True, help='path to config .yml file')
    parser.add_argument('-model_path', required=True, help='path to model checkpoint (.pkl)')
    parser.add_argument('-source_dir', required=True, default=None, help='save directory of translated images')
    parser.add_argument('-out_dir', required=True, default=None, help='save directory of translated images')
    parser.add_argument('-source_domain', default="X", help='save directory of translated images')
    add_gt_noise = True # apply quantization noise
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = create_model(util.getPredictorOdict(os.path.join("confs/", args.opt), args.model_path))
    opt = option.parse("confs/" + args.opt, is_train=True)
    device = model.netG.module.mean_shift.device


    if 'normalize' in opt['datasets']['train']:
        assert args.source_domain == 'X', 'only source_domain X supported atm when normalization is applied (as done in the AIM-RWSR DeFlow model)'
        mean_noisy_hr = np.asarray(opt['datasets']['train']['normalize']['mean_noisy_hr'])/255
        std_noisy_hr = np.asarray(opt['datasets']['train']['normalize']['std_noisy_hr'])/255
        mean_clean_hr = np.asarray(opt['datasets']['train']['normalize']['mean_clean_hr'])/255
        std_clean_hr = np.asarray(opt['datasets']['train']['normalize']['std_clean_hr'])/255
        denormalize_domY = lambda x: (((x - mean_clean_hr)/std_clean_hr)*std_noisy_hr) + mean_noisy_hr
        
    else: 
        denormalize_domY = lambda x: x
    
    for fn in tqdm(sorted(os.listdir(args.source_dir))):
        if not is_image_file(fn):
            continue
        
        gt, lq = load_at_multiple_scales(os.path.join(args.source_dir,fn), scales=[1, opt['scale']], as_tensor=True)
        labels = torch.Tensor([0]) if args.source_domain == 'X' else torch.Tensor([1])
        lq, gt, labels = lq.to(device), gt.to(device), labels.to(device)

        # precomute conditional encoding s.t. conditional features are fixed!
        lr_enc = model.netG.module.rrdbPreprocessing(lq) 

        # endcode
        zs, nll = model.get_encode_z_and_nll(lq=lq, gt=gt, y_label=labels, lr_enc=lr_enc, add_gt_noise=add_gt_noise, epses=[])

        # translate latents zs and decode
        translated = model.get_translate_with_zs(zs=zs, lq=lq, source_labels=labels, lr_enc=lr_enc, heat=1.0)

        save_path = os.path.join(args.out_dir, fn)
        imageio.imwrite(save_path, rgb(translated, denormalize_domY))