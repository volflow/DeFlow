import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.timer import Timer, TickTock
from utils.util import get_resume_paths, opt_get

import MeasureLib

import psutil


def getEnv(name): import os; return True if name in os.environ.keys() else False


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if not args.opt.endswith('.yml'):
        args.opt += '.yml'
    opt = option.parse(args.opt, is_train=True)

    #### check if gpus exist
    if not torch.cuda.is_available() and opt['gpu_ids'] is not None:
        print("Warning: CUDA detected but opt['gpu_ids'] was specified. Using cpu")
        opt['gpu_ids'] = None

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        resume_state_path, _ = get_resume_paths(opt)

        # distributed resuming: all load into default GPU
        if resume_state_path is None:
            resume_state = None
        elif torch.cuda.is_available():

            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_state_path,
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            option.check_resume(opt, resume_state['iter'])  # check resume options
        else:
            resume_state = torch.load(resume_state_path, map_location=torch.device('cpu'))
            option.check_resume(opt, resume_state['iter'])  # check resume options

    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt.get('use_tb_logger', False):# and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            conf_name = basename(args.opt).replace(".yml", "")
            exp_dir = opt['path']['experiments_root']
            log_dir_train = os.path.join(exp_dir, 'tb', conf_name, 'train')
            log_dir_valid = os.path.join(exp_dir, 'tb', conf_name, 'valid')
            tb_logger_train = SummaryWriter(log_dir=log_dir_train)
            tb_logger_valid = SummaryWriter(log_dir=log_dir_valid)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            print('Dataset created')
            train_size = int(math.floor(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            elif opt_get(opt,['datasets','train','balanced'], False):
                labels = np.asarray(train_set.y_labels)
                unique, counts = np.unique(labels, return_counts=True)
                print('balancing classes', dict(zip(unique,counts)))
                weights = np.zeros_like(labels).astype(float)
                for class_id, count in zip(unique, counts):
                    weights[labels==class_id] = len(weights)/count
                    print('class:', class_id, 'count:', count, 'total:', len(weights), 'weight:', len(weights)/count, 'real', weights[labels==class_id].mean())
                print('weights.shape', weights.shape)
                train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            else:
                train_sampler = None
            # 
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    current_step = 0 if resume_state is None else resume_state['iter']
    model = create_model(opt, current_step)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    timer = Timer()
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    timerData = TickTock()

    process = psutil.Process(os.getpid())

    ram = process.memory_info().rss / (2 ** 30)
    print('RAM usage:', ram)


    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        print('epoch', epoch)

        timerData.tick()
        for _, train_data in enumerate(train_loader):
            timerData.tock()
            
            current_step += 1
            if current_step <= total_iters:

                #### training
                model.feed_data(train_data)

                #### update learning rate
                model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
                nll = None
                
                nll = model.optimize_parameters(current_step)

                    

                if nll is None:
                    nll = 0

                #### log
                def eta(t_iter):
                    return (t_iter * (opt['train']['niter'] - current_step)) / 3600

                if current_step % opt['logger']['print_freq'] == 0 \
                        or current_step - (resume_state['iter'] if resume_state else 0) < 25:
                    avg_time = timer.get_average_and_reset()
                    avg_data_time = timerData.get_average_and_reset()
                    ram = process.memory_info().rss / (2 ** 30)
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, t:{:.2e}, td:{:.2e}, eta:{:.2e}, nll:{:.3e}, ram:{:.3f}> '.format(
                        epoch, current_step, model.get_current_learning_rate(), avg_time, avg_data_time,
                        eta(avg_time), nll, ram)
                    print(message)

                timer.tick()
                # Reduce number of logs
                if current_step % 5 == 0:
                    tb_logger_train.add_scalar('loss/nll', nll, current_step)
                    tb_logger_train.add_scalar('lr/base', model.get_current_learning_rate(), current_step)
                    tb_logger_train.add_scalar('time/iteration', timer.get_last_iteration(), current_step)
                    tb_logger_train.add_scalar('time/data', timerData.get_last_iteration(), current_step)
                    tb_logger_train.add_scalar('time/eta', eta(timer.get_last_iteration()), current_step)
                    for k, v in model.get_current_log().items():
                        tb_logger_train.add_scalar(k, v, current_step)

                #### save models and training states
                if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                    if rank <= 0:
                        logger.info('Saving models and training states.')
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)
                        # os.system(f"chmod -Rc 775 {os.path.join(opt['path']['val_images'],'/../')}")

            # validation
            if ((current_step % opt['train']['val_freq'] == 0) 
                    or current_step == total_iters + 1) and rank <= 0: # current_step == 1 or 
                flatten = lambda l: [item for sublist in l for item in sublist]
                labels = flatten(opt_get(opt, ['network_G', 'flow', 'shift', 'classes'], [[0,1]]))
                
                print("validating...")
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_lpips = 0.0

                label_psnr = {l: 0.0 for l in labels}
                label_ssim = {l: 0.0 for l in labels}
                label_lpips = {l: 0.0 for l in labels} 
                label_nll = {l: 0.0 for l in labels}                    

                label_occurences = {l: 0 for l in labels} # only tracks occurences for idx < n_visual
                label_occurences_all = {l: 0 for l in labels}

                nlls = []
                epses_dict = {}
                n_visual = 20

                for idx, val_data in enumerate(val_loader):
                    model.feed_data(val_data)

                    nll, epses, y_label = model.test()
                    if nll is None:
                        nll = 0
                    nlls.append(nll)
                    label_nll[y_label[0].item()] += nll
                    label_occurences_all[y_label[0].item()] += 1

                    # store epses for statisitcs
                    if opt_get(opt, ['datasets', 'val', 'dataroot_y_labels'], False):
                        if type(epses) != list:
                            epses = [epses]

                        epses = [eps.cpu().numpy() for eps in epses]

                        # need seperate statistics for each class !!!
                        epses_dict.setdefault(y_label[0].item(), [])
                        epses_dict[y_label[0].item()].append(epses)
                    
                    if idx < n_visual:
                        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        for y_label_target in labels:
                            visuals = model.get_current_visuals(y_label=[y_label_target])

                            sr_img = None
                            # Save SR images for reference
                            if hasattr(model, 'heats'):
                                for heat in model.heats:
                                    for i in range(model.n_sample):
                                        sr_img = util.tensor2img(visuals['SR', heat, y_label_target, i])  # uint8
                                        save_img_path = os.path.join(img_dir,
                                                                    '{:s}_{:09d}_h{:03d}_source{:d}_target{:d}_s{:d}.png'.format(img_name,
                                                                                                            current_step,
                                                                                                            int(heat * 100),
                                                                                                            y_label[0], 
                                                                                                            y_label_target,
                                                                                                            i))
                                        util.save_img(sr_img, save_img_path)
                                        print("saving:", save_img_path)
                            else:
                                sr_img = util.tensor2img(visuals['SR'])  # uint8
                                save_img_path = os.path.join(img_dir,
                                                            '{:s}_{:d}_source{:d}_target{:d}.png'.format(img_name, current_step, y_label[0], y_label_target))
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
                    
                    # calculate PSNR
                    sr_img = util.tensor2img(
                        model.get_sr_with_z(lq=val_data['LQ'], z=None, heat=1.0, y_label=y_label)
                    )
                    gt_img = util.tensor2img(val_data['GT']) 

                    crop_size = opt['scale']
                    gt_img = gt_img / 255. # seems redundant as we multiply by 255 again...
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

                    psnr, ssim, lpips = MeasureLib.measure((cropped_gt_img*255).round().astype('uint8'), (np.clip(cropped_sr_img,0,1)*255).round().astype('uint8'))

                    avg_psnr += psnr
                    avg_ssim += ssim
                    avg_lpips += lpips

                    label_psnr[y_label[0].item()] += psnr
                    label_ssim[y_label[0].item()] += ssim
                    label_lpips[y_label[0].item()] += lpips

                    label_occurences[y_label[0].item()] += 1

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_lpips = avg_lpips / idx
                avg_nll = sum(nlls) / len(nlls)

                avg_label_psnr = {l: label_psnr[l]/label_occurences[l] for l in labels}
                avg_label_ssim = {l: label_ssim[l]/label_occurences[l] for l in labels}
                avg_label_lpips = {l: label_lpips[l]/label_occurences[l] for l in labels}
                avg_label_nll = {l: label_nll[l]/label_occurences_all[l] for l in labels}

                # log
                logger.info('# Validation # PSNR: {:.4e} # SSIM: {:.4e} # LPIPS: {:.4e} '.format(avg_psnr, avg_ssim, avg_lpips))
                logger.info(f'# Validation # Labelwise NLL:{str([(k, avg_label_nll[k]) for k in avg_label_nll])}')
                logger.info(f'# Validation # Labelwise PSNR:{str([(k, avg_label_psnr[k]) for k in avg_label_psnr])}')
                logger.info(f'# Validation # Labelwise SSIM:{str([(k, avg_label_ssim[k]) for k in avg_label_ssim])}')
                logger.info(f'# Validation # Labelwise LPIPS:{str([(k, avg_label_lpips[k]) for k in avg_label_lpips])}')

                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} | ssim: {:.4e} | lpips {:.4e}'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_lpips))

                # tensorboard logger
                tb_logger_valid.add_scalar('loss/psnr', avg_psnr, current_step)
                tb_logger_valid.add_scalar('loss/ssim', avg_ssim, current_step)
                tb_logger_valid.add_scalar('loss/lpips', avg_lpips, current_step)
                tb_logger_valid.add_scalar('loss/nll', avg_nll, current_step)

                for l in labels:
                    tb_logger_valid.add_scalar(f'loss/nll_label{l}', avg_label_nll[l], current_step)
                    tb_logger_valid.add_scalar(f'loss/psnr_label{l}', avg_label_psnr[l], current_step)
                    tb_logger_valid.add_scalar(f'loss/ssim_label{l}', avg_label_ssim[l], current_step)
                    tb_logger_valid.add_scalar(f'loss/lpips_label{l}', avg_label_lpips[l], current_step)                

                tb_logger_train.flush()
                tb_logger_valid.flush()

            if current_step >= total_iters:
                break
            
            timerData.tick()

        if current_step >= total_iters:
            break

    if rank <= 0:
        print(range(start_epoch, total_epochs + 1))
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')

 

if __name__ == '__main__':
    main()



