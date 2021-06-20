'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
import imageio

def imread(path):
    img = imageio.imread(path)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=2)
    assert img.shape[2] == 3

    return np.transpose(np.asarray(img), [2, 0, 1])

def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt.get('phase', 'test')
    if phase == 'train':
        gpu_ids = opt.get('gpu_ids', None)
        gpu_ids = gpu_ids if gpu_ids else []
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
        else:
            num_workers = dataset_opt['n_workers'] * len(gpu_ids)
            batch_size = dataset_opt['batch_size']
            shuffle = True
        if sampler is not None:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    elif mode == 'LQGT_multiclass':
        from data.LQGT_multiclass_dataset import LQGTMulticlassDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

def imread(path):
    img = imageio.imread(path)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=2)
    assert img.shape[2] == 3

    return np.transpose(np.asarray(img), [2, 0, 1])

#### Crop and augmentation functions
import numpy as np
def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    return img, seg


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    return img, seg


def random_crop(hr, lr, size_hr, scale, random, alignment=None):

    # draw upper right starting edge in lr image
    size_lr = size_hr // scale
    size_lr_x = lr.shape[1]
    size_lr_y = lr.shape[2]
    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # align drawn starting edge to specified alignment
    # no specified alignemnt uses the natural alignemnt to scale
    if alignment is not None:
        assert (alignment % scale) == 0, f'alignment {alignment} must be multiple of scale {scale}'
        #assert (size_hr % alignment) == 0, f'alignment {alignment} must be divisor of size_hr {size_hr}'

        lr_alignment = alignment // scale
        start_x_lr = (start_x_lr // lr_alignment) * lr_alignment
        start_y_lr = (start_y_lr // lr_alignment) * lr_alignment

    # LR Patch
    lr_patch = lr[:, start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr]

    # HR Patch
    start_x_hr = start_x_lr * scale
    start_y_hr = start_y_lr * scale
    hr_patch = hr[:, start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr]

    return hr_patch, lr_patch


def center_crop(img, size, alignment=None):
    border_x = max(0,(img.shape[1] - size) // 2)
    border_y = max(0,(img.shape[2] - size) // 2)

    if alignment is not None:
        border_x = (border_x // alignment) * alignment
        border_y = (border_y // alignment) * alignment

    crop = img[:, border_x:border_x+size, border_y:border_y+size]

    return crop

def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]