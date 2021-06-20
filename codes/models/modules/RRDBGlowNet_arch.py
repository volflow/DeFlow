import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
import numpy as np
from models.modules.RRDBNet_arch import RRDBNet
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
import models.modules.thops as thops
import models.modules.flow as flow
import models.modules.flow_utils as flow_utils
import utils.debug as debug
from utils.LogDict import LogDict
from utils.util import opt_get


class RRDBGlowNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None):
        super(RRDBGlowNet, self).__init__()

        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])
        self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])
        hidden_channels = hidden_channels or 64
        self.RRDB_training = True  # Default is true

        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        set_RRDB_to_train = opt['network_G']['train_RRDB'] is not None and opt['network_G']['train_RRDB'] \
                            or train_RRDB_delay is not None and \
                            step > int(train_RRDB_delay * self.opt['train']['niter'])
        if set_RRDB_to_train:
            self.set_rrdb_training(True)

        self.flowUpsamplerNet = \
            FlowUpsamplerNet((160, 160, 3), hidden_channels, K,
                             flow_coupling=opt['network_G']['flow']['coupling'],
                             LU_decomposed=opt_get(self.opt, ['network_G', 'flow', 'LU_decomposed'], False),
                             opt=opt)

        # calculate C
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
            C = self.flowUpsamplerNet.C
        else:
            L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** (L - 3)
            C = 3 * 8 * 8 * fac * fac

        self.I = torch.nn.Parameter(torch.eye(C, requires_grad=False), requires_grad=False)

        # domX class is N(0,1)
        # domY class is N(mean_shift, cov_shift @ cov_shift^T)
        self.mean_shift = torch.nn.Parameter(torch.zeros(C,requires_grad=True), requires_grad=True) 
        std_init_shift = opt_get(self.opt, ['network_G', 'flow', 'shift', 'std_init_shift'], 1.0)
        self.cov_shift = torch.nn.Parameter(torch.eye(C,requires_grad=True) * std_init_shift, 
                                            requires_grad=True)

        self.register_parameter(f"mean_shift", self.mean_shift)
        self.register_parameter(f"cov_shift", self.cov_shift)

    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                p.requires_grad = trainable
            self.RRDB_training = trainable
            return True
        return False

    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None, reverse_with_grad=False, lr_enc=None,
                add_gt_noise=True, step=None, y_label=None):
                
        if not reverse:
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step, y_onehot=y_label)
        else:
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True, step=None):

        if opt_get(self.opt, ['network_G', 'flow', 'unconditional'], False):
            # zero out conditional features to emulate unconditional model
            lr = torch.zeros_like(lr)

        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise: # add quantization noise      
            # Setup
            noiseQuant = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseQuant'], True)
            noiseUniform = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseUniform'], -1)
            noiseGauss = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseGauss'], -1)
            noiseGaussLinStart = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseGaussLinStart'], -1)
            noiseGaussLinEnd = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseGaussLinEnd'], -1)
            niter = opt_get(self.opt, ['train', 'niter'], -1)

            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            elif noiseUniform > 0:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) * noiseUniform)
            elif noiseGauss > 0:
                z = z + torch.Tensor(z.shape).normal_(mean=0, std=noiseGauss).to(z.device)
            elif noiseGaussLinStart > 0:
                assert niter > 0, niter
                assert noiseGaussLinEnd > 0, noiseGaussLinEnd
                std = (noiseGaussLinEnd - noiseGaussLinStart) * (1 - step/niter) + noiseGaussLinEnd
                print(step, niter, noiseGaussLinStart, noiseGaussLinStart, std)
                z = z + torch.Tensor(z.shape).normal_(mean=0, std=std).to(z.device)

            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=epses, y_onehot=y_onehot)

        objective = logdet.clone()
        
        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses

        # Prior on epses[-1] other priors applied in self.flowUpsamplerNet and are alrady included in logdet
        if type(y_onehot) != torch.Tensor:
            y_onehot = torch.Tensor(y_onehot)
        
        assert z.shape[0] == y_onehot.shape[0], 'need one class label per datapoint'
        assert len(y_onehot.shape) == 1, 'labels must be one dimensional'
        
        domX = y_onehot == 0
        domY = y_onehot == 1

        cov_shifted = self.I + torch.matmul(self.cov_shift, self.cov_shift.T)
        mean_shifted = self.mean_shift

        ll = torch.zeros(z.shape[0], device=z.get_device() if z.get_device() >= 0 else None)

        ll[domX] = flow.Gaussian.logp(None, None, z[domX])
        ll[domY] = flow.Gaussian.logp(mean=mean_shifted, cov=cov_shifted, x=z[domY])

        objective = objective + ll


        nll = (-objective) / float(np.log(2.) * pixels)

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True):
        # decode
        if z is None and epses is None: # sample z
            # calculate size of z
            if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
                batch_size = lr.shape[0]
                C = self.flowUpsamplerNet.C
                H = int(self.opt['scale'] * lr.shape[2] // self.flowUpsamplerNet.scaleH)
                W = int(self.opt['scale'] * lr.shape[3] // self.flowUpsamplerNet.scaleW)
                
            else:
                L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
                fac = 2 ** (L - 3)
                C = 3 * 8 * 8 * fac * fac
                H = W = int(lr.shape[2] // (2 ** (L - 3))) # is this correct?

            shape = (batch_size, C, H, W)

            if eps_std is None:
                eps_std = 1
            assert eps_std == 1, "sampling with eps_std != 1 does not make sense for DeFlow"

            z = flow.GaussianDiag.sample_eps(shape, eps_std).to(lr.device)

            if type(y_onehot) != torch.Tensor:
                y_onehot = torch.Tensor(y_onehot)
            
            assert z.shape[0] == y_onehot.shape[0], 'need one class label per datapoint'
            assert len(y_onehot.shape) == 1, 'labels must be one dimesntional'
            
            domX = y_onehot == 0
            domY = y_onehot == 1

            if domY.any(): # sample and add u
                z_noise = flow.GaussianDiag.sample_eps(shape, eps_std)[domY].to(lr.device)
                z[domY] = (z[domY] + self.mean_shift.reshape(1,self.mean_shift.shape[0],1,1) 
                                   + torch.matmul(self.cov_shift, z_noise.T).T)
            
        # Setup
        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.opt['scale'] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet, y_onehot=y_onehot)
        return x, logdet


    def rrdbPreprocessing(self, lr):
        if opt_get(self.opt, ['network_G', 'flow', 'LR_noise_std'], 0) > 0:
            lr = lr + torch.Tensor(lr.shape).normal_(
                    mean=0, 
                    std=opt_get(self.opt, ['network_G', 'flow', 'LR_noise_std'], 0)
                ).to(lr.device)

        if opt_get(self.opt, ['network_G', 'flow', 'unconditional'], False):
            lr = torch.zeros_like(lr)

        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        if len(block_idxs) > 0:
            concat = torch.cat([rrdbResults["block_{}".format(idx)] for idx in block_idxs], dim=1)

            if opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'concat']) or False:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdbResults.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdbResults.keys():
                    keys.append('fea_up-1')
                if self.opt['scale'] >= 8:
                    keys.append('fea_up8')
                if self.opt['scale'] == 16:
                    keys.append('fea_up16')
                for k in keys:
                    h = rrdbResults[k].shape[2]
                    w = rrdbResults[k].shape[3]
                    rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdbResults
