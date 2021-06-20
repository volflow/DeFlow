import torch
from torch import nn as nn

from models.modules import thops
from models.modules.FlowStep import FlowStep
from models.modules.flow import Conv2dZeros, GaussianDiag
from models.modules import flow
from utils.util import opt_get

import numpy as np


class Split2d(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super().__init__()

        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume

        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = opt_get(opt, ['network_G', 'flow', 'split', 'eps'],  logs_eps)
        self.position = position
        self.opt = opt

        C = self.num_channels_consume

        # parameters to model the domain shift
        # domain X is N(0,1)
        # domain Y is N(mean_shift, cov_shift @ cov_shift^T)
        self.I = torch.nn.Parameter(torch.eye(C, requires_grad=False), requires_grad=False)
        self.mean_shift = torch.nn.Parameter(torch.zeros(C,requires_grad=True), requires_grad=True) 
        std_init_shift = opt_get(self.opt, ['network_G', 'flow', 'shift', 'std_init_shift'], 1.0)
        self.cov_shift = torch.nn.Parameter(torch.eye(C,requires_grad=True) * std_init_shift, 
                                            requires_grad=True)

        self.register_parameter(f"mean_shift", self.mean_shift)
        self.register_parameter(f"cov_shift", self.cov_shift)


    def split2d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def exp_eps(self, logs):
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'tanh'], False):
            return torch.exp(torch.tanh(logs)) + self.logs_eps

        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, y_onehot=None):
        domX = y_onehot == 0
        domY = y_onehot == 1

        if type(y_onehot) != torch.Tensor:
            y_onehot = torch.Tensor(y_onehot)

        assert len(y_onehot.shape) == 1, 'labels must be one dimensional'

        if not reverse:
            self.input = input

            z1, z2 = self.split_ratio(input)

            self.z1 = z1
            self.z2 = z2

            mean, logs = self.split2d_prior(z1, ft)
            
            eps = (z2 - mean) / (self.exp_eps(logs) + 1e-6)

            assert eps.shape[0] == y_onehot.shape[0], 'need one class label per datapoint'

            cov_shifted = self.I + torch.matmul(self.cov_shift, self.cov_shift.T)
            mean_shifted = self.mean_shift

            ll = torch.zeros(eps.shape[0], device=eps.device)
            ll[domX] = flow.Gaussian.logp(None, None, eps[domX])
            ll[domY] = flow.Gaussian.logp(mean=mean_shifted, cov=cov_shifted, x=eps[domY])

            logdet = logdet + ll - logs.sum(dim=[1,2,3])

            return z1, logdet, eps
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1, ft)

            if eps is None: # sample eps
                eps = GaussianDiag.sample_eps(mean.shape, eps_std)
                eps = eps.to(mean.device)
                
                shape = mean.shape

                if domY.any(): # sample and add u
                    z_noise = flow.GaussianDiag.sample_eps(shape, eps_std)[domY].to(mean.device)
                    eps[domY] = (eps[domY] + self.mean_shift.reshape(1,self.mean_shift.shape[0],1,1) 
                                   + torch.matmul(self.cov_shift, z_noise.T).T)
                    
                
            else:
                eps = eps.to(mean.device)

            assert eps.shape[0] == y_onehot.shape[0], 'need one class label per datapoint'

            cov_shifted = self.I + torch.matmul(self.cov_shift, self.cov_shift.T)
            mean_shifted = self.mean_shift

            ll = torch.zeros(eps.shape[0], device=eps.device)

            ll[domX] = flow.Gaussian.logp(None, None, eps[domX])
            ll[domY] = flow.Gaussian.logp(mean=mean_shifted, cov=cov_shifted, x=eps[domY])

            z2 = mean + self.exp_eps(logs) * eps

            z = thops.cat_feature(z1, z2)

            logdet = logdet - ll + logs.sum(dim=[1,2,3])

            return z, logdet

    def split_ratio(self, input):
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2