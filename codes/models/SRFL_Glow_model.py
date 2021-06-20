import logging
from collections import OrderedDict

from torch._C import device, dtype

from utils.util import get_resume_paths, opt_get

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
import models.modules.Split
from .base_model import BaseModel

logger = logging.getLogger('base')


class SRFLGLOWModel(BaseModel):
    def __init__(self, opt, step):
        super(SRFLGLOWModel, self).__init__(opt)
        self.opt = opt

        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        self.hr_size = opt_get(opt, ['datasets', 'train', 'center_crop_hr_size'])
        self.hr_size = 160 if self.hr_size is None else self.hr_size
        self.lr_size = self.hr_size // opt['scale']

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()

        self.count_nan = 0
        self.count_nan_max = 100

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):
        # optimizers
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params_RRDB = []
        optim_params_other = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    optim_params_RRDB.append(v)
                    # print('opt', k)
                else:
                    optim_params_other.append(v)
                # if self.rank <= 0: # I don't think this is the case
                #     logger.warning('Params [{:s}] will not optimize.'.format(k))

        print('rrdb params', len(optim_params_RRDB))

        self.optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": train_opt['lr_G'], 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
                {"params": optim_params_RRDB, "lr": train_opt.get('lr_RRDB', train_opt['lr_G']),
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G}
            ],
        )

        self.optimizers.append(self.optimizer_G)
        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state'],
                                                     lr_steps_invese=train_opt.get('lr_steps_inverse', [])))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def add_optimizer_and_scheduler_RRDB(self, train_opt):
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    self.optimizer_G.param_groups[1]['params'].append(v)
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
        
        self.y_label = data['y_label'] if 'y_label' in data else None

    def optimize_parameters(self, step):

        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        if train_RRDB_delay is not None and step > int(train_RRDB_delay * self.opt['train']['niter']) \
                and not self.netG.module.RRDB_training:
            if self.netG.module.set_rrdb_training(True):
                self.add_optimizer_and_scheduler_RRDB(self.opt['train'])

        # self.print_rrdb_state()

        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()

        losses = {}
        weight_fl = opt_get(self.opt, ['train', 'weight_fl'])
        weight_fl = 1 if weight_fl is None else weight_fl
        if weight_fl > 0:
            _, nll, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False, y_label=self.y_label)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl

        total_loss = sum(losses.values())
        total_loss.backward()

        found_nan = False
        if not torch.isfinite(total_loss):
            found_nan = True
            print("Nan in the loss in step {}/{}".format(self.count_nan + 1, self.count_nan_max))
        else: # check if any gradient is Nan; slows down training significantly (about 50%)
            pass 
            # for p in self.netG.parameters():
            #     if p.grad is not None and not torch.isfinite(p.grad).all():
            #         found_nan = True
            #         print("Nan in the grad in step {}/{}".format(self.count_nan + 1, self.count_nan_max))
            #         print("in layer", p)
            #         break

        if found_nan:
            self.count_nan += 1
        else:
            self.optimizer_G.step()
            self.count_nan = 0

        if self.count_nan > self.count_nan_max:
            raise RuntimeError("Too many NAN during training")

        mean = total_loss.item()
        return mean

    def print_rrdb_state(self):
        for name, param in self.netG.module.named_parameters():
            if "RRDB.conv_first.weight" in name:
                print(name, param.requires_grad, param.data.abs().sum())
        print('params', [len(p['params']) for p in self.optimizer_G.param_groups])

    def test(self):
        self.netG.eval()

        with torch.no_grad():
            epses, nll, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False, y_label=self.y_label, epses=[])
        self.netG.train()
        return nll.mean().item(), epses, self.y_label

    def get_encode_nll(self, lq, gt, y_label=None):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, y_label=y_label)
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None, y_label=None, add_gt_noise=True, lr_enc=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses, y_label=y_label, add_gt_noise=add_gt_noise, lr_enc=lr_enc)[0]

    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True, y_label=None, lr_enc=None):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise, y_label=y_label, lr_enc=lr_enc)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, gt, epses=None, y_label=None, add_gt_noise=True, lr_enc=None):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, y_label=y_label, add_gt_noise=add_gt_noise, lr_enc=lr_enc)
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None, y_label=None, add_gt_noise=True, lr_enc=None):
        self.netG.eval()

        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses, y_label=y_label, add_gt_noise=add_gt_noise, lr_enc=lr_enc)
        self.netG.train()
        return sr


    def get_shift_dists(self):    
        def stats_from_param_dict(param_dict):        
            mean_u = param_dict['mean_shift']
            std_u = param_dict['cov_shift']
            
            return mean_u, std_u

        dists = []
        # get eps stats from all split layers
        for i, layer in enumerate(self.netG.module.flowUpsamplerNet.layers[::-1]): 
            if type(layer) != models.modules.Split.Split2d:
                continue
            param_dict = layer._parameters
            dists.append(stats_from_param_dict(param_dict))
        
        # get eps stats of the last level
        param_dict = self.netG.module._parameters
        dists.append(stats_from_param_dict(param_dict))
        
        return dists

    def get_translate_with_zs(self, zs, lq, source_labels, lr_enc, heat=1.0):
        # heat=None, seed=None, z=None, epses=None, y_label=None, add_gt_noise=True, lr_enc=None):
        def translate_z(z, mean_u, std_u, source_labels, heat=1.0):
            """     
            Given: eps ~ N(0, I)
            Samples: u ~ N(m_u, cov_u)
            Returns: eps + (u * heat)
            """
            from models.modules import flow
            z_noise = flow.GaussianDiag.sample_eps(z.shape, 1.0).to(z.device)
            z_shifted = torch.empty_like(z)
            #############################
            # formward translation
            #############################
            domX = source_labels == 0
            if domX.any():
                # import numpy as np
                # shift = np.random.multivariate_normal(mean_u.detach().cpu(), 
                #                                     (std_u @ std_u.T).detach().cpu(), 
                #                                     size=z[:,0,:,:].shape).transpose(0,3,1,2)
                # z_shifted[domX] = z[domX] + torch.tensor(shift, dtype=torch.float, device=z.device)
                z_noise_domX = z_noise[domX]
                z_shifted[domX] = z[domX] + (mean_u.reshape(1,-1,1,1) + torch.matmul(std_u, z_noise_domX.T).T)
            
            #############################
            # reverse translation (see. Bishop p.93 (2.116) with A=I, \lambda^{-1}=I and \mu=0)
            #############################
            domY = source_labels == 1
            if domY.any():
                cov_u = torch.matmul(std_u, std_u.T) # L^-1

                y_cov = torch.eye(mean_u.shape[0], device=mean_u.device) + cov_u # (I + L^-1)
                y_prec = torch.inverse(y_cov) # (I + L^-1)^-1

                # use  MCB 161: (I + L)^-1 = L^-1(L^-1+I)^-1 = cov_noise @ y_prec
                x_cond_cov = torch.matmul(cov_u, y_prec) # (I + L)^-1

                # (I + L^-1)^-1 (y - b)
                x_cond_mean = torch.matmul(y_prec, (z[domY] - mean_u.reshape(1,-1,1,1)).T).T 

                z_noise_noisy = z_noise[domY]
                # use svd to obtain x_cond_cov = std_matrix @ std_matrix.T
                # as cholesky decomposition fails with degenerate (i.e singular) covariance matrices
                u, s, _ = torch.svd(x_cond_cov)
                std_matrix = u * torch.sqrt(s)

                z_noise_noisy = torch.matmul(std_matrix, z_noise_noisy.T).T
                z_shifted[domY] = z_noise_noisy + x_cond_mean
            
            return (1.0-heat)*z + heat*z_shifted
        
        self.netG.eval()

        # flip lables from 0 to 1 and from 1 to 0
        target_labels = torch.abs(source_labels - 1)

        # get shift statistics of each level
        shift_dists = self.get_shift_dists()

        # translate epses:
        translated_zs = []
        for z, (mean_u, std_u) in zip(zs,shift_dists):
            translated_zs.append(translate_z(z, mean_u, std_u, source_labels, heat))

        with torch.no_grad():
            translated, _ = self.netG(lr=lq, z=None, eps_std=1.0, reverse=True, epses=translated_zs, y_label=target_labels, add_gt_noise=True, lr_enc=lr_enc)
        self.netG.train()
        return translated

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True, y_label=None):
        self.netG.eval()
        self.fake_H = {}

        if y_label is None:
            y_label = self.y_label

        assert y_label is not None

        for heat in self.heats:
            for i in range(self.n_sample):
                # z is sampled in model
                with torch.no_grad():
                    self.fake_H[(heat, y_label[0], i)], logdet = self.netG(lr=self.var_L, z=None, eps_std=heat, reverse=True, y_label=y_label)
        self.netG.train()

        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR', heat, y_label[0], i)] = self.fake_H[(heat, y_label[0], i)].detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def get_shift_visuals(self, means0, stds0, means1, stds1):
        self.netG.eval()

        def shift(epses):
            mod_epses = []

            for eps, mean0, std0, mean1, std1 in zip(epses, means0, stds0, means1, stds1):
                if self.y_label[0] == 0:
                    mod_eps = (eps-torch.tensor(mean0, device=eps.device))*torch.tensor(std1/std0, device=eps.device)+torch.tensor(mean1, device=eps.device)
                else:
                    mod_eps = (eps-torch.tensor(mean1, device=eps.device))*torch.tensor(std0/std1, device=eps.device)+torch.tensor(mean0, device=eps.device)

                mod_epses.append(mod_eps)
            return mod_epses

        assert self.var_L is not None
        with torch.no_grad():
            epses, nll, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False, y_label=self.y_label, epses=[])
            mod_epses = shift(epses)

            sr, _ = self.netG(lr=self.var_L, reverse=True, epses=mod_epses, y_label=self.y_label)
            sr = sr.detach()[0].float().cpu()
        self.netG.train()

        return sr, self.y_label

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=self.opt['path'].get('strict_load', True), submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
