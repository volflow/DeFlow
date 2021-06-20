import numpy as np
import torch
from torch import nn as nn

import models.modules.Split
from models.modules import flow, thops
from models.modules.BypassSplit import BypassSplit
from models.modules.FlowSqueezeShift import SqueezeShift, shift_list_to_sequence, get_random_shifts
from models.modules.flow import AffineImageInjector, SeperableFixedFilter
from models.modules.Split import Split2d
from models.modules.glow_arch import f_conv2d_bias
from models.modules.FlowStep import FlowStep
from utils.util import opt_get


def get_linenumber():
    from inspect import currentframe
    cf = currentframe()
    return cf.f_back.f_lineno


class FlowUpsamplerNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L=None,
                 actnorm_scale=1.0,
                 flow_permutation=None,
                 flow_coupling="affine",
                 LU_decomposed=False, opt=None):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.L = opt_get(opt, ['network_G', 'flow', 'L'])
        self.K = opt_get(opt, ['network_G', 'flow', 'K'])
        if isinstance(self.K, int):
            self.K = [K for K in [K, ] * (self.L + 1)]

        self.opt = opt
        H, W, self.C = image_shape
        self.check_image_shape()

        if opt['scale'] == 16:
            self.levelToName = {
                0: 'fea_up16',
                1: 'fea_up8',
                2: 'fea_up4',
                3: 'fea_up2',
                4: 'fea_up1',
            }

        if opt['scale'] == 8:
            self.levelToName = {
                0: 'fea_up8',
                1: 'fea_up4',
                2: 'fea_up2',
                3: 'fea_up1',
                4: 'fea_up0'
            }

        elif opt['scale'] == 4:
            self.levelToName = {
                0: 'fea_up4',
                1: 'fea_up2',
                2: 'fea_up1',
                3: 'fea_up0',
                4: 'fea_up-1'
            }

        affineInCh = self.get_affineInCh(opt_get)
        self.get_image_injector_settings(opt)
        flow_permutation = self.get_flow_permutation(flow_permutation, opt)

        normOpt = opt_get(opt, ['network_G', 'flow', 'norm'])

        self.arch_rgbAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt, opt_get)
        self.arch_image_injector_hr(H, W, opt, opt_get)

        conditional_channels = {}
        n_rrdb = self.get_n_rrdb_channels(opt, opt_get)
        n_bypass_channels = opt_get(opt, ['network_G', 'flow', 'levelConditional', 'n_channels'])
        conditional_channels[0] = n_rrdb
        for level in range(1, self.L + 1):
            # Level 1 gets conditionals from 2, 3, 4 => L - level
            # Level 2 gets conditionals from 3, 4
            # Level 3 gets conditionals from 4
            # Level 4 gets conditionals from None
            n_bypass = 0 if n_bypass_channels is None else (self.L - level) * n_bypass_channels
            conditional_channels[level] = n_rrdb + n_bypass

        H, W = self.arch_upsampleAndSqueeze(H, W, opt)

        # Upsampler
        for level in range(1, self.L + 1):
            self.arch_separableFixedFilter(H, W, opt)

            # 1. Squeeze
            H, W = self.arch_squeeze(H, W)

            # 2. K FlowStep
            self.arch_additionalFlowAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt)
            self.arch_FlowStep(H, self.K[level], LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling,
                               flow_permutation,
                               hidden_channels, normOpt, opt, opt_get,
                               n_conditinal_channels=conditional_channels[level])
            self.arch_level_conditional(H, W, opt, opt_get)

            # Split
            self.arch_split(H, W, level, self.L, opt, opt_get)
            # self.arch_preFlow(self.K, LU_decomposed, actnorm_scale, hidden_channels, opt, opt_get)

        if opt_get(opt, ['network_G', 'flow', 'split', 'enable']):
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64 // 2 // 2)
        else:
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64)

        self.H = H
        self.W = W
        self.scaleH = 160 / H
        self.scaleW = 160 / W

    def get_n_rrdb_channels(self, opt, opt_get):
        blocks = opt_get(opt, ['network_G', 'flow', 'stackRRDB', 'blocks'])
        n_rrdb = 64 if blocks is None else (len(blocks) + 1) * 64
        return n_rrdb

    def arch_level_conditional(self, H, W, opt, opt_get):
        levelConditionalOpt = opt_get(opt, ['network_G', 'flow', 'levelConditional'])
        if levelConditionalOpt is not None and levelConditionalOpt['type'] == 'rgb':
            self.layers.append(BypassSplit(n_split=3))
            self.C = self.C + 3
            self.output_shapes.append([-1, self.C, H, W])

    def arch_FlowStep(self, H, K, LU_decomposed, W, actnorm_scale, affineInCh, flow_coupling, flow_permutation,
                      hidden_channels, normOpt, opt, opt_get, n_conditinal_channels=None):
        condAff = self.get_condAffSetting(opt, opt_get)
        if condAff is not None:
            condAff['in_channels_rrdb'] = n_conditinal_channels

        if opt_get(opt, ["network_G", "flow", "SqueezeShiftPerFlowStep"]):
            shifts_sequence = shift_list_to_sequence(get_random_shifts(K=K, channels=self.C // 4))

        for k in range(K):
            if opt_get(opt, ["network_G", "flow", "SqueezeShiftPerFlowStep"]):
                self.layers.append(SqueezeShift(shifts_sequence[k]))
                self.output_shapes.append([-1, self.C, H, W])

            position_name = get_position_name(H, self.opt['scale'])
            if normOpt: normOpt['position'] = position_name

            self.layers.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         acOpt=condAff,
                         position=position_name,
                         LU_decomposed=LU_decomposed, opt=opt, idx=k, normOpt=normOpt))
            self.output_shapes.append(
                [-1, self.C, H, W])

            self.arch_FlowStep_image_injector(H, W, affineInCh)
            self.arch_FlowStep_image_injector_after_flow_step(H, W, affineInCh, opt, opt_get)

        if opt_get(opt, ["network_G", "flow", "SqueezeShiftPerFlowStep"]):
            self.layers.append(SqueezeShift(shifts_sequence[k]))
            self.output_shapes.append([-1, self.C, H, W])

    def get_condAffSetting(self, opt, opt_get):
        condAff = opt_get(opt, ['network_G', 'flow', 'condAff']) or None
        condAff = opt_get(opt, ['network_G', 'flow', 'condFtAffine']) or condAff
        return condAff

    def arch_FlowStep_image_injector(self, H, W, affineInCh):
        if self.image_injector_style:
            self.layers.append(
                flow.AffineImageInjector(affineInCh, self.C, eps=self.image_injector_eps,
                                         f=self.image_injector_style,
                                         hidden_layers=self.image_injector_layers,
                                         scale=H // 20))
            self.output_shapes.append([-1, self.C, H, W])

    def arch_FlowStep_image_injector_after_flow_step(self, H, W, affineInCh, opt, opt_get):
        image_injector_after_flow_step = opt_get(opt, ['network_G', 'flow', 'image_injector_after_flow_step'])
        if image_injector_after_flow_step:
            assert image_injector_after_flow_step['scale'] == 1, image_injector_after_flow_step['scale']
            assert image_injector_after_flow_step['position'] == 'auto', image_injector_after_flow_step['position']
            position_name = get_position_name(H, self.opt['scale'])
            for _ in range(image_injector_after_flow_step['n_steps']):
                self.layers.append(
                    AffineImageInjector(in_channels=affineInCh,
                                        out_channels=self.C,
                                        hidden_layers=image_injector_after_flow_step['hidden_layers'],
                                        eps=image_injector_after_flow_step['eps'],
                                        position=position_name,
                                        scale=1,
                                        f=image_injector_after_flow_step['style']
                                        )
                )
                self.output_shapes.append(
                    [-1, self.C, H, W])

    def arch_preFlow(self, K, LU_decomposed, actnorm_scale, hidden_channels, opt, opt_get):
        self.preFlow = nn.ModuleList()
        preFlow = opt_get(opt, ['network_G', 'flow', 'preFlow'])
        flow_permutation = self.get_flow_permutation(None, opt)
        if preFlow:
            for k in range(K):
                self.preFlow.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt))

    def arch_split(self, H, W, L, levels, opt, opt_get):
        correct_splits = opt_get(opt, ['network_G', 'flow', 'split', 'correct_splits'], False)
        correction = 0 if correct_splits else 1
        if opt_get(opt, ['network_G', 'flow', 'split', 'enable']) and L < levels - correction:
            logs_eps = opt_get(opt, ['network_G', 'flow', 'split', 'logs_eps']) or 0
            consume_ratio = opt_get(opt, ['network_G', 'flow', 'split', 'consume_ratio']) or 0.5
            position_name = get_position_name(H, self.opt['scale'])
            position = position_name if opt_get(opt, ['network_G', 'flow', 'split', 'conditional']) else None
            cond_channels = opt_get(opt, ['network_G', 'flow', 'split', 'cond_channels'])
            cond_channels = 0 if cond_channels is None else cond_channels

            t = opt_get(opt, ['network_G', 'flow', 'split', 'type'], 'Split2d')

            if t == 'Split2d':
                split = models.modules.Split.Split2d(num_channels=self.C, logs_eps=logs_eps, position=position,
                                                     cond_channels=cond_channels, consume_ratio=consume_ratio, opt=opt)
            elif t == 'FlowSplit2d':
                split = models.modules.Split.FlowSplit2d(num_channels=self.C, logs_eps=logs_eps, position=position,
                                                         cond_channels=cond_channels, consume_ratio=consume_ratio,
                                                         opt=opt)

            self.layers.append(split)
            self.output_shapes.append([-1, split.num_channels_pass, H, W])
            self.C = split.num_channels_pass

    def arch_additionalFlowAffine(self, H, LU_decomposed, W, actnorm_scale, hidden_channels, opt):
        if 'additionalFlowNoAffine' in opt['network_G']['flow']:
            n_additionalFlowNoAffine = int(opt['network_G']['flow']['additionalFlowNoAffine'])
            flow_permutation = self.get_flow_permutation(None, opt)

            for _ in range(n_additionalFlowNoAffine):
                self.layers.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt))
                self.output_shapes.append(
                    [-1, self.C, H, W])

    def arch_upsampleAndSqueeze(self, H, W, opt):
        if not 'UpsampleAndSqueeze' in opt['network_G']['flow']:
            return H, W

        self.C = self.C * 2 * 2
        self.layers.append(flow.UpsampleAndSqueezeLayer())
        self.output_shapes.append([-1, self.C, H, W])
        return H, W

    def arch_squeeze(self, H, W):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.layers.append(flow.SqueezeLayer(factor=2))
        self.output_shapes.append([-1, self.C, H, W])
        return H, W

    def arch_separableFixedFilter(self, H, W, opt):
        if 'SeparableFixedFilter' in opt['network_G']['flow'] \
                and opt['network_G']['flow']['SeparableFixedFilter']['position'] == 'beforeSqueeze':
            self.layers.append(
                SeperableFixedFilter(opt['network_G']['flow']['SeparableFixedFilter']['coefficients']))
            self.output_shapes.append([-1, self.C, H, W])

    def arch_image_injector_hr(self, H, W, opt, opt_get):
        image_injector_hr = opt_get(opt, ['network_G', 'flow', 'image_injector_hr'])
        if image_injector_hr:
            in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'image_injector_hr', 'in_channels_rrdb'])
            in_channels_rrdb = 64 if in_channels_rrdb is None else in_channels_rrdb
            scale = image_injector_hr['scale']
            for _ in range(image_injector_hr['n_steps']):
                self.layers.append(
                    AffineImageInjector(in_channels=in_channels_rrdb,
                                        out_channels=self.C,
                                        hidden_layers=image_injector_hr['hidden_layers'],
                                        eps=image_injector_hr['eps'],
                                        position=image_injector_hr['position'],
                                        scale=image_injector_hr['scale'],
                                        f=image_injector_hr['style']
                                        )
                )
                self.output_shapes.append(
                    [-1, self.C, H, W])

    def arch_rgbAffine(self, H, LU_decomposed, W, actnorm_scale, hidden_channels, opt, opt_get):
        rgbAffine = opt_get(opt, ['network_G', 'flow', 'rgbAffine'])
        if rgbAffine is not None:
            for _ in range(rgbAffine['n_steps']):
                self.layers.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='affineCustom',
                             LU_decomposed=LU_decomposed, opt=opt,
                             acOpt=rgbAffine))
                self.output_shapes.append(
                    [-1, self.C, H, W])

    def get_flow_permutation(self, flow_permutation, opt):
        flow_permutation = opt['network_G']['flow'].get('flow_permutation', 'invconv')
        return flow_permutation

    def get_image_injector_settings(self, opt):
        self.image_injector_style = None
        if 'image_injector' in opt['network_G']['flow'].keys():
            self.image_injector_style = opt['network_G']['flow']['image_injector']['style']
            self.image_injector_eps = opt['network_G']['flow']['image_injector']['eps']
            self.image_injector_layers = opt['network_G']['flow']['image_injector'].get('hidden_layers', 1)

    def get_affineInCh(self, opt_get):
        affineInCh = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        affineInCh = (len(affineInCh) + 1) * 64
        return affineInCh

    def check_image_shape(self):
        assert self.C == 1 or self.C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                            "self.C == 1 or self.C == 3")

    def forward(self, gt=None, rrdbResults=None, z=None, epses=None, logdet=0., reverse=False, eps_std=None, y_onehot=None):

        if reverse:
            epses_copy = [eps for eps in epses] if isinstance(epses, list) else epses

            sr, logdet = self.decode(rrdbResults, z, eps_std, epses=epses_copy, logdet=logdet, y_onehot=y_onehot)
            return sr, logdet
        else:
            assert gt is not None
            assert rrdbResults is not None
            z, logdet = self.encode(gt, rrdbResults, logdet=logdet, epses=epses, y_onehot=y_onehot)

            return z, logdet

    def encode(self, gt, rrdbResults, logdet=0.0, epses=None, y_onehot=None):
        fl_fea = gt
        reverse = False
        level_conditionals = {}
        bypasses = {}

        L = opt_get(self.opt, ['network_G', 'flow', 'L'])

        for level in range(1, L + 1):
            bypasses[level] = torch.nn.functional.interpolate(gt, scale_factor=2 ** -level, mode='bilinear')

        for layer, shape in zip(self.layers, self.output_shapes):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))

            if level > 0 and level not in level_conditionals.keys():
                if opt_get(self.opt, ['network_G', 'flow', 'levelConditional', 'conditional']) == True:

                    conditionals = []
                    conditionals.append(rrdbResults[self.levelToName[level]])
                    for l in range(level, L):
                        # level = 1 -> all previous levels
                        # 4 -> upscale 2^(1 - 4) = 2^(level -l)
                        conditionals.append(
                            torch.nn.functional.interpolate(bypasses[l], scale_factor=size / bypasses[l].shape[2]))
                    level_conditionals[level] = torch.cat(conditionals, dim=1)

                else:
                    level_conditionals[level] = rrdbResults[self.levelToName[level]]

            if opt_get(self.opt, ['flow', 'image_injector_hr']) is not None:
                level_conditionals[0] = rrdbResults[self.levelToName[0]]

            level_conditionals[level] = rrdbResults[self.levelToName[level]]

            if isinstance(layer, flow.AffineImageInjector):
                fl_fea, logdet = self.forward_affine_image_injector_not_reverse(fl_fea, layer, logdet, reverse,
                                                                                level_conditionals[level])
            elif isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse, rrdbResults=level_conditionals[level])
            elif isinstance(layer, Split2d):
                fl_fea, logdet = self.forward_split2d(epses, fl_fea, layer, logdet, reverse, level_conditionals[level], y_onehot=y_onehot)
            elif isinstance(layer, BypassSplit):
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse, bypass=bypasses[level])
            else:
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse)

        fl_fea, logdet = self.forward_inital_image_injector_not_reverse(fl_fea, logdet, level_conditionals[level])

        fl_fea, logdet = self.forward_preFlow(fl_fea, logdet, reverse)

        z = fl_fea

        if not isinstance(epses, list):
            return z, logdet

        epses.append(z)
        return epses, logdet

    def forward_preFlow(self, fl_fea, logdet, reverse):
        if hasattr(self, 'preFlow'):
            for l in self.preFlow:
                fl_fea, logdet = l(fl_fea, logdet, reverse=reverse)
        return fl_fea, logdet

    def forward_inital_image_injector_not_reverse(self, fl_fea, logdet, rrdbResults):
        if not opt_get(self.opt, ['network_G', 'flow', 'noInitialInj']):
            h = self.f(rrdbResults['last_lr_fea'])
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            fl_fea = fl_fea + shift
            fl_fea = fl_fea * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        return fl_fea, logdet

    def forward_split2d(self, epses, fl_fea, layer, logdet, reverse, rrdbResults, y_onehot=None):
        ft = None if layer.position is None else rrdbResults[layer.position]
        fl_fea, logdet, eps = layer(fl_fea, logdet, reverse=reverse, eps=epses, ft=ft, y_onehot=y_onehot)

        if isinstance(epses, list):
            epses.append(eps)
        return fl_fea, logdet

    def forward_affine_image_injector_not_reverse(self, fl_fea, layer, logdet, reverse, rrdbResults):
        if layer.position is None:
            fl_fea, logdet = layer(fl_fea, rrdbResults['last_lr_fea'], logdet, reverse=reverse)
        elif isinstance(layer.position, float):
            rrdbScale = rrdbResults['last_lr_fea']
            rrdbScale = torch.nn.functional.interpolate(rrdbScale, scale_factor=layer.position, mode='bilinear')
            fl_fea, logdet = layer(fl_fea, rrdbScale, logdet, reverse=reverse)
        elif isinstance(rrdbResults, torch.Tensor):
            fl_fea, logdet = layer(fl_fea, rrdbResults, logdet, reverse=reverse)
        else:
            fl_fea, logdet = layer(fl_fea, rrdbResults.get(layer.position, None), logdet, reverse=reverse)
        return fl_fea, logdet

    def decode(self, rrdbResults, z, eps_std=None, epses=None, logdet=0.0, y_onehot=None):
        reverse = True
 
        lr_fea = rrdbResults['last_lr_fea']

        z = epses.pop() if isinstance(epses, list) else z

        z, logdet = self.forward_preFlow(z, logdet=logdet, reverse=reverse)

        z, logdet = self.forward_inital_image_injector_reverse(lr_fea, logdet, z)

        fl_fea = z
        bypasses = {}
        level_conditionals = {}
        if not opt_get(self.opt, ['network_G', 'flow', 'levelConditional', 'conditional']) == True:
            for level in range(self.L + 1):
                level_conditionals[level] = rrdbResults[self.levelToName[level]]

        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))

            if isinstance(layer, (Split2d)):
                fl_fea, logdet = self.forward_split2d_reverse(eps_std, epses, fl_fea, layer,
                                                              rrdbResults[self.levelToName[level]], logdet=logdet, y_onehot=y_onehot)
            elif isinstance(layer, flow.AffineImageInjector):
                fl_fea, logdet = self.forward_affine_image_injector_reverse(fl_fea, layer, level_conditionals[level])
            elif isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True, rrdbResults=level_conditionals[level])
            elif isinstance(layer, BypassSplit):
                z_bypass_dict, logdet = layer(fl_fea, logdet, reverse=reverse)
                fl_fea = z_bypass_dict['z']
                bypass = z_bypass_dict['bypass']
                bypasses[level] = bypass

                if level > 0 and level not in level_conditionals.keys():
                    if opt_get(self.opt, ['network_G', 'flow', 'levelConditional', 'conditional']) == True:

                        conditionals = []
                        conditionals.append(rrdbResults[self.levelToName[level]])
                        for l in range(level, self.L):
                            # level = 1 -> all previous levels
                            # 4 -> upscale 2^(1 - 4) = 2^(level -l)
                            conditionals.append(
                                torch.nn.functional.interpolate(bypasses[l], scale_factor=size / bypasses[l].shape[2]))
                        level_conditionals[level] = torch.cat(conditionals, dim=1)

            else:
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True)

        sr = fl_fea

        # debug.imwrite("sr", sr)

        assert sr.shape[1] == 3
        return sr, logdet

    def forward_affine_image_injector_reverse(self, fl_fea, layer, rrdbResults):
        if layer.position is None:
            fl_fea, logdet = layer(fl_fea, rrdbResults['last_lr_fea'], logdet=0, reverse=True)
        elif isinstance(layer.position, float):
            rrdbScale = rrdbResults['last_lr_fea']
            rrdbScale = torch.nn.functional.interpolate(rrdbScale, scale_factor=layer.position, mode='bilinear')
            fl_fea, logdet = layer(fl_fea, rrdbScale, logdet=0, reverse=True)
        elif isinstance(rrdbResults, torch.Tensor):
            fl_fea, logdet = layer(fl_fea, rrdbResults, logdet=0, reverse=True)
        else:
            fl_fea, logdet = layer(fl_fea, rrdbResults[layer.position], logdet=0, reverse=True)
        return fl_fea, logdet

    def forward_split2d_reverse(self, eps_std, epses, fl_fea, layer, rrdbResults, logdet, y_onehot=None):
        ft = None if layer.position is None else rrdbResults[layer.position]

        fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True,
                               eps=epses.pop() if isinstance(epses, list) else None,
                               eps_std=eps_std, ft=ft, y_onehot=y_onehot)
        return fl_fea, logdet

    def forward_inital_image_injector_reverse(self, lr_fea, logdet, z):
        if not opt_get(self.opt, ['network_G', 'flow', 'noInitialInj']):
            h = self.f(lr_fea)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z = z / scale
            z = z - shift
            logdet = logdet - thops.sum(torch.log(scale), dim=[1, 2, 3])
        return z, logdet


def get_position_name(H, scale):
    # scale 8, H = 80 -> fea_up4 (160 / 80 = 2, 8/2 = 4
    # scale 4, H = 80 -> fea_up2 (160 / 80 = 2, 4/2 = 2
    # scale 4, H = 40 -> fea_up1 (160 / 40 = 4, 4/4 = 1
    # scale 4, H = 20 -> fea_up0 (160 / 20 = 8, 4/8 = 0.5
    downscale_factor = 160 // H
    position_name = 'fea_up{}'.format(scale / downscale_factor)
    return position_name
