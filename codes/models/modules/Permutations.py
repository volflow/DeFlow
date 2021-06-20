import math

import numpy as np
import scipy.linalg
import torch
from torch import nn as nn
from torch.nn import functional as F

from models.modules import thops
from models.modules.flow import unsqueeze2d, squeeze2d, SqueezeLayer


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.long)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4
        if not reverse:
            return input[:, self.indices, :, :]
        else:
            return input[:, self.indices_inverse, :, :]

def clip_singular_values(M, min=1e-6):
    U, s, V = torch.svd(M)
    s = torch.clamp(s, min=min)
    return torch.mm(torch.mm(U, torch.diag(s)), V.t())

def clip_singular_values_forward_hook(self, grad_input, grad_output):
    self.weight.copy_(clip_singular_values(self.weight.double(), min=self.min_singular).float())

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False, min_singular=None):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed


        self.min_singular = min_singular
        #self.register_backward_hook(clip_singular_values_forward_hook)


    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float() \
                    .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """

        if self.min_singular is not None:
            # clip singular values
            with torch.no_grad():
                self.weight.copy_(clip_singular_values(self.weight.double(), min=self.min_singular).float())

        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class InvertibleConv1x1Resqueeze(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False, resqueeze=None):
        super().__init__()
        self.invertibleConv1x1 = InvertibleConv1x1(num_channels // (2 ** 2) * (resqueeze ** 2), LU_decomposed)
        self.resqueeze = resqueeze

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            z = input

            z = unsqueeze2d(z, factor=2)
            z = squeeze2d(z, factor=self.resqueeze)

            z, logdet = self.invertibleConv1x1(z, logdet, reverse=False)

            z = unsqueeze2d(z, factor=self.resqueeze)
            z = squeeze2d(z, factor=2)

            return z, logdet
        else:
            z = input

            z = unsqueeze2d(z, factor=2)
            z = squeeze2d(z, factor=self.resqueeze)

            z, logdet = self.invertibleConv1x1(z, logdet, reverse=True)

            z = unsqueeze2d(z, factor=self.resqueeze)
            z = squeeze2d(z, factor=2)
            return z, logdet


class SqueezeInvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False, squeezeFactor=1):
        super().__init__()
        self.squeeze = SqueezeLayer(squeezeFactor)
        self.invertibleConv1x1 = InvertibleConv1x1(num_channels * (squeezeFactor ** 2), LU_decomposed)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            z = input
            z, logdet = self.squeeze(z, logdet, reverse=False)
            z, logdet = self.invertibleConv1x1(z, logdet, reverse=False)
            z, logdet = self.squeeze(z, logdet, reverse=True)
            return z, logdet
        else:
            z = input
            z, logdet = self.squeeze(z, logdet, reverse=False)
            z, logdet = self.invertibleConv1x1(z, logdet, reverse=True)
            z, logdet = self.squeeze(z, logdet, reverse=True)
            return z, logdet


class InvConditionalLinear(nn.Module):
    def get_weight(self, weight):
        w_shape = round(math.sqrt(weight.shape[1]))
        weight = weight.permute(0, 2, 3, 1)
        weight = weight.view(*weight.shape[:-1], w_shape, w_shape)
        weight_logdet = torch.sum(torch.slogdet(weight)[1], dim=(1, 2))
        return weight, weight_logdet

    def forward(self, input, weight, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        assert input.shape[1] ** 2 == weight.shape[1], (input.shape, weight.shape)
        assert input.shape[2:] == weight.shape[2:] or input.shape[2:] == (1, 1)

        weight, weight_logdet = self.get_weight(weight)
        if not reverse:
            z = torch.solve(input.permute(0, 2, 3, 1).unsqueeze(-1), weight)[0].permute(0, 3, 1, 2, 4).squeeze(
                -1).contiguous()
            if logdet is not None:
                logdet = logdet - weight_logdet
            return z, logdet
        else:
            z = torch.matmul(weight, input.permute(0, 2, 3, 1).unsqueeze(-1)).permute(0, 3, 1, 2, 4).squeeze(
                -1).contiguous()
            if logdet is not None:
                logdet = logdet + weight_logdet
            return z, logdet


class InvConditionalUnitriLinear(nn.Module):
    def get_weight(self, weight, d):
        weight = weight.permute(0, 2, 3, 1)

        vec = weight.new_zeros(*weight.shape[:-1], d)
        vec[..., 0] = 1

        # Create triangular weight
        weight = torch.stack(
            [torch.cat((weight[..., (j * (j - 1)) // 2:j * (j + 1) // 2], vec[..., :d - j]), dim=-1) for j in range(d)],
            dim=-1)

        return weight

    def forward(self, input, weight, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        input_dim = input.shape[1]
        assert (input_dim ** 2 - input_dim) / 2 == weight.shape[1]
        assert input.shape[2:] == weight.shape[2:] or input.shape[2:] == (1, 1)

        weight = self.get_weight(weight, input_dim)
        if not reverse:
            z = torch.triangular_solve(input.permute(0, 2, 3, 1).unsqueeze(-1).double(), weight.double(),
                                       unitriangular=True)[0].permute(
                0, 3, 1, 2, 4).squeeze(-1).float().contiguous()
            # z = torch.matmul(weight.double().inverse(), input.permute(0, 2, 3, 1).unsqueeze(-1).double()).permute(0, 3, 1, 2, 4).squeeze(
            #     -1).float().contiguous()
        else:
            z = torch.matmul(weight.double(), input.permute(0, 2, 3, 1).unsqueeze(-1).double()).permute(0, 3, 1, 2,
                                                                                                        4).squeeze(
                -1).float().contiguous()
        return z, logdet


class InvertibleConv1x1GridAlign(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.inds = self.get_indices(num_channels)

    def get_indices(self, num_channels):
        def generate_ind(ind):
            return squeeze2d(ind.view(1, 1, 2, 2).repeat(1, num_channels // 4, 1, 1), factor=2).view(-1)

        m = torch.ones(num_channels, dtype=torch.bool)
        t = generate_ind(torch.BoolTensor([[0, 0], [1, 1]]))
        b = ~t
        l = generate_ind(torch.BoolTensor([[0, 1], [0, 1]]))
        r = ~l
        tl = t & l
        tr = t & r
        bl = b & l
        br = b & r

        return m, t, b, l, r, tl, tr, bl, br

    def get_weight(self, input, reverse):
        h, w = input.shape[-2:]
        pixels = [(h - 1) * (w - 1), w - 1, w - 1, h - 1, h - 1, 1, 1, 1, 1]

        weights = [self.weight[ind, :][:, ind] for ind in self.inds]

        dlogdet = sum([torch.slogdet(wgt)[1] * num_pix for wgt, num_pix in zip(weights, pixels)])

        if not reverse:
            weights = [wgt.view(wgt.shape[0], -1, 1, 1) for wgt in weights]
        else:
            weights = [torch.inverse(wgt.double()).float().view(wgt.shape[0], -1, 1, 1) for wgt in weights]
        return weights, dlogdet

    def partition_input(self, input):
        z = squeeze2d(F.pad(unsqueeze2d(input, factor=2), (1, 1, 1, 1), value=-1), factor=2)

        m, t, b, l, r, tl, tr, bl, br = self.inds

        z_m = z[..., 1:-1, 1:-1]
        z_t = z[:, t, :1, 1:-1]
        z_b = z[:, b, -1:, 1:-1]
        z_l = z[:, l, 1:-1, :1]
        z_r = z[:, r, 1:-1, -1:]
        z_tl = z[:, tl, :1, :1]
        z_tr = z[:, tr, :1, -1:]
        z_bl = z[:, bl, -1:, :1]
        z_br = z[:, br, -1:, -1:]

        return z_m, z_t, z_b, z_l, z_r, z_tl, z_tr, z_bl, z_br

    def merge_output(self, z_m, z_t, z_b, z_l, z_r, z_tl, z_tr, z_bl, z_br):
        def unsqueeze_line(ze):
            B, C = ze.shape[:2]
            sz = [1, -1] if ze.shape[-2] == 1 else [-1, 1]
            return ze.view(B, C // 2, 2, -1).permute(0, 1, 3, 2).reshape(B, C // 2, *sz)

        # Unsqueeze things
        z_m = unsqueeze2d(z_m, factor=2)
        z_t, z_b, z_l, z_r = [unsqueeze_line(zi) for zi in [z_t, z_b, z_l, z_r]]

        z_tt = torch.cat([z_tl, z_t, z_tr], dim=3)
        z_mm = torch.cat([z_l, z_m, z_r], dim=3)
        z_bb = torch.cat([z_bl, z_b, z_br], dim=3)

        return squeeze2d(torch.cat([z_tt, z_mm, z_bb], dim=2), factor=2)

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """

        weights, dlogdet = self.get_weight(input, reverse)
        inputs = self.partition_input(input)
        outputs = [F.conv2d(z, wgt) for z, wgt in zip(inputs, weights)]
        output = self.merge_output(*outputs)

        if not reverse:
            if logdet is not None:
                logdet = logdet + dlogdet
        else:
            if logdet is not None:
                logdet = logdet - dlogdet

        return output, logdet


class InvertibleConv1x1GridAlignIndepBorder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.inds = self.get_indices(num_channels)
        self.param_names = ['weight_' + wn for wn in ['m', 't', 'b', 'l', 'r', 'tl', 'tr', 'bl', 'br']]
        for w_name, ind in zip(self.param_names, self.inds):
            d = ind.long().sum().item()
            w_shape = [d, d]
            w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
            self.register_parameter(w_name, nn.Parameter(torch.Tensor(w_init)))

    def get_indices(self, num_channels):
        def generate_ind(ind):
            return squeeze2d(ind.view(1, 1, 2, 2).repeat(1, num_channels // 4, 1, 1), factor=2).view(-1)

        m = torch.ones(num_channels, dtype=torch.bool)
        t = generate_ind(torch.BoolTensor([[0, 0], [1, 1]]))
        b = ~t
        l = generate_ind(torch.BoolTensor([[0, 1], [0, 1]]))
        r = ~l
        tl = t & l
        tr = t & r
        bl = b & l
        br = b & r

        return m, t, b, l, r, tl, tr, bl, br

    def get_weight(self, input, reverse):
        h, w = input.shape[-2:]
        pixels = [(h - 1) * (w - 1), w - 1, w - 1, h - 1, h - 1, 1, 1, 1, 1]

        weights = [getattr(self, w_name) for w_name in self.param_names]

        dlogdet = sum([torch.slogdet(wgt)[1] * num_pix for wgt, num_pix in zip(weights, pixels)])

        if not reverse:
            weights = [wgt.view(wgt.shape[0], -1, 1, 1) for wgt in weights]
        else:
            weights = [torch.inverse(wgt.double()).float().view(wgt.shape[0], -1, 1, 1) for wgt in weights]
        return weights, dlogdet

    def partition_input(self, input):
        z = squeeze2d(F.pad(unsqueeze2d(input, factor=2), (1, 1, 1, 1), value=-1), factor=2)

        m, t, b, l, r, tl, tr, bl, br = self.inds

        z_m = z[..., 1:-1, 1:-1]
        z_t = z[:, t, :1, 1:-1]
        z_b = z[:, b, -1:, 1:-1]
        z_l = z[:, l, 1:-1, :1]
        z_r = z[:, r, 1:-1, -1:]
        z_tl = z[:, tl, :1, :1]
        z_tr = z[:, tr, :1, -1:]
        z_bl = z[:, bl, -1:, :1]
        z_br = z[:, br, -1:, -1:]

        return z_m, z_t, z_b, z_l, z_r, z_tl, z_tr, z_bl, z_br

    def merge_output(self, z_m, z_t, z_b, z_l, z_r, z_tl, z_tr, z_bl, z_br):
        def unsqueeze_line(ze):
            B, C = ze.shape[:2]
            sz = [1, -1] if ze.shape[-2] == 1 else [-1, 1]
            return ze.view(B, C // 2, 2, -1).permute(0, 1, 3, 2).reshape(B, C // 2, *sz)

        # Unsqueeze things
        z_m = unsqueeze2d(z_m, factor=2)
        z_t, z_b, z_l, z_r = [unsqueeze_line(zi) for zi in [z_t, z_b, z_l, z_r]]

        z_tt = torch.cat([z_tl, z_t, z_tr], dim=3)
        z_mm = torch.cat([z_l, z_m, z_r], dim=3)
        z_bb = torch.cat([z_bl, z_b, z_br], dim=3)

        return squeeze2d(torch.cat([z_tt, z_mm, z_bb], dim=2), factor=2)

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """

        weights, dlogdet = self.get_weight(input, reverse)
        inputs = self.partition_input(input)
        outputs = [F.conv2d(z, wgt) for z, wgt in zip(inputs, weights)]
        output = self.merge_output(*outputs)

        if not reverse:
            if logdet is not None:
                logdet = logdet + dlogdet
        else:
            if logdet is not None:
                logdet = logdet - dlogdet

        return output, logdet


def rnd(f):
    return int(round(f))


class InvertibleConv1x1SubblocksShuf(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False, n_blocks=None):
        super().__init__()
        assert num_channels > n_blocks, (num_channels, n_blocks)

        self.perm = Permute2d(num_channels, shuffle=True)

        self.n_blocks = n_blocks
        self.permutations = torch.nn.ModuleList()
        self.num_channels = num_channels

        borders = [rnd(num_channels / n_blocks * i) for i in range(n_blocks + 1)]
        begs = borders[:-1]
        ends = borders[1:]
        channels = [end - beg for beg, end in zip(begs, ends)]

        self.begs = begs
        self.ends = ends

        for i in range(self.n_blocks):
            self.permutations.append(InvertibleConv1x1(channels[i], LU_decomposed))

    def forward(self, input, logdet=None, reverse=False):
        z = input

        if not reverse:  # Encode
            self.perm(z, reverse)

        # z = self.perm(z, reverse=reverse)

        c = self.num_channels // self.n_blocks

        # Split into groups
        zs_out = []
        dlogdet = 0

        for i, permutation in enumerate(self.permutations):
            z_out, logdet_out = permutation(z[:, self.begs[i]:self.ends[i]], logdet=0, reverse=reverse)
            zs_out.append(z_out)
            dlogdet = dlogdet + logdet_out

        if reverse:  # Decode
            self.perm(z, reverse)

        if reverse:
            logdet = logdet - dlogdet
        else:
            logdet = logdet + dlogdet
        return torch.cat(zs_out, dim=1), logdet


class RandomRotation(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        n_sample = 10 #arch.get('n_samples', 10)

        w_shape = [num_channels, num_channels]
        Q, QInv = self.sample_Q(num_channels, n_sample)
        self.register_buffer("Q", Q.view(w_shape[0], w_shape[1], 1, 1))
        self.register_buffer("QInv", QInv.view(w_shape[0], w_shape[1], 1, 1))
        self.w_shape = w_shape

    def sample_Q(self, num_channels, n_sample):
        Qs = []
        errs = []
        for i in range(n_sample):
            A = torch.rand(num_channels, num_channels, dtype=torch.float32)
            Q, R = torch.qr(A)
            Qs.append(Q)
            err = torch.max(torch.abs(torch.inverse(torch.inverse(Q)) - Q)).item()
            errs.append(err)
        idx_best = np.argmin(errs)
        Q = Qs[idx_best]
        return Q, torch.inverse(Q)

    def forward(self, x, logdet, reverse=False):
        if reverse:
            return F.conv2d(x, self.Q), logdet
        else:
            return F.conv2d(x, self.QInv), logdet



