import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.modules.FlowActNorms import ActNorm2d
from . import thops


class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__(in_channels, out_channels)
        self.logscale_factor = logscale_factor
        # set logs parameter
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = exp(logs) ** 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def likelihood_var(mean, var, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            return -0.5 * (torch.log(var) + ((x - mean) ** 2) / var + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return thops.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def logp_var(mean, var, x):
        likelihood = GaussianDiag.likelihood_var(mean, var, x)
        return thops.sum(likelihood, dim=[1, 2, 3])


    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.normal(mean=torch.zeros(shape),
                           std=torch.ones(shape) * eps_std)
        return eps

class Gaussian:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, cov, x):
        """
        lnL = -1/2 * { ln|Cov| + k*ln(2*PI) + ((X - Mu)^T)(Var^-1)(X - Mu)  }
        """
        x = x.permute(0,2,3,1)
        k = x.shape[-1]
        
        if mean is None and cov is None:
            # assume mean 0 and cov I
            return -0.5*(k*Gaussian.Log2PI + torch.sum(x**2, dim=-1))
        else:
            x = x - mean
            cov_inv = torch.inverse(cov)
            logdet = torch.slogdet(cov)[1]
            return  -0.5*(logdet + k*Gaussian.Log2PI + torch.sum((x @ cov_inv) * x, dim=-1))
        
    @staticmethod
    def logp(mean, cov, x):
        likelihood = Gaussian.likelihood(mean, cov, x)
        return thops.sum(likelihood, dim=[1, 2])

def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = squeeze2d(input, self.factor)  # Squeeze in forward
            return output, logdet
        else:
            output = unsqueeze2d(input, self.factor)
            return output, logdet


class UnsqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = unsqueeze2d(input, self.factor)  # Unsqueeze in forward
            return output, logdet
        else:
            output = squeeze2d(input, self.factor)
            return output, logdet


class Exp(nn.Module):
    r"""Exponential activation
    """

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = torch.exp(input)
        else:
            output = torch.log(input)

        if logdet is not None:
            dlogdet = torch.sum(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return output, logdet


class BentIdentPar(nn.Module):
    r"""BentIdent parametric activation
    """

    def __init__(self, a=0.2, b=1.0):
        super().__init__()
        self.b = b
        self.a = a

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            # Encode
            output = (1.0 - self.a) / 2.0 * (torch.sqrt(input * input + 4.0 * self.b * self.b) - 2.0 * self.b) + (
                    1.0 + self.a) / 2.0 * input
        else:
            # Decode
            a = self.a
            b = self.b
            output = (b - a ** 2 * b + input + a * input + (a - 1) * torch.sqrt(
                b ** 2 + 2 * a * b ** 2 + a ** 2 * b ** 2 + 2 * b * input - 2 * a * b * input + input ** 2)) / (2 * a)

        if logdet is not None:
            dlogdet = torch.sum(torch.log(self.derivative(input)), dim=(1, 2, 3))
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return output, logdet

    def derivative(self, input, reverse=False):
        a = self.a
        b = self.b
        if not reverse:
            # Encode
            return (1.0 - self.a) / 2.0 * (input / torch.sqrt(input * input + 4.0 * self.b * self.b)) + (
                    1.0 + self.a) / 2.0
        else:
            # Decode
            return (1 + a + ((a - 1) * (2 * b - 2 * a * b + 2 * input)) / (2 * torch.sqrt(
                b ** 2 + 2 * a * b ** 2 + a ** 2 * b ** 2 + 2 * b * input - 2 * a * b * input + input ** 2))) / (2 * a)


class SeperableFixedFilter(nn.Module):
    r"""Separable blur filter
    """

    def __init__(self, coefficients: list):
        super().__init__()
        assert isinstance(coefficients, list), coefficients
        assert len(coefficients) % 2 == 1, coefficients

        self.weights_reverse = torch.Tensor(coefficients)
        self.weights_forward_h = None
        self.weights_forward_w = None

    def forward(self, input: torch.Tensor, logdet=None, reverse=False):
        if not reverse:
            # Encode
            self.generate_forward_matrices(input)
            x = input
            x = torch.matmul(self.weights_forward_h, x)  # Height
            x = torch.matmul(self.weights_forward_w, x.unsqueeze(-1))  # Height
            output = x.squeeze(-1)
        else:
            # Decode
            x = input.view(-1, 1, *input.shape[2:])
            x = F.conv2d(x, self.weights_reverse.view(1, 1, 1, -1).to(input.device),
                         padding=[0, self.weights_reverse.numel() // 2])  # Width
            x = F.conv2d(x, self.weights_reverse.view(1, 1, -1, 1).to(input.device),
                         padding=[self.weights_reverse.numel() // 2, 0])  # Height
            output = x.view(input.shape)

        return output, logdet

    def generate_inverse(self, size: int):
        assert isinstance(size, int), size

        convMatrix = torch.zeros(size, size, dtype=torch.float64)
        half_filter_size = self.weights_reverse.numel() // 2
        for i in range(size):
            j_beg = i - half_filter_size
            j_end = i + half_filter_size + 1

            j_beg_val = max(j_beg, 0)
            j_end_val = min(j_end, size)

            c_beg = j_beg_val - j_beg
            c_end = j_end_val - j_end + self.weights_reverse.numel()

            convMatrix[i, j_beg_val:j_end_val] = self.weights_reverse[c_beg:c_end]

        convMatrixInv = torch.inverse(convMatrix)

        err = torch.max(torch.abs(convMatrix @ convMatrixInv - torch.eye(size, size, dtype=torch.float64)))
        assert err < 1e-3, err

        return convMatrixInv

    def generate_forward_matrices(self, input):
        if self.weights_forward_h is None or self.weights_forward_h.shape[0] != input.shape[2]:
            self.weights_forward_h = self.generate_inverse(input.shape[2]).float().to(input.device)

        if self.weights_forward_w is None or self.weights_forward_w.shape[0] != input.shape[3]:
            self.weights_forward_w = self.generate_inverse(input.shape[3]).float().to(input.device)


class AffineImageInjector(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers=1, hidden_channels=64, eps=0, f='linear', scale=1,
                 position=None):
        super().__init__()
        if f == 'linear':
            self.NN = self.f_lin(in_channels, out_channels * 2)
        elif f == 'linHiddenZero':
            self.NN = self.f(in_channels, out_channels * 2, hidden_channels, kernel_hidden=1)
        elif f == 'linHiddenZeroK3':
            self.NN = self.f(in_channels, out_channels * 2, hidden_channels, kernel_hidden=3,
                             n_hidden_layers=hidden_layers)
        else:
            raise NotImplementedError()
        self.eps = eps
        self.scale = scale
        self.position = position

    def forward(self, input, ft, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, ft, logdet)
        else:
            return self.reverse_flow(input, ft, logdet)

    def normal_flow(self, z, ft, logdet):
        shift, scale = self.get_coef(ft)

        z = z + shift
        z = z * scale

        logdet = self.get_logdet(scale, logdet)
        return z, logdet

    def reverse_flow(self, z, ft, logdet):
        shift, scale = self.get_coef(ft)

        z = z / scale
        z = z - shift

        logdet = -self.get_logdet(scale, logdet)
        return z, logdet

    @staticmethod
    def get_logdet(scale, logdet):
        logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        return logdet

    def get_coef(self, ft):
        if self.scale > 1:
            ft = F.interpolate(ft, scale_factor=self.scale, mode='bilinear')
        h = self.NN(ft)
        shift, scale = thops.split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2.) + self.eps
        return shift, scale

    @staticmethod
    def f_lin(in_channels, out_channels):
        return nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                            padding=(1, 1), bias=True))

    @staticmethod
    def f(in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class UpsampleAndSqueezeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = 2

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            x = nn.functional.interpolate(input, scale_factor=self.factor)  # Upsample
            output = squeeze2d(x, self.factor)  # Squeeze to original resolution
            return output, logdet
        else:
            x = unsqueeze2d(input, self.factor)
            output = nn.functional.interpolate(x, scale_factor=0.5)
            return output, logdet
