# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from MeasureLib.models import PerceptualLoss
import numpy as np
import torch


def ssim(imgA, imgB):
    score, diff = compare_ssim(imgA, imgB, full=True,
                               multichannel=True)  # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
    return score


def psnr(imgA, imgB):
    psnr = compare_psnr(imgA, imgB)
    return psnr

lpips_model = PerceptualLoss(model='net-lin', net='alex', use_gpu=False)

def lpips(imgA, imgB, model=lpips_model):
    #model = PerceptualLoss(model='net-lin', net='alex', use_gpu=False)
    model.eval()
    with torch.no_grad():
        dist01 = model.forward(t(imgA), t(imgB)).item()
    return dist01


def measure(imgA, imgB, model=lpips_model):
    assert imgA.shape[0] == imgB.shape[0], [imgA.shape, imgB.shape]
    assert imgA.shape[1] == imgB.shape[1], [imgA.shape, imgB.shape]
    assert imgA.shape[2] == imgB.shape[2], [imgA.shape, imgB.shape]
    return [f(imgA, imgB) for f in [psnr, ssim, lambda x,y: lpips(x,y,model)]]


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1
