from unittest import TestCase
from MeasureLib import measure
import numpy as np
import imageio

from utils.util import dl_if_gcp


class Test(TestCase):
    def test_measure(self):
        imgA = np.ones([64, 64, 3]).astype(np.uint8) * 127
        imgB = np.ones([64, 64, 3]).astype(np.uint8) * 125
        print(measure(imgA, imgB))

    def test_measure_img(self):
        imgA = imageio.imread(dl_if_gcp("gs://srfl-log/SrgChqRRDBQuantNoise32_K12_L4_sp_CondAffineSeparatedAndCond_FlowNoAff2_n20000_fullDSNoRot_lr1e-4s05s075s09s095_RrdbD05_RRDBStack4/experiments/train/val_images/0/0_000020000_h075_s0.png"))
        imgB = imageio.imread(dl_if_gcp("gs://srfl-log/SrgChqRRDBQuantNoise32_K12_L4_sp_CondAffineSeparatedAndCond_FlowNoAff2_n20000_fullDSNoRot_lr1e-4s05s075s09s095_RrdbD05_RRDBStack4/experiments/train/val_images/0/0_GT.png"))
        print(measure(imgA, imgB))
