from math import ceil, floor

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_msssim import SSIM
import yaml

from src.model_modules import *

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

class MetricsSR():
    def __init__(self, scale, win_size, max_val=1.0, cuda=False):
        self.border = scale
        self.max_val = max_val
        self.color = RGB2YCBCR()
        self.ssim_module = SSIM(data_range=1.0, win_size=win_size, size_average=True, channel=1,
                                nonnegative_ssim=True)

        if cuda:
            self.color = self.color.cuda()
            self.ssim_module = self.ssim_module.cuda()

    def PSNR(self, x, y):
        # TODO Is this important?
        """
        x = self.color(x)[:, 0:1, self.border:-self.border, self.border:-self.border]
        x = x.view(x.size(0), -1)
        y = self.color(y)[:, 0:1, self.border:-self.border, self.border:-self.border]
        y = y.view(y.size(0), -1)
        """

        mse  = ((x-y)**2).mean(dim=1)
        psnr = 10*torch.log10(self.max_val/mse)
        psnr = psnr.mean()

        return psnr

    def SSIMCustom(self, x, y):
        # TODO Is this important?
        """
        x = self.color(x)[:, 0:1, self.border:-self.border, self.border:-self.border]
        y = self.color(y)[:, 0:1, self.border:-self.border, self.border:-self.border]
        """
        
        return self.ssim_module(x, y)