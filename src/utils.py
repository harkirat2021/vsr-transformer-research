from math import ceil, floor

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_msssim import SSIM

# TODO - orgainize stuff in the utils into different files

class BicubicResize(nn.Module):
    KERNEL_WIDTH = 4.0

    def __init__(self, scale, im_size, cuda=True):
        super(BicubicResize, self).__init__()
        self.input_size = im_size
        self.gpu = cuda
        self.set_scale(scale)

    def set_scale(self, scale):
        self.scale = scale
        self.output_size = self.deriveSizeFromScale(self.input_size)

        self.weights = []
        self.indices = []

        for k in range(2):
            w, ind = self.contributions(self.input_size[k], self.output_size[k],
                                   self.scale[k], self.cubic, self.KERNEL_WIDTH)
            self.weights.append(w)
            self.indices.append(ind)

    def deriveSizeFromScale(self, input_size ):
        output_shape = [int(ceil(self.scale[0] * input_size[0])),
                        int(ceil(self.scale[1] * input_size[1])),
                        ]

        return output_shape

    def cubic(self, x):
        absx = np.abs(x)
        absx2 = absx*absx
        absx3 = absx2*absx
        f = ((1.5*absx3 - 2.5*absx2 + 1)*(absx <= 1)) + ((-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) * (absx <= 2)))

        return f

    def contributions(self, in_length, out_length, scale, kernel, k_width):
        if scale < 1:
            h = lambda x: scale * kernel(scale * x)
            kernel_width = 1.0 * k_width / scale
        else:
            h = kernel
            kernel_width = k_width
        x = np.arange(1, out_length+1).astype(np.float64)
        u = x / scale + 0.5 * (1 - 1 / scale)
        left = np.floor(u - kernel_width / 2)
        P = int(ceil(kernel_width)) + 2
        ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
        indices = ind.astype(np.int32)
        weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
        weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
        aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
        indices = aux[np.mod(indices, aux.size)]
        ind2store = np.nonzero(np.any(weights, axis=0))
        weights = torch.from_numpy(weights[:, ind2store]).float()
        indices = torch.from_numpy(np.squeeze(indices[:, ind2store])).contiguous().long()

        if self.gpu:
            weights = weights.cuda()
            indices = indices.cuda()

        return Variable(weights), Variable(indices)

    def resizeAlongDim(self, inimg, weights, indices, dim):
        w_shape = weights.size()
        idx_size = indices.size()
        indices = indices.view(-1)

        if dim==0:
            im_slice = torch.index_select(inimg, 1, indices)
            im_slice = im_slice.view(im_slice.size(0), idx_size[0], idx_size[1], -1, im_slice.size(3))
            im_slice = torch.transpose(im_slice, 2, 3)
            w = weights.view(1, w_shape[0], 1, 1, -1)
        elif dim==1:
            im_slice = torch.index_select(inimg, 2, indices)
            im_slice = im_slice.view(im_slice.size(0), -1, idx_size[0], idx_size[1], im_slice.size(3))
            w = weights.view(1, 1, w_shape[0], 1, -1)

        outimg = torch.squeeze(torch.matmul(w, im_slice), dim=3)

        return outimg

    def forward(self, x):
        scale = self.scale
        scale_np = np.array(self.scale)
        order = np.argsort(scale_np)

        if self.train:
            weights = self.weights
            indices = self.indices
        else:
            weights = []
            indices = []

            for k in range(2):
                input_size = x.size()[2:]
                output_size = self.deriveSizeFromScale(input_size)
                w, ind = self.contributions(input_size[k], output_size[k],
                                   self.scale[k], self.cubic, self.KERNEL_WIDTH)
                weights.append(w)
                indices.append(ind)

        x = torch.squeeze(torch.transpose(torch.unsqueeze(x, dim=4), 1, 4), dim=1)

        for k in range(2):
            dim = order[k]
            x = self.resizeAlongDim(x, weights[dim], indices[dim], dim)

        x = torch.squeeze(torch.transpose(torch.unsqueeze(x, dim=1), 1, 4), dim=4)

        return x


class RGB2YCBCR(nn.Module):
    def __init__(self):
        super(RGB2YCBCR, self).__init__()
        ycbcr_from_rgb = torch.from_numpy(np.array([[    65.481,   128.553,    24.966],
                            [   -37.797,   -74.203,   112.0  ],
                            [   112.0  ,   -93.786,   -18.214]])).float().view(3, 3)*(1/255.0)
        sum_const = torch.from_numpy(np.array([    16,   128,    128])).float().view(1, 3, 1, 1)*(1/255.0)
        self.register_buffer('ycbcr_from_rgb', ycbcr_from_rgb)
        self.register_buffer('sum_const', sum_const)

    def forward(self, x):
        or_size = x.size()
        x = torch.matmul(self.ycbcr_from_rgb, x.view(x.size(0), 3, -1))
        x = x.view(or_size)+self.sum_const

        return x


class MetricsSR():
    def __init__(self, scale, max_val=1.0, cuda=False):
        self.border      = scale
        self.max_val     = max_val
        self.color       = RGB2YCBCR()
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1,
                                nonnegative_ssim=True)

        if cuda:
            self.color       = self.color.cuda()
            self.ssim_module = self.ssim_module.cuda()

    def PSNR(self, x, y):
        x = self.color(x)[:, 0:1, self.border:-self.border, self.border:-self.border]
        x = x.view(x.size(0), -1)
        y = self.color(y)[:, 0:1, self.border:-self.border, self.border:-self.border]
        y = y.view(y.size(0), -1)

        mse  = ((x-y)**2).mean(dim=1)
        psnr = 10*torch.log10(self.max_val/mse)
        psnr = pnsr.mean()

        return psnr

    def SSIMCustom(self, x, y):
        x = self.color(x)[:, 0:1, self.border:-self.border, self.border:-self.border]
        y = self.color(y)[:, 0:1, self.border:-self.border, self.border:-self.border]

        return self.ssim_module(x, y)