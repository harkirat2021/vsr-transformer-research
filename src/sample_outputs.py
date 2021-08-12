import os
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluated(model, data_module):
    model.eval()

    eval_dataloader = data_module.val_dataloader()
    metrics_sr = MetricsSR(scale=2)

    psnr = 0
    ssim = 0
    num_batches = 0

    with torch.no_grad():
        for x, y in eval_dataloader:
            out = model.forward(x, model.src_mask)
            psnr += metrics_sr.PSNR(x, y)
            ssim += metrics_sr.SSIMCustom(x, y)
            num_batches += 1

    return psnr / num_batches, ssim / num_batches
