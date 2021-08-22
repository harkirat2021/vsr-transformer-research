import os
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import *
from src.metrics import *

# Get all config values
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

def evaluate(model, data, eval_dataloader):
    model.eval()

    metrics_sr = MetricsSR(scale=config[data]["SCALE"], win_size=config[data]["HR_PATCH_SHAPE"][0]+1)

    psnr = 0
    ssim = 0
    b = 0
    num_batches = len(eval_dataloader)

    with torch.no_grad():
        for x, y in eval_dataloader:
            b += 1
            if b % len(eval_dataloader) // 10 == 0:
                print("{} / {}".format(b, len(eval_dataloader)))
            out = model(x)
            out = out[:,config[data]["SEQ_LEN"]//2,:,:,:] # Only keep middle frame
            y = y[:,0,:,:,:] # Use only dim
            psnr += metrics_sr.PSNR(out, y) / num_batches
            #ssim += metrics_sr.SSIMCustom(out, y) / num_batches

    return psnr, ssim
