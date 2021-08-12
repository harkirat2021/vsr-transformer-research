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

def evaluate(model, eval_dataloader):
    model.eval()

    metrics_sr = MetricsSR(scale=config["SCALE"])

    psnr = 0
    ssim = 0
    num_batches = 0

    with torch.no_grad():
        for x, y in eval_dataloader:
            out = model(x)
            psnr += metrics_sr.PSNR(out, y)
            ssim += metrics_sr.SSIMCustom(out, y)
            num_batches += 1

    return psnr / num_batches, ssim / num_batches
