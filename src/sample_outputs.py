import os
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Get all config values
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

def sample_outputs(model, eval_dataloader):
    model.eval()

    x = eval_dataloader.dataset[:][0]
    y = eval_dataloader.dataset[:][1]

    img_ids = [2400, 4500, 2000, 6000]
    for i in img_ids:
        plt.imshow(np.swapaxes(np.swapaxes(x[i][0][:][:][:], -3, -1), -2, -3))
        plt.show()

        plt.imshow(np.swapaxes(np.swapaxes(model(torch.unsqueeze(x[i], dim=0))[0][0][:][:][:].detach().numpy(), -3, -1), -2, -3))
        plt.show()

        plt.imshow(np.swapaxes(np.swapaxes(y[i][0][:][:][:], -3, -1), -2, -3))
        plt.show()
