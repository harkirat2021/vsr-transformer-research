import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import *

""" Run training """
def train(data_module, seq_len, gpus):
    print("Training...")

    # Init model
    model = VSRTE1(c=3, h=36, w=64, embed_dim=8, n_head=4, h_dim=20, n_layers=2, dropout=0.5)
    model.set_src_mask(seq_len)

    # Train
    trainer = pl.Trainer(gpus=gpus, max_epochs=5)
    trainer.fit(model, data_module)

    print("Done")

    return trainer, model