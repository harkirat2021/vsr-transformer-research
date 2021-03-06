import os
import numpy as np
import yaml

import torch
import torch.nn as nn

from src.models import *

# Get all config values
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

""" Run training """
def train(model, data_module, experiment_dir, max_epochs, gpus, save=True, experiment_name=""):
    # Init logger
    tb_logger = pl.loggers.TensorBoardLogger(config["EXPERIMENT_SAVE_DIR"], name=os.path.relpath(experiment_dir, config["EXPERIMENT_SAVE_DIR"]))

    # Train
    trainer = pl.Trainer(logger=(tb_logger if save else False), gpus=gpus, max_epochs=max_epochs, checkpoint_callback=save, check_val_every_n_epoch=config["CHECKPOINT_VALID_EVERY_N_EPOCH"])
    trainer.fit(model, data_module)

    return model